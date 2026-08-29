import torch
import torch.nn.functional as F
from torch import Tensor, nn
import re

def compute_loss_per_sample(model, batch, device, ignored_token_ids):
    """
    核心 Loss 计算 (优化版)：
    直接修改 labels 为 -100 来屏蔽 loss。
    """
    # 确保 ignored_token_ids 是 Tensor 且在正确的设备上
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits.float()

    # 1. 进行错位
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous().clone()  # clone 一份，避免修改原始数据

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        mask_to_ignore = torch.isin(shift_labels, ignored_token_ids)
        shift_labels[mask_to_ignore] = -100

    # 4. 计算 Loss
    # reduction='none' 确保返回的是每个 token 的 loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    # 计算出来的 token_losses 在被忽略的位置上已经是 0 了
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

    # 5. 计算有效 token 数量 (即 label 不为 -100 的位置)
    valid_mask = shift_labels.ne(-100).float()
    num_valid  = valid_mask.sum(dim=1)

    # 6. 计算 sum 和 mean loss
    sum_loss = token_losses.sum(dim=1)

    # 避免除以 0
    mean_loss = sum_loss / (num_valid + 1e-9)

    return mean_loss, token_losses


def compute_gradients(
        model,
        batch,
        param_filter_fn,
        device,
        ignored_token_ids
):
    model.eval()
    model.zero_grad(set_to_none=True)

    # Explicitly re-enable requires_grad for filtered params.
    params = []
    for name, param in model.named_parameters():
        if param_filter_fn is None or param_filter_fn(name, param):
            param.requires_grad_(True)
            params.append(param)

    if not params:
        raise RuntimeError(
            "compute_gradients: no parameters matched param_filter_fn. "
            "Check that the filter is correct and the model has matching layers."
        )

    # Use torch.enable_grad() rather than torch.set_grad_enabled(True):
    # enable_grad() works even when called from inside a torch.no_grad() scope,
    # guaranteeing the forward pass builds a computation graph.
    with torch.enable_grad():
        mean_loss, _ = compute_loss_per_sample(model, batch, device, ignored_token_ids)
        loss = mean_loss.mean()
        if loss.numel() > 1:
            loss = loss.mean()

        grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
    return list(grads)


@torch.no_grad()
def compute_lm_head_ce_gradient_no_backward(
    model,
    batch,
    device,
    ignored_token_ids,
) -> Tensor:
    """Compute d(CE)/d(lm_head.weight) from hidden states without backprop.

    The coarse screening path only compares LM-head CE gradients. For a causal
    LM this gradient is exactly `(softmax(logits) - one_hot(label)) outer h`,
    so we can avoid a full backward through the transformer for every sample.
    """
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
    labels = inputs["labels"]

    base_model = getattr(model, "model", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if base_model is None or lm_head is None:
        raise RuntimeError("Expected a HuggingFace causal LM with .model and output embeddings.")

    outputs = base_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state

    shift_hidden = hidden[..., :-1, :]
    shift_labels = labels[..., 1:].clone().to(shift_hidden.device)

    if ignored_token_ids is not None and ignored_token_ids.numel() > 0:
        ignored_on_label_device = ignored_token_ids.to(shift_labels.device)
        shift_labels[torch.isin(shift_labels, ignored_on_label_device)] = -100

    valid_mask = shift_labels.ne(-100)
    head_device = lm_head.weight.device
    if not bool(valid_mask.any().item()):
        return torch.zeros_like(lm_head.weight, device=head_device)

    valid_hidden = shift_hidden[valid_mask].to(head_device)
    valid_labels = shift_labels[valid_mask].to(head_device)

    logits = lm_head(valid_hidden).float()
    grad_logits = torch.softmax(logits, dim=-1)
    grad_logits[torch.arange(valid_labels.numel(), device=head_device), valid_labels] -= 1.0
    grad_logits /= valid_labels.numel()

    grad = grad_logits.t().to(valid_hidden.dtype).matmul(valid_hidden)
    return grad.to(dtype=lm_head.weight.dtype)


@torch.no_grad()
def compute_lm_head_ce_gradient_scores_no_backward(
    model,
    batch,
    device,
    ignored_token_ids,
    test_ce_grad: Tensor,
    score_device,
) -> list[float]:
    """Score each sample's analytic LM-head CE gradient against test_ce_grad.

    This runs the transformer once for a whole batch and then computes each
    sample's LM-head gradient/cosine score separately, avoiding both backward
    and storing a full [batch, vocab, hidden] gradient tensor.
    """
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
    labels = inputs["labels"]

    base_model = getattr(model, "model", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if base_model is None or lm_head is None:
        raise RuntimeError("Expected a HuggingFace causal LM with .model and output embeddings.")

    outputs = base_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    shift_hidden = hidden[..., :-1, :]
    shift_labels = labels[..., 1:].clone().to(shift_hidden.device)

    if ignored_token_ids is not None and ignored_token_ids.numel() > 0:
        ignored_on_label_device = ignored_token_ids.to(shift_labels.device)
        shift_labels[torch.isin(shift_labels, ignored_on_label_device)] = -100

    head_device = lm_head.weight.device
    test_flat = test_ce_grad.to(score_device)
    scores: list[float] = []

    for sample_idx in range(shift_labels.size(0)):
        valid_mask = shift_labels[sample_idx].ne(-100)
        if not bool(valid_mask.any().item()):
            scores.append(float("-inf"))
            continue

        valid_hidden = shift_hidden[sample_idx][valid_mask].to(head_device)
        valid_labels = shift_labels[sample_idx][valid_mask].to(head_device)

        logits = lm_head(valid_hidden).float()
        grad_logits = torch.softmax(logits, dim=-1)
        grad_logits[torch.arange(valid_labels.numel(), device=head_device), valid_labels] -= 1.0
        grad_logits /= valid_labels.numel()

        grad = grad_logits.t().to(valid_hidden.dtype).matmul(valid_hidden)
        score = F.cosine_similarity(test_flat, grad.reshape(-1).to(score_device), dim=0).item()
        scores.append(score)

        del valid_hidden, valid_labels, logits, grad_logits, grad

    del outputs, hidden, shift_hidden, shift_labels
    torch.cuda.empty_cache()
    return scores


_COUNTSKETCH_CACHE: dict[tuple[int, int, int], tuple[Tensor, Tensor]] = {}


def _get_countsketch_hashes(
    size: int,
    sketch_dim: int,
    seed: int,
    device,
) -> tuple[Tensor, Tensor]:
    """Deterministic CountSketch hash/sign vectors, cached on CPU then moved."""
    key = (int(size), int(sketch_dim), int(seed))
    if key not in _COUNTSKETCH_CACHE:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        hashes = torch.randint(sketch_dim, (size,), generator=gen, dtype=torch.long)
        signs = torch.randint(2, (size,), generator=gen, dtype=torch.int8)
        signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
        _COUNTSKETCH_CACHE[key] = (hashes, signs)

    hashes, signs = _COUNTSKETCH_CACHE[key]
    return hashes.to(device), signs.to(device)


def _countsketch_rows(x: Tensor, sketch_dim: int, seed: int) -> Tensor:
    """CountSketch each row of x from [rows, dim] to [rows, sketch_dim]."""
    if x.dim() != 2:
        raise ValueError(f"_countsketch_rows expects [rows, dim], got {tuple(x.shape)}")
    hashes, signs = _get_countsketch_hashes(x.size(1), sketch_dim, seed, x.device)
    out = torch.zeros((x.size(0), sketch_dim), dtype=torch.float32, device=x.device)
    index = hashes.unsqueeze(0).expand(x.size(0), -1)
    values = x.float() * signs.unsqueeze(0)
    out.scatter_add_(1, index, values)
    return out


def _tensor_sketch_lm_head_gradient(
    grad_logits: Tensor,
    hidden: Tensor,
    *,
    sketch_dim: int,
    sketch_seed: int,
) -> Tensor:
    """
    TensorSketch approximation for vec(sum_t grad_logits_t outer hidden_t).

    CountSketch(a outer b) is the circular convolution of CountSketch(a) and
    CountSketch(b), which preserves inner products in expectation while avoiding
    materializing the vocab_size x hidden_size LM-head gradient.
    """
    if grad_logits.size(0) == 0:
        return torch.zeros(sketch_dim, dtype=torch.float32, device=hidden.device)

    vocab_sketch = _countsketch_rows(grad_logits, sketch_dim, sketch_seed)
    hidden_sketch = _countsketch_rows(hidden, sketch_dim, sketch_seed + 1)
    prod_fft = torch.fft.rfft(vocab_sketch, n=sketch_dim) * torch.fft.rfft(hidden_sketch, n=sketch_dim)
    token_sketches = torch.fft.irfft(prod_fft, n=sketch_dim)
    return token_sketches.mean(dim=0)


@torch.no_grad()
def compute_lm_head_ce_gradient_sketches_no_backward(
    model,
    batch,
    device,
    ignored_token_ids,
    *,
    sketch_dim: int = 8192,
    sketch_seed: int = 42,
) -> Tensor:
    """Compute low-dimensional TensorSketches of LM-head CE gradients.

    Returns one normalized sketch per sample with shape [batch, sketch_dim].
    These sketches are intended for fast coarse pre-screen retrieval/cache.
    """
    if sketch_dim <= 0:
        raise ValueError("sketch_dim must be positive.")
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
    labels = inputs["labels"]

    base_model = getattr(model, "model", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if base_model is None or lm_head is None:
        raise RuntimeError("Expected a HuggingFace causal LM with .model and output embeddings.")

    outputs = base_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    shift_hidden = hidden[..., :-1, :]
    shift_labels = labels[..., 1:].clone().to(shift_hidden.device)

    if ignored_token_ids is not None and ignored_token_ids.numel() > 0:
        ignored_on_label_device = ignored_token_ids.to(shift_labels.device)
        shift_labels[torch.isin(shift_labels, ignored_on_label_device)] = -100

    head_device = lm_head.weight.device
    sketches = []
    for sample_idx in range(shift_labels.size(0)):
        valid_mask = shift_labels[sample_idx].ne(-100)
        if not bool(valid_mask.any().item()):
            sketches.append(torch.zeros(sketch_dim, dtype=torch.float32, device=head_device))
            continue

        valid_hidden = shift_hidden[sample_idx][valid_mask].to(head_device)
        valid_labels = shift_labels[sample_idx][valid_mask].to(head_device)

        logits = lm_head(valid_hidden).float()
        grad_logits = torch.softmax(logits, dim=-1)
        grad_logits[torch.arange(valid_labels.numel(), device=head_device), valid_labels] -= 1.0

        sketch = _tensor_sketch_lm_head_gradient(
            grad_logits,
            valid_hidden.float(),
            sketch_dim=sketch_dim,
            sketch_seed=sketch_seed,
        )
        sketch = F.normalize(sketch, dim=0, eps=1e-12)
        sketches.append(sketch.detach())

        del valid_hidden, valid_labels, logits, grad_logits, sketch

    del outputs, hidden, shift_hidden, shift_labels
    torch.cuda.empty_cache()
    return torch.stack(sketches, dim=0)


def _repeat_kv_for_alti(value_states: Tensor, num_attention_heads: int) -> Tensor:
    """
    Expand grouped-query value states from [B, H_kv, S, D] to [B, H, S, D].
    Qwen2 uses GQA/MQA in some sizes, while attention probabilities are already
    expanded to the query-head count.
    """
    num_kv_heads = value_states.size(1)
    if num_kv_heads == num_attention_heads:
        return value_states
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot repeat {num_kv_heads} KV heads to {num_attention_heads} attention heads."
        )
    n_rep = num_attention_heads // num_kv_heads
    bsz, _, seq_len, head_dim = value_states.shape
    return (
        value_states[:, :, None, :, :]
        .expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, num_attention_heads, seq_len, head_dim)
    )


def _infer_qwen_attention_layout(model, self_attn, attention_probs: Tensor) -> tuple[int, int, int]:
    """Infer (num_attention_heads, head_dim, num_key_value_heads) from config/weights.

    Some Qwen checkpoints do not expose `self_attn.num_key_value_heads`. For
    GQA/MQA models such as Qwen 1.5B, falling back to num_attention_heads is
    wrong because `v_proj.out_features = num_key_value_heads * head_dim`.
    """
    config = getattr(model, "config", None)
    num_heads = int(attention_probs.size(1))

    head_dim = getattr(self_attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        q_out = getattr(self_attn.q_proj, "out_features", self_attn.q_proj.weight.shape[0])
        if q_out % num_heads != 0:
            raise ValueError(
                f"Cannot infer head_dim: q_proj out_features={q_out}, num_heads={num_heads}."
            )
        head_dim = q_out // num_heads
    head_dim = int(head_dim)

    v_out = getattr(self_attn.v_proj, "out_features", self_attn.v_proj.weight.shape[0])
    if v_out % head_dim != 0:
        cfg_kv_heads = getattr(config, "num_key_value_heads", None) if config is not None else None
        if cfg_kv_heads is None:
            raise ValueError(
                f"Cannot infer num_key_value_heads: v_proj out_features={v_out}, head_dim={head_dim}."
            )
        num_kv_heads = int(cfg_kv_heads)
    else:
        num_kv_heads = int(v_out // head_dim)

    o_in = self_attn.o_proj.weight.shape[1]
    if o_in != num_heads * head_dim:
        raise ValueError(
            f"Unexpected o_proj in_features={o_in}; expected num_heads({num_heads}) * head_dim({head_dim})."
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads={num_heads} must be divisible by num_key_value_heads={num_kv_heads}."
        )

    return num_heads, head_dim, num_kv_heads


def _normalize_alti_importance(
    source_vectors: Tensor,
    *,
    p: int = 1,
    eps: float = 1e-9,
) -> Tensor:
    """
    Convert ALTI source contribution vectors T_i(x_j) into row-stochastic scalar
    weights. This follows the paper implementation's min_sum normalization:

        max(||y_i||_p - ||T_i(x_j) - y_i||_p, 0), normalized over j

    where y_i is the reconstructed attention-block output for target position i.
    """
    resultant = source_vectors.sum(dim=1)
    if p == 1:
        resultant_norm = resultant.abs().sum(dim=-1, keepdim=True)
        distances = torch.zeros(
            source_vectors.shape[:-1],
            dtype=source_vectors.dtype,
            device=source_vectors.device,
        )
        hidden_chunk = 2048
        for start in range(0, source_vectors.size(-1), hidden_chunk):
            end = min(start + hidden_chunk, source_vectors.size(-1))
            distances += (
                source_vectors[..., start:end] - resultant[:, None, start:end]
            ).abs().sum(dim=-1)
    else:
        resultant_norm = torch.linalg.vector_norm(resultant, ord=p, dim=-1, keepdim=True)
        distances = torch.linalg.vector_norm(source_vectors - resultant[:, None, :], ord=p, dim=-1)
    scores = torch.clamp(resultant_norm - distances, min=0.0)

    denom = scores.sum(dim=-1, keepdim=True)
    if torch.all(denom > eps):
        return scores / denom.clamp_min(eps)

    # Rare numerical fallback: if distance-based scores are all zero for a row,
    # fall back to contribution vector norms so the rollout remains well-defined.
    norm_scores = torch.linalg.vector_norm(source_vectors, ord=p, dim=-1)
    norm_denom = norm_scores.sum(dim=-1, keepdim=True)
    normalized = norm_scores / norm_denom.clamp_min(eps)

    zero_rows = denom <= eps
    if torch.any(zero_rows):
        uniform = torch.full_like(scores, 1.0 / max(scores.size(-1), 1))
        normalized = torch.where((norm_denom <= eps) & zero_rows, uniform, normalized)
        scores = torch.where(zero_rows, normalized, scores / denom.clamp_min(eps))
    return scores


@torch.no_grad()
def _compute_qwen_alti_layer_matrix(
    model,
    layer_idx: int,
    hidden_states: Tensor,
    attention_probs: Tensor,
    *,
    p: int = 1,
    chunk_size: int = 8,
) -> Tensor:
    """
    Compute one Qwen decoder layer's ALTI token-to-token contribution matrix.

    Rows are target/query positions and columns are source/key positions. The
    matrix is row-stochastic and includes the self residual contribution.
    FFN blocks are position-wise, so they do not introduce token mixing; this
    matrix tracks the attention block's mixing in the residual stream.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("compute_alti_saliency_vector currently expects a Qwen-style model.model.layers stack.")

    layer = model.model.layers[layer_idx]
    self_attn = layer.self_attn

    device = self_attn.v_proj.weight.device
    hidden_states = hidden_states.to(device)
    attention_probs = attention_probs.to(device)

    if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
        raise ValueError("ALTI saliency currently supports batch_size=1.")
    if attention_probs.dim() != 4 or attention_probs.size(0) != 1:
        raise ValueError("Expected attention_probs with shape [1, heads, seq, seq].")

    bsz, seq_len, hidden_dim = hidden_states.shape
    num_heads, head_dim, num_kv_heads = _infer_qwen_attention_layout(
        model,
        self_attn,
        attention_probs,
    )

    normed_states = layer.input_layernorm(hidden_states)
    value_states = self_attn.v_proj(normed_states)
    value_states = value_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = _repeat_kv_for_alti(value_states, num_heads)

    out_weight = self_attn.o_proj.weight.to(device)
    out_dim, in_dim = out_weight.shape
    if in_dim != num_heads * head_dim:
        raise ValueError(
            f"Unexpected o_proj shape {tuple(out_weight.shape)} for {num_heads} heads × {head_dim}."
        )
    out_weight_by_head = out_weight.view(out_dim, num_heads, head_dim)

    # Per-head value vectors after the corresponding slice of W_O:
    # [heads, source, hidden_dim].
    transformed_values = torch.einsum(
        "bhsd,ohd->bhso",
        value_states,
        out_weight_by_head,
    )[0].float()

    attention_probs = attention_probs[0].float()
    residual_states = hidden_states[0].float()

    contribution_rows = []
    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        attn_chunk = attention_probs[:, q_start:q_end, :]  # [heads, q_chunk, source]

        source_vectors = torch.einsum(
            "hqs,hso->qso",
            attn_chunk,
            transformed_values,
        )

        q_positions = torch.arange(q_start, q_end, device=device)
        local_rows = torch.arange(q_end - q_start, device=device)
        source_vectors[local_rows, q_positions, :] += residual_states[q_positions]

        contribution_rows.append(_normalize_alti_importance(source_vectors, p=p))

        del attn_chunk, source_vectors

    return torch.cat(contribution_rows, dim=0)


def _compute_qwen_alti_layer_relevance(
    model,
    layer_idx: int,
    hidden_states: Tensor,
    attention_probs: Tensor,
    prev_relevance: Tensor,
    *,
    p: int = 1,
    chunk_size: int = 8,
) -> Tensor:
    """
    Compute C_l @ prev_relevance for one Qwen decoder layer without materializing
    the full rollout. This is the differentiable counterpart used for pair-level
    ALTI gradients.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("compute_alti_correlation_gradient currently expects a Qwen-style model.model.layers stack.")

    layer = model.model.layers[layer_idx]
    self_attn = layer.self_attn

    device = self_attn.v_proj.weight.device
    hidden_states = hidden_states.to(device)
    attention_probs = attention_probs.to(device)
    prev_relevance = prev_relevance.to(device=device, dtype=torch.float32)

    if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
        raise ValueError("ALTI correlation gradients currently support batch_size=1.")
    if attention_probs.dim() != 4 or attention_probs.size(0) != 1:
        raise ValueError("Expected attention_probs with shape [1, heads, seq, seq].")

    bsz, seq_len, hidden_dim = hidden_states.shape
    if prev_relevance.numel() != seq_len:
        raise ValueError(
            f"prev_relevance length {prev_relevance.numel()} does not match sequence length {seq_len}."
        )

    num_heads, head_dim, num_kv_heads = _infer_qwen_attention_layout(
        model,
        self_attn,
        attention_probs,
    )

    normed_states = layer.input_layernorm(hidden_states)
    value_states = self_attn.v_proj(normed_states)
    value_states = value_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = _repeat_kv_for_alti(value_states, num_heads)

    out_weight = self_attn.o_proj.weight.to(device)
    out_dim, in_dim = out_weight.shape
    if in_dim != num_heads * head_dim:
        raise ValueError(
            f"Unexpected o_proj shape {tuple(out_weight.shape)} for {num_heads} heads × {head_dim}."
        )
    out_weight_by_head = out_weight.view(out_dim, num_heads, head_dim)

    transformed_values = torch.einsum(
        "bhsd,ohd->bhso",
        value_states,
        out_weight_by_head,
    )[0].float()

    attention_probs = attention_probs[0].float()
    residual_states = hidden_states[0].float()

    next_relevance_chunks = []
    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        attn_chunk = attention_probs[:, q_start:q_end, :]

        source_vectors = torch.einsum(
            "hqs,hso->qso",
            attn_chunk,
            transformed_values,
        )

        q_positions = torch.arange(q_start, q_end, device=device)
        local_rows = torch.arange(q_end - q_start, device=device)
        source_vectors[local_rows, q_positions, :] += residual_states[q_positions]

        contribution_chunk = _normalize_alti_importance(source_vectors, p=p)
        next_relevance_chunks.append(torch.matmul(contribution_chunk, prev_relevance))

        del attn_chunk, source_vectors, contribution_chunk

    return torch.cat(next_relevance_chunks, dim=0)


def _compute_qwen_alti_layer_target_relevance(
    model,
    layer_idx: int,
    hidden_states: Tensor,
    attention_probs: Tensor,
    prev_relevance: Tensor,
    query_idx: int,
    *,
    p: int = 1,
) -> Tensor:
    """
    Compute one row of C_l @ prev_relevance for a single target query.

    This is the memory-critical fast path for last-layer-only matching: the
    final score needs only the target row, not all query rows in the sequence.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("compute_alti_correlation_gradient currently expects a Qwen-style model.model.layers stack.")

    layer = model.model.layers[layer_idx]
    self_attn = layer.self_attn

    device = self_attn.v_proj.weight.device
    hidden_states = hidden_states.to(device)
    attention_probs = attention_probs.to(device)
    prev_relevance = prev_relevance.to(device=device, dtype=torch.float32)

    if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
        raise ValueError("ALTI correlation gradients currently support batch_size=1.")
    if attention_probs.dim() != 4 or attention_probs.size(0) != 1:
        raise ValueError("Expected attention_probs with shape [1, heads, seq, seq].")

    bsz, seq_len, hidden_dim = hidden_states.shape
    if query_idx < 0 or query_idx >= seq_len:
        raise ValueError(f"query_idx={query_idx} is outside [0, {seq_len}).")
    if prev_relevance.numel() != seq_len:
        raise ValueError(
            f"prev_relevance length {prev_relevance.numel()} does not match sequence length {seq_len}."
        )

    num_heads, head_dim, num_kv_heads = _infer_qwen_attention_layout(
        model,
        self_attn,
        attention_probs,
    )

    normed_states = layer.input_layernorm(hidden_states)
    value_states = self_attn.v_proj(normed_states)
    value_states = value_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = _repeat_kv_for_alti(value_states, num_heads)

    out_weight = self_attn.o_proj.weight.to(device)
    out_dim, in_dim = out_weight.shape
    if in_dim != num_heads * head_dim:
        raise ValueError(
            f"Unexpected o_proj shape {tuple(out_weight.shape)} for {num_heads} heads × {head_dim}."
        )
    out_weight_by_head = out_weight.view(out_dim, num_heads, head_dim)

    transformed_values = torch.einsum(
        "bhsd,ohd->bhso",
        value_states,
        out_weight_by_head,
    )[0].float()

    attn_row = attention_probs[0, :, query_idx, :].float()
    source_vectors = torch.einsum(
        "hs,hso->so",
        attn_row,
        transformed_values,
    )
    source_vectors[query_idx, :] += hidden_states[0, query_idx].float()

    contribution_row = _normalize_alti_importance(source_vectors.unsqueeze(0), p=p)[0]
    return torch.dot(contribution_row, prev_relevance)


@torch.no_grad()
def compute_alti_saliency_vector(
    model,
    batch,
    target_idx_in_seq: int,
    *,
    p: int = 1,
    chunk_size: int = 8,
) -> list[float]:
    """
    Forward-only ALTI token-to-token saliency for Qwen-style causal LMs.

    `target_idx_in_seq` is the sequence index of the token being predicted. The
    model is run on input_ids[:, :target_idx_in_seq], and the returned vector
    has length target_idx_in_seq. Entry j is the rollout contribution from
    source token j to the final prefix position target_idx_in_seq - 1, whose
    hidden state predicts token target_idx_in_seq.

    This avoids embedding gradients and second-order derivatives. It uses each
    layer's attention probabilities, V projection, O projection, and residual
    stream to build ALTI contribution matrices, then rolls them out across
    layers by matrix multiplication.
    """
    if target_idx_in_seq <= 0:
        raise ValueError("target_idx_in_seq must be > 0 because it denotes the next token to predict.")

    model.eval()
    device = model.device

    input_ids = batch["input_ids"][:, :target_idx_in_seq].to(device)
    inputs = {"input_ids": input_ids}
    if "attention_mask" in batch:
        inputs["attention_mask"] = batch["attention_mask"][:, :target_idx_in_seq].to(device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    if hidden_states is None or attentions is None:
        raise RuntimeError(
            "Model did not return hidden_states/attentions. Ensure output_hidden_states and "
            "output_attentions are supported; Qwen may need attn_implementation='eager'."
        )

    rollout = None
    num_layers = len(attentions)
    for layer_idx in range(num_layers):
        if attentions[layer_idx] is None:
            raise RuntimeError("Encountered None attention tensor; use eager attention when computing ALTI.")

        layer_contrib = _compute_qwen_alti_layer_matrix(
            model,
            layer_idx,
            hidden_states[layer_idx],
            attentions[layer_idx],
            p=p,
            chunk_size=chunk_size,
        )
        rollout = layer_contrib if rollout is None else torch.matmul(layer_contrib, rollout.to(layer_contrib.device))

        del layer_contrib

    if rollout is None:
        raise RuntimeError("No transformer layers were found while computing ALTI saliency.")

    query_pos = target_idx_in_seq - 1
    result = rollout[query_pos, :target_idx_in_seq].detach().cpu().tolist()

    del outputs, hidden_states, attentions, rollout
    torch.cuda.empty_cache()

    return result


def _selected_layer_start_from_filter(model, param_filter_fn) -> int:
    if param_filter_fn is None or not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return 0

    selected_layers = []
    for name, param in model.named_parameters():
        if param_filter_fn(name, param):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match:
                selected_layers.append(int(match.group(1)))

    return min(selected_layers) if selected_layers else 0


def _selected_layers_from_filter(model, param_filter_fn) -> list[int]:
    if param_filter_fn is None or not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return []

    selected_layers = set()
    for name, param in model.named_parameters():
        if param_filter_fn(name, param):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match:
                selected_layers.add(int(match.group(1)))

    return sorted(selected_layers)


def compute_alti_correlation_gradient(
    model,
    batch,
    target_idx_in_seq: int,
    source_idx_in_seq: int,
    param_filter_fn,
    *,
    device=None,
    p: int = 1,
    chunk_size: int = 8,
    return_score: bool = False,
):
    """
    Compute a first-order parameter-space feature for one ALTI correlation pair:

        ∇_θ ALTI(source_idx_in_seq -> target_idx_in_seq)

    `target_idx_in_seq` is the token being predicted, so the model runs on
    prefix [:target_idx_in_seq] and the final query position is
    target_idx_in_seq - 1.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Disable torch.inference_mode() before calling this function.")
    if target_idx_in_seq <= 0:
        raise ValueError("target_idx_in_seq must be > 0.")
    if source_idx_in_seq < 0 or source_idx_in_seq >= target_idx_in_seq:
        raise ValueError(
            f"source_idx_in_seq={source_idx_in_seq} must be in [0, {target_idx_in_seq})."
        )

    model.eval()
    model.zero_grad(set_to_none=True)

    target_params = []
    original_flags = []
    for name, param in model.named_parameters():
        selected = param_filter_fn is None or param_filter_fn(name, param)
        original_flags.append((param, param.requires_grad))
        param.requires_grad_(selected)
        if selected:
            target_params.append(param)

    if not target_params:
        for param, flag in original_flags:
            param.requires_grad_(flag)
        raise RuntimeError("compute_alti_correlation_gradient: no parameters matched param_filter_fn.")

    device = device or model.device
    input_ids = batch["input_ids"][:, :target_idx_in_seq].to(device)
    inputs = {"input_ids": input_ids}
    if "attention_mask" in batch:
        inputs["attention_mask"] = batch["attention_mask"][:, :target_idx_in_seq].to(device)

    grad_start_layer = _selected_layer_start_from_filter(model, param_filter_fn)
    selected_layers = _selected_layers_from_filter(model, param_filter_fn)

    try:
        with torch.enable_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

            hidden_states = list(outputs.hidden_states or [])
            attentions = list(outputs.attentions or [])
            del outputs
            if not hidden_states or not attentions:
                raise RuntimeError(
                    "Model did not return hidden_states/attentions. Use eager attention for ALTI gradients."
                )

            seq_len = input_ids.size(1)
            relevance = torch.zeros(seq_len, dtype=torch.float32, device=device)
            relevance[source_idx_in_seq] = 1.0
            last_layer_idx = len(attentions) - 1

            if selected_layers == [last_layer_idx]:
                for layer_idx in range(last_layer_idx):
                    if attentions[layer_idx] is None:
                        raise RuntimeError("Encountered None attention tensor; use eager attention for ALTI gradients.")
                    with torch.no_grad():
                        relevance = _compute_qwen_alti_layer_relevance(
                            model,
                            layer_idx,
                            hidden_states[layer_idx].detach(),
                            attentions[layer_idx].detach(),
                            relevance.detach(),
                            p=p,
                            chunk_size=chunk_size,
                        ).detach()
                    hidden_states[layer_idx] = None
                    attentions[layer_idx] = None

                if attentions[last_layer_idx] is None:
                    raise RuntimeError("Encountered None attention tensor; use eager attention for ALTI gradients.")

                alti_score = _compute_qwen_alti_layer_target_relevance(
                    model,
                    last_layer_idx,
                    hidden_states[last_layer_idx],
                    attentions[last_layer_idx],
                    relevance,
                    target_idx_in_seq - 1,
                    p=p,
                )
                hidden_states[last_layer_idx] = None
                attentions[last_layer_idx] = None
            else:
                for layer_idx in range(len(attentions)):
                    if attentions[layer_idx] is None:
                        raise RuntimeError("Encountered None attention tensor; use eager attention for ALTI gradients.")

                    if layer_idx < grad_start_layer:
                        with torch.no_grad():
                            relevance = _compute_qwen_alti_layer_relevance(
                                model,
                                layer_idx,
                                hidden_states[layer_idx].detach(),
                                attentions[layer_idx].detach(),
                                relevance.detach(),
                                p=p,
                                chunk_size=chunk_size,
                            ).detach()
                    else:
                        relevance = _compute_qwen_alti_layer_relevance(
                            model,
                            layer_idx,
                            hidden_states[layer_idx],
                            attentions[layer_idx],
                            relevance,
                            p=p,
                            chunk_size=chunk_size,
                        )

                    hidden_states[layer_idx] = None
                    attentions[layer_idx] = None

                alti_score = relevance[target_idx_in_seq - 1]

            grads = torch.autograd.grad(
                alti_score,
                target_params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

        flat_grad = torch.cat([
            g.reshape(-1).detach().cpu().float()
            if g is not None else torch.zeros(p.numel(), dtype=torch.float32)
            for g, p in zip(grads, target_params)
        ])
        score_value = float(alti_score.detach().cpu().item())
        del hidden_states, attentions, relevance

    finally:
        for param, flag in original_flags:
            param.requires_grad_(flag)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    if return_score:
        return flat_grad, score_value
    return flat_grad
