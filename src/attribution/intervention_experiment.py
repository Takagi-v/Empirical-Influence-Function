from __future__ import annotations

import os
import json
import torch
import torch.nn.functional as F
import hashlib
from functools import partial
from heapq import nlargest
from accelerate import Accelerator
from tqdm import tqdm

from src.attribution.NIF import (
    load_model_and_tokenizer,
    load_samples_from_formal_jsonl,
    build_train_dataset,
    build_single_sample_dataset,
    NewInferenceFunction,
    DatasetWrapper,
    CustomCollator,
    round_floats,
    _find_subseq_start,
)
from src.attribution.process_data import process_func_chatml
from src.attribution.saliency import (
    compute_alti_correlation_gradient,
    compute_alti_saliency_vector,
    compute_lm_head_ce_gradient_no_backward,
    compute_lm_head_ce_gradient_scores_no_backward,
    compute_lm_head_ce_gradient_sketches_no_backward,
)
from transformers import DataCollatorForSeq2Seq, set_seed

# ====== CONFIGURATION ======
SEED = 42
SELECTED_TEST_SAMPLE_INDEX = 58
TOKEN_INDEX_TO_RETRIEVE = 703  # The "first wrong token" we are investigating (single-token mode)

TOP_K_PROMPT_TOKENS = 4        # How many test correlation features to extract
TOP_K_TRAIN_SAMPLES = 10       # How many top train samples from coarse screening
TOP_TARGETS = 8                # How many response tokens to scan per train sample
TOP_K_SOURCE_PER_TARGET = 3    # Top source tokens per target (includes response-internal tokens)
CONTEXT_WINDOW_SIZE = 3        # Tokens shown on each side of source/target for annotation
FINE_MATCH_LAST_N_LAYERS = 1   # ALTI-gradient matching params: last N layers
FINE_MATCH_PROJ = "qk"         # Attention projections used for fine matching: qk, qkvo, vo, q/k/v/o, all
ALTI_CHUNK_SIZE = 8            # Query chunk size for ALTI contribution computation
ALTI_GRAD_CHUNK_SIZE = 32      # Pair-gradient starts fast and falls back on OOM
ALTI_GRAD_MAX_SEQ_LEN = None   # Skip ALTI-gradient pairs beyond this prefix length; <=0 disables

# All-tokens mode parameters
MAX_OUTPUT_TOKENS = 40         # Max response tokens to analyze in all-tokens mode
# Global pre-screen pool size. The full training set is scanned ONCE with the full-response
# CE gradient to obtain this pool, then each per-token re-ranking only scans the pool
# (COARSE_POOL_SIZE samples) instead of the full training set.
# Cost: 1 × N_train (global) + N_tokens × COARSE_POOL_SIZE (per-token re-rank)
COARSE_POOL_SIZE = 100
PRESCREEN_BATCH_SIZE = 1       # Increase via --prescreen-batch-size when GPU memory allows
PRESCREEN_SAMPLE_LIMIT = None  # Limit coarse prescreen scan for quick/debug runs
PRESCREEN_MAX_SEQ_LEN = 3000   # Skip longer train samples during prescreen/rerank; <=0 disables
PRESCREEN_SKETCH_DIM = 8192    # <=0 disables cached TensorSketch coarse retrieval
PRESCREEN_SKETCH_SEED = 42
PRESCREEN_SKETCH_CACHE_DIR = ".cache/prescreen_sketch"

# Token strings (after strip) that carry no semantic content and should be skipped
# in all-tokens mode. Single non-alphanumeric characters are also skipped.
_TRIVIAL_STRIPPED = {"{", "}", "(", ")", "[", "]", ",", ";"}
_CHAT_TEMPLATE_STRIPPED = {
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "system",
    "user",
    "assistant",
}


def lm_head_filter(name, param):
    """Only select the LM head weight — a single dense matrix that
    projects the last hidden state to vocabulary logits.
    This gives a compact, task-agnostic feature for every token prediction.
    """
    return name == "lm_head.weight"


def _fine_match_projection_names(proj_mode: str) -> tuple[str, ...]:
    """Map a compact projection mode such as 'qk' to module name fragments."""
    normalized = (proj_mode or "").lower().replace("_", "").replace("-", "").replace(",", "")
    if normalized == "all":
        normalized = "qkvo"
    if not normalized or any(ch not in "qkvo" for ch in normalized):
        raise ValueError(
            f"Unsupported fine-match projection mode: {proj_mode!r}. "
            "Use a combination of q/k/v/o, e.g. 'qk', 'vo', 'qkvo', or 'all'."
        )

    selected = set(normalized)
    return tuple(f"{ch}_proj" for ch in "qkvo" if ch in selected)


def make_attention_projection_filter(model, last_n_layers: int, proj_mode: str):
    """Select requested attention projection parameters in the final N decoder layers."""
    num_layers = len(model.model.layers)
    start_layer = max(0, num_layers - last_n_layers)
    projection_names = _fine_match_projection_names(proj_mode)

    def _filter(name, param):
        for layer_idx in range(start_layer, num_layers):
            prefix = f"model.layers.{layer_idx}.self_attn."
            if name.startswith(prefix) and any(proj in name for proj in projection_names):
                return True
        return False

    return _filter


def is_trivial_token(tokenizer, token_id: int) -> bool:
    """Return True for tokens that carry no semantic meaning.

    Skips chat-template tokens, role labels, pure whitespace, lone punctuation
    ({, }, (, ), [, ], comma, semicolon), and any single non-alphanumeric /
    non-underscore character.
    """
    all_special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if token_id in all_special_ids:
        return True

    tok_str = tokenizer.decode([token_id])
    stripped = tok_str.strip()
    if not stripped:
        return True
    if stripped in _CHAT_TEMPLATE_STRIPPED:
        return True
    raw_tok = tokenizer.convert_ids_to_tokens([token_id])[0]
    if raw_tok.strip() in _CHAT_TEMPLATE_STRIPPED:
        return True
    if stripped in _TRIVIAL_STRIPPED:
        return True
    if len(stripped) == 1 and not (stripped.isalnum() or stripped == "_"):
        return True
    return False


def top_nontrivial_saliency_sources(tokenizer, input_ids_1d, sal_vec, k: int):
    """Return top-k saliency sources after removing chat-template/trivial tokens."""
    candidates = (
        (idx, score)
        for idx, score in enumerate(sal_vec)
        if not is_trivial_token(tokenizer, int(input_ids_1d[idx].item()))
    )
    return nlargest(k, candidates, key=lambda x: x[1])


def find_first_valid_token_index(tokenizer, input_ids_tensor, start_idx):
    """
    Skip formatting characters like \\n, \\t, spaces, {, } to find the
    first token that carries actual semantic meaning.
    """
    valid_token_index = start_idx
    input_len = input_ids_tensor.size(1)
    while valid_token_index < input_len:
        tok_id = input_ids_tensor[0, valid_token_index].item()
        tok_str = tokenizer.decode([tok_id])
        if tok_str.strip() not in ["", "{", "}"]:
            break
        valid_token_index += 1
    return valid_token_index


def get_context_window(tokenizer, input_ids_1d, idx, window=CONTEXT_WINDOW_SIZE):
    """
    Return a list of token strings centered on `idx`.
    The focal token is wrapped in →[...]← for easy visual identification during annotation.
    """
    seq_len = input_ids_1d.size(0)
    tokens = []
    for i in range(max(0, idx - window), min(seq_len, idx + window + 1)):
        tok_str = tokenizer.decode([input_ids_1d[i].item()])
        tokens.append(f"→[{tok_str}]←" if i == idx else tok_str)
    return tokens


def _gather_scores(
    accelerator,
    local_scores: list[tuple[int, float]],
    device: torch.device,
) -> list[tuple[int, float]]:
    """Gather (train_idx, score) pairs from all processes into the main process.

    Handles variable-length lists per process by padding with sentinel -1 indices.
    In single-process mode this is a no-op.
    """
    if accelerator.num_processes == 1:
        return local_scores

    n = len(local_scores)
    all_lens = accelerator.gather(torch.tensor(n, device=device, dtype=torch.long))
    max_len = int(all_lens.max().item())

    padded_scores = torch.full((max_len,), float("nan"), device=device, dtype=torch.float32)
    padded_indices = torch.full((max_len,), -1, device=device, dtype=torch.long)
    if n > 0:
        padded_scores[:n] = torch.tensor([s for _, s in local_scores], device=device, dtype=torch.float32)
        padded_indices[:n] = torch.tensor([i for i, _ in local_scores], device=device, dtype=torch.long)

    all_scores = accelerator.gather(padded_scores)
    all_indices = accelerator.gather(padded_indices)

    valid = all_indices != -1
    return list(zip(
        all_indices[valid].cpu().tolist(),
        all_scores[valid].cpu().tolist(),
    ))


def _screen_training_set(
    model,
    test_ce_grad: torch.Tensor,   # flat tensor already on lm_head_device
    filtered_params: list,
    train_loader,
    accel_device: torch.device,   # accelerator.device for batch loading
    lm_head_device: torch.device, # device where lm_head lives (grad computed here)
    desc: str = "Scanning Train Samples",
    allowed_indices: set | None = None,
    max_seq_len: int | None = None,
) -> list[tuple[int, float]]:
    """Compute cosine similarity on GPU (lm_head_device). Returns LOCAL scores for this process."""
    sample_scores: list[tuple[int, float]] = []
    empty_ignored = torch.tensor([], device=accel_device)
    skipped_long = 0

    for batch in tqdm(train_loader, desc=desc, leave=False):
        train_indices = batch["sample_index"].view(-1).tolist()
        keep_rows = []
        for row, train_idx in enumerate(train_indices):
            if allowed_indices is not None and int(train_idx) not in allowed_indices:
                continue
            if max_seq_len is not None:
                seq_len = int(batch["attention_mask"][row].sum().item())
                if seq_len > max_seq_len:
                    skipped_long += 1
                    continue
            keep_rows.append(row)

        if not keep_rows:
            continue

        rows = torch.tensor(keep_rows, dtype=torch.long, device=batch["input_ids"].device)
        batch_kept = {
            k: v.index_select(0, rows).to(accel_device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor) and k != "sample_index"
        }

        scores = compute_lm_head_ce_gradient_scores_no_backward(
            model=model,
            batch=batch_kept,
            device=accel_device,
            ignored_token_ids=empty_ignored,
            test_ce_grad=test_ce_grad,
            score_device=lm_head_device,
        )

        for row, score in zip(keep_rows, scores):
            sample_scores.append((int(train_indices[row]), float(score)))

        del batch_kept, scores, rows

    if skipped_long:
        print(f"  {desc}: skipped {skipped_long} samples longer than {max_seq_len} tokens")

    return sample_scores


def _dataset_fingerprint(train_ds, *, max_seq_len: int | None, sketch_dim: int, sketch_seed: int) -> str:
    """Stable-enough fingerprint for the tokenized train set used by the sketch cache."""
    h = hashlib.sha1()
    h.update(f"n={len(train_ds)}|max_seq_len={max_seq_len}|dim={sketch_dim}|seed={sketch_seed}".encode())
    for i in range(len(train_ds)):
        item = train_ds[i]
        ids = item["input_ids"]
        labels = item.get("labels")
        sample_index = int(item.get("sample_index", i))
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        h.update(str(sample_index).encode())
        h.update(str(int(ids.numel())).encode())
        h.update(str(int(ids[: min(32, ids.numel())].long().sum().item())).encode())
        h.update(str(int(ids[-min(32, ids.numel()):].long().sum().item())).encode())
        if isinstance(labels, torch.Tensor):
            valid = int(labels.ne(-100).sum().item())
        else:
            valid = 0
        h.update(str(valid).encode())
    return h.hexdigest()[:16]


def _prescreen_sketch_cache_path(
    model,
    train_ds,
    *,
    max_seq_len: int | None,
    sketch_dim: int,
    sketch_seed: int,
    cache_dir: str,
) -> str:
    model_name = str(getattr(getattr(model, "config", None), "_name_or_path", "model"))
    model_hash = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    data_hash = _dataset_fingerprint(
        train_ds,
        max_seq_len=max_seq_len,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
    )
    return os.path.join(
        cache_dir,
        f"lmhead_sketch_{model_hash}_{data_hash}_d{sketch_dim}_s{sketch_seed}.pt",
    )


def _load_or_build_prescreen_sketch_cache(
    model,
    train_ds,
    train_loader,
    accelerator,
    *,
    max_seq_len: int | None,
    sketch_dim: int,
    sketch_seed: int,
    cache_dir: str,
):
    """Build/load cached low-dimensional train LM-head gradient sketches."""
    if sketch_dim <= 0:
        return None
    if accelerator.num_processes != 1:
        print("Prescreen sketch cache disabled for multi-process Accelerator runs.")
        return None

    cache_path = _prescreen_sketch_cache_path(
        model,
        train_ds,
        max_seq_len=max_seq_len,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
        cache_dir=cache_dir,
    )
    if os.path.exists(cache_path):
        print(f"Loading prescreen sketch cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Building prescreen sketch cache: {cache_path}")
    print("  First run is expected to be slow; later test samples reuse this file.")
    sample_ids = []
    sketch_chunks = []
    empty_ignored = torch.tensor([], device=accelerator.device)
    skipped_long = 0

    for batch in tqdm(train_loader, desc="Build Prescreen Sketch Cache", leave=False):
        train_indices = batch["sample_index"].view(-1).tolist()
        keep_rows = []
        for row, train_idx in enumerate(train_indices):
            if max_seq_len is not None:
                seq_len = int(batch["attention_mask"][row].sum().item())
                if seq_len > max_seq_len:
                    skipped_long += 1
                    continue
            keep_rows.append(row)

        if not keep_rows:
            continue

        rows = torch.tensor(keep_rows, dtype=torch.long, device=batch["input_ids"].device)
        batch_kept = {
            k: v.index_select(0, rows).to(accelerator.device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor) and k != "sample_index"
        }
        sketches = compute_lm_head_ce_gradient_sketches_no_backward(
            model=model,
            batch=batch_kept,
            device=accelerator.device,
            ignored_token_ids=empty_ignored,
            sketch_dim=sketch_dim,
            sketch_seed=sketch_seed,
        ).detach().cpu().to(torch.float16)

        sample_ids.extend(int(train_indices[row]) for row in keep_rows)
        sketch_chunks.append(sketches)
        del batch_kept, sketches, rows

    if not sketch_chunks:
        print("Prescreen sketch cache is empty; falling back to exact scanning.")
        return None

    cache = {
        "sample_ids": torch.tensor(sample_ids, dtype=torch.long),
        "sketches": torch.cat(sketch_chunks, dim=0).contiguous(),
        "sketch_dim": int(sketch_dim),
        "sketch_seed": int(sketch_seed),
        "max_seq_len": max_seq_len,
    }
    torch.save(cache, cache_path)
    if skipped_long:
        print(f"  Sketch cache skipped {skipped_long} samples longer than {max_seq_len} tokens.")
    print(f"  Saved {cache['sketches'].size(0)} train sketches.")
    return cache


def _score_prescreen_sketch_cache(
    query_sketch: torch.Tensor,
    cache,
    device,
    *,
    allowed_indices: set[int] | None = None,
) -> list[tuple[int, float]]:
    ids = cache["sample_ids"]
    sketches = cache["sketches"]
    if allowed_indices is not None:
        allowed = torch.tensor(sorted(int(x) for x in allowed_indices), dtype=torch.long)
        mask = torch.isin(ids, allowed)
        ids = ids[mask]
        sketches = sketches[mask]
    if ids.numel() == 0:
        return []

    q = F.normalize(query_sketch.to(device=device, dtype=torch.float32), dim=0, eps=1e-12)
    scores = sketches.to(device=device, dtype=torch.float32).matmul(q)
    return list(zip(ids.cpu().tolist(), scores.detach().cpu().tolist()))


def run_causal_intervention_experiment(
    all_tokens: bool = False,
    prescreen_batch_size: int = PRESCREEN_BATCH_SIZE,
    prescreen_limit: int | None = PRESCREEN_SAMPLE_LIMIT,
    prescreen_max_seq_len: int | None = PRESCREEN_MAX_SEQ_LEN,
    alti_grad_chunk_size: int = ALTI_GRAD_CHUNK_SIZE,
    alti_grad_max_seq_len: int | None = ALTI_GRAD_MAX_SEQ_LEN,
    fine_match_proj: str = FINE_MATCH_PROJ,
    prescreen_sketch_dim: int = PRESCREEN_SKETCH_DIM,
    prescreen_sketch_seed: int = PRESCREEN_SKETCH_SEED,
    prescreen_sketch_cache_dir: str = PRESCREEN_SKETCH_CACHE_DIR,
):
    accelerator = Accelerator()
    set_seed(SEED)
    prescreen_batch_size = max(1, int(prescreen_batch_size))
    alti_grad_chunk_size = max(1, int(alti_grad_chunk_size))
    if prescreen_limit is not None and prescreen_limit <= 0:
        prescreen_limit = None
    if prescreen_max_seq_len is not None and prescreen_max_seq_len <= 0:
        prescreen_max_seq_len = None
    if alti_grad_max_seq_len is not None and alti_grad_max_seq_len <= 0:
        alti_grad_max_seq_len = None
    prescreen_sketch_dim = int(prescreen_sketch_dim or 0)
    fine_match_proj = (fine_match_proj or FINE_MATCH_PROJ).lower()
    fine_match_projection_names = _fine_match_projection_names(fine_match_proj)

    model, tokenizer = load_model_and_tokenizer()
    param_filter = lm_head_filter
    fine_param_filter = make_attention_projection_filter(model, FINE_MATCH_LAST_N_LAYERS, fine_match_proj)
    print(
        "Fine matching uses ALTI gradients over "
        f"last {FINE_MATCH_LAST_N_LAYERS} layer(s), projections={fine_match_proj} "
        f"({', '.join(fine_match_projection_names)})"
    )

    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    test_samples  = load_samples_from_formal_jsonl("sft_test.jsonl")

    SEQUENCE_LENGTH_LIMIT = 3000

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, label_pad_token_id=-100, return_tensors="pt",
    )
    collator = CustomCollator(base_collator)

    train_ds = build_train_dataset(train_samples, convert_to_chatml)
    prescreen_train_ds = train_ds
    if prescreen_limit is not None and prescreen_limit < len(train_ds):
        prescreen_train_ds = train_ds.select(range(prescreen_limit))
        print(f"Prescreen limited: scanning {len(prescreen_train_ds)} / {len(train_ds)} train samples")

    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(prescreen_train_ds), batch_size=prescreen_batch_size, collate_fn=collator,
    )
    train_loader = accelerator.prepare(train_loader)
    prescreen_sketch_cache = _load_or_build_prescreen_sketch_cache(
        model,
        prescreen_train_ds,
        train_loader,
        accelerator,
        max_seq_len=prescreen_max_seq_len,
        sketch_dim=prescreen_sketch_dim,
        sketch_seed=prescreen_sketch_seed,
        cache_dir=prescreen_sketch_cache_dir,
    )

    infer_fw = NewInferenceFunction(
        model=model, tokenizer=tokenizer,
        train_loader=train_loader, accelerator=accelerator,
        param_filter_fn=param_filter, top_k=20,
    )

    # lm_head might live on a different device under device_map="auto"
    filtered_params = [p for n, p in model.named_parameters() if param_filter(n, p)]
    lm_head_device  = filtered_params[0].device if filtered_params else accelerator.device

    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    # ── Build full test sequence (prompt + generated response) ──────────────
    test_ds       = build_single_sample_dataset(test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_to_chatml)
    raw_test_batch = base_collator([test_ds[0]])
    raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}

    infer_fw.model.eval()
    gen_result = infer_fw.infer(raw_test_batch, compute_saliency=False)
    prompt_len = int(gen_result["target_idx"][0])
    prompt_ids = raw_test_batch["input_ids"][0, :prompt_len]
    pred_ids   = torch.tensor(gen_result["pred_ids"][0], device=prompt_ids.device, dtype=prompt_ids.dtype)
    new_input_ids      = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels         = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100

    test_batch = {
        "input_ids":      new_input_ids,
        "attention_mask": new_attention_mask,
        "labels":         new_labels,
    }
    seq_len = test_batch["input_ids"].size(1)

    def _alti_grad_chunks() -> list[int]:
        chunks = []
        chunk = alti_grad_chunk_size
        while chunk >= 1:
            if chunk not in chunks:
                chunks.append(chunk)
            chunk //= 2
        if 1 not in chunks:
            chunks.append(1)
        return chunks

    def _is_cuda_alloc_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cublas_status_alloc_failed" in msg
            or "cublascreate" in msg
            or "cuda error: cublas" in msg
        )

    def _compute_alti_correlation_gradient_retry(**kwargs):
        target_idx = int(kwargs["target_idx_in_seq"])
        if alti_grad_max_seq_len is not None and target_idx > alti_grad_max_seq_len:
            print(
                f"  Skipping ALTI-gradient target={target_idx}: "
                f"exceeds --alti-grad-max-seq-len={alti_grad_max_seq_len}"
            )
            return None

        chunks = _alti_grad_chunks()
        last_error_message = None
        for chunk in chunks:
            try:
                return compute_alti_correlation_gradient(
                    **kwargs,
                    chunk_size=chunk,
                )
            except torch.OutOfMemoryError as exc:
                last_error_message = str(exc)
                print(f"  OOM in ALTI-gradient chunk={chunk}; retrying smaller chunk...")
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if not _is_cuda_alloc_error(exc):
                    raise
                last_error_message = str(exc)
                print(f"  OOM in ALTI-gradient chunk={chunk}; retrying smaller chunk...")
                torch.cuda.empty_cache()
        print(f"  Skipping ALTI-gradient after OOM chunks {chunks}: {last_error_message}")
        return None

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: Stage 3 processing for one train sample
    # Returns (train_sample_detail_dict, pair_records_list)
    # detail is None if the sample must be skipped (too long / no marker).
    # If the sample was already computed (cache hit), pass existing detail + candidate_pairs.
    # ════════════════════════════════════════════════════════════════════════════
    def _process_train_sample_stage3(
        train_idx: int,
        coarse_score: float,
        test_corr_features: dict,
        target_tok_text_for_record: str,
        target_tok_idx_for_record: int,
        pair_id_prefix: str,
        pair_id_start: int,
        cached_detail: dict | None = None,
    ) -> tuple[dict | None, list, int]:
        """Returns (detail_dict_or_None, pair_records, next_pair_id)."""
        nonlocal model, tokenizer, base_collator, accelerator, fine_param_filter

        if cached_detail is None:
            tr_ds   = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
            tr_batch = base_collator([tr_ds[0]])
            tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

            if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
                print(f"  Skipping train {train_idx}: sequence too long")
                return None, [], pair_id_start

            try:
                start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + 3
            except ValueError:
                print(f"  Skipping train {train_idx}: assistant marker not found")
                return None, [], pair_id_start

            response_start = find_first_valid_token_index(tokenizer, tr_batch["input_ids"], start_sys)
            tr_seq_len     = tr_batch["input_ids"].size(1)
            full_tokens    = tokenizer.convert_ids_to_tokens(tr_batch["input_ids"][0].tolist())
            target_saliencies: dict[int, list[float]] = {}
            candidate_pairs: list[tuple[float, int, int]] = []

            with torch.inference_mode(False):
                t_tr = response_start
                kept_targets = 0
                while t_tr < tr_seq_len and kept_targets < TOP_TARGETS:
                    if is_trivial_token(tokenizer, int(tr_batch["input_ids"][0, t_tr].item())):
                        t_tr += 1
                        continue
                    sal_vec = compute_alti_saliency_vector(
                        model,
                        tr_batch,
                        t_tr,
                        chunk_size=ALTI_CHUNK_SIZE,
                    )
                    target_saliencies[t_tr] = [round(float(s), 6) for s in sal_vec]
                    for s_idx, s_score in top_nontrivial_saliency_sources(
                        tokenizer,
                        tr_batch["input_ids"][0],
                        sal_vec,
                        TOP_K_SOURCE_PER_TARGET,
                    ):
                        candidate_pairs.append((float(s_score), t_tr, int(s_idx)))
                    kept_targets += 1
                    t_tr += 1

            print(f"  Step A: {len(candidate_pairs)} candidate pairs "
                  f"({TOP_TARGETS}t × {TOP_K_SOURCE_PER_TARGET}s each)")

            cached_detail = {
                "full_tokens":        full_tokens,
                "answer_start_index": response_start,
                "coarse_cos_sim":     float(coarse_score),
                "saliencies_by_token": {str(k): v for k, v in target_saliencies.items()},
                "_candidate_pairs":   candidate_pairs,
                "_tr_batch_cpu":      {k: v.cpu() for k, v in tr_batch.items()},
                "_feature_cache":     {},
            }
        else:
            # Update coarse score to the maximum seen across test tokens
            if coarse_score > cached_detail["coarse_cos_sim"]:
                cached_detail["coarse_cos_sim"] = float(coarse_score)
            candidate_pairs = cached_detail["_candidate_pairs"]
            cached_detail.setdefault("_feature_cache", {})

        # Move tr_batch to device for ALTI-gradient correlation matching
        tr_batch_gpu = {k: v.to(accelerator.device) for k, v in cached_detail["_tr_batch_cpu"].items()}
        ids_1d       = tr_batch_gpu["input_ids"][0]
        response_start = cached_detail["answer_start_index"]
        pair_counter   = pair_id_start
        pair_records   = []

        with torch.inference_mode(False):
            for saliency_score, t_tr, s_idx in candidate_pairs:
                train_target_tok   = tokenizer.decode([ids_1d[t_tr].item()])
                train_source_tok   = tokenizer.decode([ids_1d[s_idx].item()])
                response_tok_offset = t_tr - response_start

                print(f"  Step B: '{train_source_tok}' -> '{train_target_tok}' "
                      f"(offset={response_tok_offset}, sal={saliency_score:.4f})")

                feature_key = (int(t_tr), int(s_idx))
                if feature_key in cached_detail["_feature_cache"]:
                    train_feat = cached_detail["_feature_cache"][feature_key]
                    if train_feat is None:
                        continue
                    print("    feature cache hit")
                else:
                    train_feat = _compute_alti_correlation_gradient_retry(
                        model=model,
                        batch=tr_batch_gpu,
                        target_idx_in_seq=t_tr,
                        source_idx_in_seq=s_idx,
                        param_filter_fn=fine_param_filter,
                        device=accelerator.device,
                    )
                    cached_detail["_feature_cache"][feature_key] = train_feat
                    if train_feat is None:
                        continue
                source_ctx = get_context_window(tokenizer, ids_1d, s_idx)
                target_ctx = get_context_window(tokenizer, ids_1d, t_tr)

                for test_p_idx, (test_feat, test_src_text, test_saliency) in test_corr_features.items():
                    cos_sim = F.cosine_similarity(
                        test_feat,
                        train_feat,
                        dim=0,
                    ).item()

                    pair_records.append({
                        "id":            f"{pair_id_prefix}_{pair_counter:04d}",
                        "cos_sim":       float(cos_sim),
                        "coarse_cos_sim": float(coarse_score),
                        "train_sample_id": train_idx,
                        "test_correlation": {
                            "source_token":       test_src_text,
                            "source_token_index": test_p_idx,
                            "target_token":       target_tok_text_for_record,
                            "target_token_index": target_tok_idx_for_record,
                            "saliency_score":     float(test_saliency),
                        },
                        "train_correlation": {
                            "source_token":         train_source_tok,
                            "source_token_index":   s_idx,
                            "target_token":         train_target_tok,
                            "target_token_index":   t_tr,
                            "saliency_score":       saliency_score,
                            "response_token_offset": response_tok_offset,
                        },
                        "train_context": {
                            "source_context": source_ctx,
                            "target_context": target_ctx,
                        },
                        "annotation": None,
                    })
                    pair_counter += 1

        del tr_batch_gpu
        torch.cuda.empty_cache()
        return cached_detail, pair_records, pair_counter

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: compute test-side CE gradient for one token position
    # ════════════════════════════════════════════════════════════════════════════
    def _test_ce_grad_for_token(tok_idx: int) -> torch.Tensor:
        ce_labels = torch.full_like(test_batch["input_ids"], -100)
        ce_labels[0, tok_idx] = test_batch["input_ids"][0, tok_idx]
        single_tok_batch = {
            "input_ids":      test_batch["input_ids"],
            "attention_mask": test_batch["attention_mask"],
            "labels":         ce_labels,
        }
        grad = compute_lm_head_ce_gradient_no_backward(
            model=model, batch=single_tok_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
        )
        return grad.reshape(-1).to(lm_head_device).detach()

    def _test_ce_sketch_for_token(tok_idx: int) -> torch.Tensor:
        ce_labels = torch.full_like(test_batch["input_ids"], -100)
        ce_labels[0, tok_idx] = test_batch["input_ids"][0, tok_idx]
        single_tok_batch = {
            "input_ids":      test_batch["input_ids"],
            "attention_mask": test_batch["attention_mask"],
            "labels":         ce_labels,
        }
        return compute_lm_head_ce_gradient_sketches_no_backward(
            model=model,
            batch=single_tok_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
            sketch_dim=prescreen_sketch_dim,
            sketch_seed=prescreen_sketch_seed,
        )[0].detach()

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: compute test-side CE gradient aggregated over ALL response tokens
    # Used for the global one-shot coarse pre-screen in all_tokens mode.
    # test_batch["labels"] already masks prompt positions with -100, so
    # The analytic LM-head gradient naturally aggregates CE loss across all
    # response tokens.
    # ════════════════════════════════════════════════════════════════════════════
    def _test_ce_grad_full_response() -> torch.Tensor:
        grad = compute_lm_head_ce_gradient_no_backward(
            model=model, batch=test_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
        )
        return grad.reshape(-1).to(lm_head_device).detach()

    def _test_ce_sketch_full_response() -> torch.Tensor:
        return compute_lm_head_ce_gradient_sketches_no_backward(
            model=model,
            batch=test_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
            sketch_dim=prescreen_sketch_dim,
            sketch_seed=prescreen_sketch_seed,
        )[0].detach()

    # ════════════════════════════════════════════════════════════════════════════
    # ── SINGLE-TOKEN MODE ────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════════
    if not all_tokens:
        print("\n=== Stage 1: Extracting test correlation features ===")

        target_tok_id   = test_batch["input_ids"][0, TOKEN_INDEX_TO_RETRIEVE].item()
        target_tok_text = tokenizer.decode([target_tok_id])
        baseline_saliency = compute_alti_saliency_vector(
            model,
            test_batch,
            TOKEN_INDEX_TO_RETRIEVE,
            chunk_size=ALTI_CHUNK_SIZE,
        )

        top_test_corr = top_nontrivial_saliency_sources(
            tokenizer,
            test_batch["input_ids"][0],
            baseline_saliency,
            TOP_K_PROMPT_TOKENS,
        )
        top_test_correlations = [
            {
                "source_token_index": idx,
                "source_token":       tokenizer.decode([test_batch["input_ids"][0, idx].item()]),
                "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
                "target_token":       target_tok_text,
                "saliency_score":     float(score),
            }
            for idx, score in top_test_corr
        ]

        print("Computing ALTI-gradient test features...")
        test_corr_features: dict = {}
        with torch.inference_mode(False):
            for item in top_test_correlations:
                p_idx = item["source_token_index"]
                print(f"  '{item['source_token']}' -> '{item['target_token']}' (sal={item['saliency_score']:.4f})")
                feat = _compute_alti_correlation_gradient_retry(
                    model=model,
                    batch=test_batch,
                    target_idx_in_seq=TOKEN_INDEX_TO_RETRIEVE,
                    source_idx_in_seq=p_idx,
                    param_filter_fn=fine_param_filter,
                    device=accelerator.device,
                )
                if feat is None:
                    continue
                test_corr_features[p_idx] = (feat, item["source_token"], item["saliency_score"])

        print(f"\n=== Stage 2: Coarse screening (single token {TOKEN_INDEX_TO_RETRIEVE}) ===")
        if prescreen_sketch_cache is not None:
            test_ce_sketch = _test_ce_sketch_for_token(TOKEN_INDEX_TO_RETRIEVE)
            sample_scores = _score_prescreen_sketch_cache(
                test_ce_sketch,
                prescreen_sketch_cache,
                accelerator.device,
            )
        else:
            test_ce_grad  = _test_ce_grad_for_token(TOKEN_INDEX_TO_RETRIEVE)
            local_scores  = _screen_training_set(
                model, test_ce_grad, filtered_params, train_loader,
                accelerator.device, lm_head_device, desc="Stage 2",
            )
            sample_scores  = _gather_scores(accelerator, local_scores, accelerator.device)
        related_samples = nlargest(TOP_K_TRAIN_SAMPLES, sample_scores, key=lambda x: x[1])
        print(f"  Top-{TOP_K_TRAIN_SAMPLES} selected: {[(i, round(s,4)) for i,s in related_samples]}")

        print("\n=== Stage 3: Fine-grained correlation matching ===")
        all_pair_records: list = []
        train_sample_details: dict = {}
        pair_id_counter = 0

        for rank, (train_idx, coarse_score) in enumerate(related_samples):
            print(f"\n--- Train {train_idx}  (rank={rank+1}, coarse={coarse_score:.4f}) ---")
            detail, pairs, pair_id_counter = _process_train_sample_stage3(
                train_idx, coarse_score, test_corr_features,
                target_tok_text, TOKEN_INDEX_TO_RETRIEVE,
                "pair", pair_id_counter,
                cached_detail=train_sample_details.get(str(train_idx)),
            )
            if detail is not None:
                train_sample_details[str(train_idx)] = detail
                all_pair_records.extend(pairs)

        all_pair_records.sort(key=lambda x: x["cos_sim"], reverse=True)

        report_json = {
            "experiment_meta": {
                "test_sample_index":  SELECTED_TEST_SAMPLE_INDEX,
                "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
                "mode":               "single_token",
                "config": {
                    "TOP_K_PROMPT_TOKENS":    TOP_K_PROMPT_TOKENS,
                    "TOP_K_TRAIN_SAMPLES":    TOP_K_TRAIN_SAMPLES,
                    "TOP_TARGETS":            TOP_TARGETS,
                    "TOP_K_SOURCE_PER_TARGET": TOP_K_SOURCE_PER_TARGET,
                    "CONTEXT_WINDOW_SIZE":    CONTEXT_WINDOW_SIZE,
                    "SALIENCY_METHOD":        "alti",
                    "MATCHING_METHOD":        f"alti_gradient_{fine_match_proj}",
                    "FINE_MATCH_LAST_N_LAYERS": FINE_MATCH_LAST_N_LAYERS,
                    "FINE_MATCH_PROJ":        fine_match_proj,
                    "ALTI_CHUNK_SIZE":        ALTI_CHUNK_SIZE,
                    "ALTI_GRAD_CHUNK_SIZE":   alti_grad_chunk_size,
                    "ALTI_GRAD_MAX_SEQ_LEN":  alti_grad_max_seq_len,
                    "PRESCREEN_BATCH_SIZE":   prescreen_batch_size,
                    "PRESCREEN_SAMPLE_LIMIT": prescreen_limit,
                    "PRESCREEN_MAX_SEQ_LEN":  prescreen_max_seq_len,
                    "PRESCREEN_SKETCH_DIM":   prescreen_sketch_dim,
                    "PRESCREEN_SKETCH_SEED":  prescreen_sketch_seed,
                    "PRESCREEN_SKETCH_CACHE": prescreen_sketch_cache is not None,
                },
            },
            "test_sample_baseline": {
                "target_token":        target_tok_text,
                "target_token_index":  TOKEN_INDEX_TO_RETRIEVE,
                "full_tokens":         gen_result["pred_full_tokens"][0],  # prompt + model output
                "correct_full_tokens": gen_result["full_tokens"][0],    # prompt + ground truth answer
                "top_correlations":    top_test_correlations,
            },
            "correlation_pairs": all_pair_records,
            "train_sample_details": {
                k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
                for k, v in train_sample_details.items()
            },
        }

        total = len(all_pair_records)
        print(f"\nTotal pairs: {total}  (top-10 by cos_sim below)")
        for r in all_pair_records[:10]:
            print(f"  [{r['id']}] train#{r['train_sample_id']:3d} "
                  f"'{r['train_correlation']['source_token']}'->"
                  f"'{r['train_correlation']['target_token']}' "
                  f"| test '{r['test_correlation']['source_token']}'->"
                  f"'{r['test_correlation']['target_token']}' "
                  f"| cos_sim={r['cos_sim']:.4f}")

        report_filename = f"correlation_matching_results_test{SELECTED_TEST_SAMPLE_INDEX}_tok{TOKEN_INDEX_TO_RETRIEVE}.json"

    # ════════════════════════════════════════════════════════════════════════════
    # ── ALL-TOKENS MODE ──────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════════
    else:
        max_end = min(prompt_len + MAX_OUTPUT_TOKENS, seq_len)
        valid_test_tokens = [
            t for t in range(prompt_len, max_end)
            if not is_trivial_token(tokenizer, int(test_batch["input_ids"][0, t].item()))
        ]
        print(f"\nAll-tokens mode: {len(valid_test_tokens)} semantic tokens "
              f"(from {max_end - prompt_len} response tokens, trivial skipped)")

        # ── Stage 0: Global coarse pre-screen (one pass over the full training set) ──
        # Aggregate CE loss over all response tokens → single gradient vector G_test.
        # Cost: 1 × N_train  (vs. N_tokens × N_train in the old per-token approach)
        print(f"\n=== Global Pre-Screen: full training set → Top-{COARSE_POOL_SIZE} pool ===")
        if prescreen_sketch_cache is not None:
            full_response_ce_sketch = _test_ce_sketch_full_response()
            all_pool_scores = _score_prescreen_sketch_cache(
                full_response_ce_sketch,
                prescreen_sketch_cache,
                accelerator.device,
            )
            del full_response_ce_sketch
        else:
            full_response_ce_grad = _test_ce_grad_full_response()
            local_pool_scores = _screen_training_set(
                model, full_response_ce_grad, filtered_params, train_loader,
                accelerator.device, lm_head_device, desc="Global Pre-Screen",
                max_seq_len=prescreen_max_seq_len,
            )
            all_pool_scores = _gather_scores(accelerator, local_pool_scores, accelerator.device)
            del full_response_ce_grad
        coarse_pool: set[int] = {
            idx for idx, _ in nlargest(COARSE_POOL_SIZE, all_pool_scores, key=lambda x: x[1])
        }
        print(f"  Coarse pool ({len(coarse_pool)} samples): {sorted(coarse_pool)}")

        # Materialise pool as a dedicated DataLoader so per-token Stage 2 truly iterates
        # only COARSE_POOL_SIZE times (not N_train times with skip logic).
        idx_to_row  = {int(train_ds[i]["sample_index"]): i for i in range(len(train_ds))}
        pool_rows   = sorted(idx_to_row[idx] for idx in coarse_pool if idx in idx_to_row)
        pool_ds     = train_ds.select(pool_rows)
        pool_loader = torch.utils.data.DataLoader(
            DatasetWrapper(pool_ds), batch_size=prescreen_batch_size, collate_fn=collator,
        )
        pool_loader = accelerator.prepare(pool_loader)
        print(f"  pool_loader built: {len(pool_rows)} samples")

        # ── Per-token loop: re-rank within pool_loader, then run Stage 3 ──
        per_token_results: list  = []
        train_sample_cache: dict = {}   # str(train_idx) → detail dict (with _candidate_pairs, _tr_batch_cpu)
        pair_id_counter = 0

        for t in tqdm(valid_test_tokens, desc="Test tokens",
                      disable=not accelerator.is_local_main_process):
            target_tok_id   = int(test_batch["input_ids"][0, t].item())
            target_tok_text = tokenizer.decode([target_tok_id])
            print(f"\n=== Token {t}: '{target_tok_text}' ===")

            # Stage 1a: cheap saliency at t
            with torch.inference_mode(False):
                sal_vec = compute_alti_saliency_vector(
                    model,
                    test_batch,
                    t,
                    chunk_size=ALTI_CHUNK_SIZE,
                )
            top_test_corr = top_nontrivial_saliency_sources(
                tokenizer,
                test_batch["input_ids"][0],
                sal_vec,
                TOP_K_PROMPT_TOKENS,
            )
            top_test_correlations = [
                {
                    "source_token_index": idx,
                    "source_token":       tokenizer.decode([int(test_batch["input_ids"][0, idx].item())]),
                    "target_token_index": t,
                    "target_token":       target_tok_text,
                    "saliency_score":     float(score),
                }
                for idx, score in top_test_corr
            ]

            # Stage 1b: ALTI-gradient test features (freed after this token)
            print(f"  Computing {len(top_test_correlations)} test ALTI-gradient features...")
            test_corr_features: dict = {}
            with torch.inference_mode(False):
                for item in top_test_correlations:
                    p_idx = item["source_token_index"]
                    feat = _compute_alti_correlation_gradient_retry(
                        model=model,
                        batch=test_batch,
                        target_idx_in_seq=t,
                        source_idx_in_seq=p_idx,
                        param_filter_fn=fine_param_filter,
                        device=accelerator.device,
                    )
                    if feat is None:
                        continue
                    test_corr_features[p_idx] = (feat, item["source_token"], item["saliency_score"])

            if not test_corr_features:
                print(f"  No ALTI-gradient test features survived for token {t}; skipping Stage 2/3.")
                per_token_results.append({
                    "target_token_index": t,
                    "target_token":       target_tok_text,
                    "top_correlations":   top_test_correlations,
                    "correlation_pairs":  [],
                })
                torch.cuda.empty_cache()
                continue

            # Stage 2: per-token CE grad → re-rank within pool_loader only
            # pool_loader contains exactly COARSE_POOL_SIZE samples — no skip logic needed.
            if prescreen_sketch_cache is not None:
                token_ce_sketch = _test_ce_sketch_for_token(t)
                all_scores = _score_prescreen_sketch_cache(
                    token_ce_sketch,
                    prescreen_sketch_cache,
                    accelerator.device,
                    allowed_indices=coarse_pool,
                )
                del token_ce_sketch
            else:
                token_ce_grad  = _test_ce_grad_for_token(t)
                local_scores   = _screen_training_set(
                    model, token_ce_grad, filtered_params, pool_loader,
                    accelerator.device, lm_head_device,
                    desc=f"Stage2 t={t}",
                    max_seq_len=prescreen_max_seq_len,
                )
                all_scores     = _gather_scores(accelerator, local_scores, accelerator.device)
                del token_ce_grad
            related_samples = nlargest(TOP_K_TRAIN_SAMPLES, all_scores, key=lambda x: x[1])
            print(f"  Top-{TOP_K_TRAIN_SAMPLES} from pool: "
                  f"{[(i, round(s,4)) for i,s in related_samples]}")

            # Stage 3: fine-grained matching for this token's top train samples
            token_pair_records: list = []
            for rank, (train_idx, coarse_score) in enumerate(related_samples):
                print(f"\n  --- Train {train_idx} (rank={rank+1}, coarse={coarse_score:.4f}) ---")
                cached = train_sample_cache.get(str(train_idx))
                detail, pairs, pair_id_counter = _process_train_sample_stage3(
                    train_idx, coarse_score, test_corr_features,
                    target_tok_text, t,
                    f"t{t}", pair_id_counter,
                    cached_detail=cached,
                )
                if detail is not None:
                    train_sample_cache[str(train_idx)] = detail
                    token_pair_records.extend(pairs)

            token_pair_records.sort(key=lambda x: x["cos_sim"], reverse=True)
            per_token_results.append({
                "target_token_index": t,
                "target_token":       target_tok_text,
                "top_correlations":   top_test_correlations,
                "correlation_pairs":  token_pair_records,
            })

            # Free test-side features before next token
            del test_corr_features
            torch.cuda.empty_cache()

        # Clean internal cache keys before saving
        train_sample_details = {
            k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
            for k, v in train_sample_cache.items()
        }

        report_json = {
            "experiment_meta": {
                "test_sample_index":  SELECTED_TEST_SAMPLE_INDEX,
                "mode":               "all_tokens",
                "max_output_tokens":  MAX_OUTPUT_TOKENS,
                "tokens_analyzed":    len(per_token_results),
                "screening":          "global_pool_then_per_token_rerank",
                "config": {
                    "TOP_K_PROMPT_TOKENS":     TOP_K_PROMPT_TOKENS,
                    "TOP_K_TRAIN_SAMPLES":     TOP_K_TRAIN_SAMPLES,
                    "COARSE_POOL_SIZE":        COARSE_POOL_SIZE,
                    "PRESCREEN_BATCH_SIZE":    prescreen_batch_size,
                    "TOP_TARGETS":             TOP_TARGETS,
                    "TOP_K_SOURCE_PER_TARGET": TOP_K_SOURCE_PER_TARGET,
                    "CONTEXT_WINDOW_SIZE":     CONTEXT_WINDOW_SIZE,
                    "SALIENCY_METHOD":         "alti",
                    "MATCHING_METHOD":         f"alti_gradient_{fine_match_proj}",
                    "FINE_MATCH_LAST_N_LAYERS": FINE_MATCH_LAST_N_LAYERS,
                    "FINE_MATCH_PROJ":         fine_match_proj,
                    "ALTI_CHUNK_SIZE":         ALTI_CHUNK_SIZE,
                    "ALTI_GRAD_CHUNK_SIZE":    alti_grad_chunk_size,
                    "ALTI_GRAD_MAX_SEQ_LEN":   alti_grad_max_seq_len,
                    "PRESCREEN_SAMPLE_LIMIT":  prescreen_limit,
                    "PRESCREEN_MAX_SEQ_LEN":   prescreen_max_seq_len,
                    "PRESCREEN_SKETCH_DIM":    prescreen_sketch_dim,
                    "PRESCREEN_SKETCH_SEED":   prescreen_sketch_seed,
                    "PRESCREEN_SKETCH_CACHE":  prescreen_sketch_cache is not None,
                },
            },
            "test_sample_baseline": {
                "full_tokens":         gen_result["pred_full_tokens"][0],  # prompt + model output (clickable)
                "correct_full_tokens": gen_result["full_tokens"][0],       # prompt + ground truth answer
                "prompt_len":          prompt_len,
            },
            "per_token_results":    per_token_results,
            "train_sample_details": train_sample_details,
        }

        report_filename = f"correlation_matching_results_test{SELECTED_TEST_SAMPLE_INDEX}_all_tokens.json"

    # ── Save (main process only in multi-GPU) ────────────────────────────────
    if accelerator.is_main_process:
        report_json = round_floats(report_json, 5)
        base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_path = os.path.join(base_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, ensure_ascii=False)
        print(f"\nExperiment completed. Results → {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run causal intervention experiment for a single test sample + token."
    )
    parser.add_argument(
        "--test-index", type=int, default=None,
        help="Index of the test sample to analyse (overrides SELECTED_TEST_SAMPLE_INDEX).",
    )
    parser.add_argument(
        "--token-index", type=int, default=None,
        help="Token position to investigate (overrides TOKEN_INDEX_TO_RETRIEVE). Single-token mode only.",
    )
    parser.add_argument(
        "--all-tokens", action="store_true",
        help="Attribute all output tokens (up to MAX_OUTPUT_TOKENS) instead of a single token.",
    )
    parser.add_argument(
        "--prescreen-batch-size", type=int, default=PRESCREEN_BATCH_SIZE,
        help="Batch size for coarse prescreen and pool rerank scoring.",
    )
    parser.add_argument(
        "--prescreen-limit", type=int, default=PRESCREEN_SAMPLE_LIMIT,
        help="Limit coarse prescreen to the first N train samples. Use <=0 to disable.",
    )
    parser.add_argument(
        "--prescreen-max-seq-len", type=int, default=PRESCREEN_MAX_SEQ_LEN,
        help="Skip train samples longer than this during prescreen/rerank. Use <=0 to disable.",
    )
    parser.add_argument(
        "--prescreen-sketch-dim", type=int, default=PRESCREEN_SKETCH_DIM,
        help="TensorSketch dimension for cached coarse prescreen retrieval. Use <=0 to disable.",
    )
    parser.add_argument(
        "--prescreen-sketch-seed", type=int, default=PRESCREEN_SKETCH_SEED,
        help="Random seed for deterministic TensorSketch hashes.",
    )
    parser.add_argument(
        "--prescreen-sketch-cache-dir", type=str, default=PRESCREEN_SKETCH_CACHE_DIR,
        help="Directory used to store/reuse train coarse sketch caches.",
    )
    parser.add_argument(
        "--alti-grad-chunk-size", type=int, default=ALTI_GRAD_CHUNK_SIZE,
        help="Initial query chunk size for ALTI-gradient matching; OOM retries use smaller chunks.",
    )
    parser.add_argument(
        "--alti-grad-max-seq-len", type=int, default=ALTI_GRAD_MAX_SEQ_LEN,
        help="Skip ALTI-gradient pairs whose target prefix is longer than this. Use <=0 to disable.",
    )
    parser.add_argument(
        "--fine-match-proj", type=str, default=FINE_MATCH_PROJ,
        help=(
            "Attention projections used for ALTI-gradient fine matching. "
            "Default: qk. Use qkvo/all for the previous behavior, or vo/q/k/v/o for ablations."
        ),
    )
    parser.add_argument(
        "--top-targets", type=int, default=None,
        help="How many response target tokens to scan per train sample.",
    )
    parser.add_argument(
        "--top-k-source-per-target", type=int, default=None,
        help="How many source tokens to keep for each train response target.",
    )
    args = parser.parse_args()

    if args.test_index is not None:
        SELECTED_TEST_SAMPLE_INDEX = args.test_index
    if args.token_index is not None:
        TOKEN_INDEX_TO_RETRIEVE = args.token_index
    if args.top_targets is not None:
        TOP_TARGETS = max(1, int(args.top_targets))
    if args.top_k_source_per_target is not None:
        TOP_K_SOURCE_PER_TARGET = max(1, int(args.top_k_source_per_target))

    mode_str = "all_tokens" if args.all_tokens else f"single_token tok={TOKEN_INDEX_TO_RETRIEVE}"
    print(f"[intervention] test_index={SELECTED_TEST_SAMPLE_INDEX}  mode={mode_str}")
    run_causal_intervention_experiment(
        all_tokens=args.all_tokens,
        prescreen_batch_size=args.prescreen_batch_size,
        prescreen_limit=args.prescreen_limit,
        prescreen_max_seq_len=args.prescreen_max_seq_len,
        alti_grad_chunk_size=args.alti_grad_chunk_size,
        alti_grad_max_seq_len=args.alti_grad_max_seq_len,
        fine_match_proj=args.fine_match_proj,
        prescreen_sketch_dim=args.prescreen_sketch_dim,
        prescreen_sketch_seed=args.prescreen_sketch_seed,
        prescreen_sketch_cache_dir=args.prescreen_sketch_cache_dir,
    )
