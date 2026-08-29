from collections.abc import Callable, Sequence, Mapping
from functools import partial
from heapq import nlargest
import json
import os
import re
from pprint import pprint
from time import time
from typing import Literal
import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq, GenerationMixin, PreTrainedTokenizer, Qwen2ForCausalLM, set_seed
from accelerate import Accelerator

from src.sft.inference import print_query_and_answer
from src.attribution.process_data import process_func_chatml, CustomCollator, list_of_dicts_to_dict_of_lists as dataset_list_to_dict
from src.attribution.saliency import compute_alti_saliency_vector, compute_gradients, compute_loss_per_sample
from src.attribution.auto_annotate import annotate_samples

from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from torch.nn import Parameter


SEED = 42
SEQUENCE_LENGTH_LIMIT = 3000
TRAIN_SAMPLE_RETRIEVE_LIMIT = 100
SELECTED_TEST_SAMPLE_INDEX = 58
TOKEN_INDEX_TO_RETRIEVE = 703


# dev-only patch, to see the shape of tensor when debugging

def patch_torch_to_inspect_tensor_shape():
    _orig_repr = torch.Tensor.__repr__

    def _repr_with_shape(self: torch.Tensor, tensor_contents = None) -> str:
        # 1) prepend shape/dtype/device
        # 2) keep original tensor formatting
        head = f"tensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})--"
        return head + _orig_repr(self)

    torch.Tensor.__repr__ = _repr_with_shape


patch_torch_to_inspect_tensor_shape()


# disable the "Map: ..." progress bar in Dataset

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


def round_floats(obj, ndigits: int):
    # 1) recursively walk containers
    # 2) round floats only
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, Mapping):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


class Qwen2ForCausalLMWithLastAttn(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.last_attention = None

    def forward(self, *args, save_last_attention: bool = False, **kwargs):
        self.last_attention = None
        handle = None

        if save_last_attention:
            def _hook(_module, _inputs, output):
                _, attn_weights = output
                self.last_attention = attn_weights

            # Hook only the last layer's attention
            last_attn = self.model.layers[-1].self_attn
            if not isinstance(last_attn, nn.Module):
                raise TypeError("Expected `self_attn` to be an `nn.Module`.")
            handle = last_attn.register_forward_hook(_hook)

        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

        if not save_last_attention:
            return outputs

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=(self.last_attention,) if self.last_attention is not None else None,
        )


# Only use attention layers
# this part was used in IF_HF.py, and is deprecated, now we use all parameters

target_layer_keywords = ["embed_tokens.weight"]  # or ["model.layers.27.mlp"]

def filter_params(name: str, param: Parameter):
    return any(key in name for key in target_layer_keywords) and param.requires_grad

def unfreeze_target_params(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if any(key in name for key in target_layer_keywords):
            param.requires_grad = True


def load_samples_from_formal_jsonl(jsonl_path: str):
    sample_list = []
    seen_inputs  = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
                    sys  = obj['messages'][0]['content']
                    inp  = obj['messages'][1]['content']
                    outp = obj['messages'][2]['content']
                    if inp in seen_inputs:
                        continue
                    sample_list.append(
                        {
                            'system': sys,
                            'input':  inp,
                            'output': outp
                        }
                    )
                    seen_inputs.add(inp)
            # if len(sample_list) >= 1000:
            #     break

    return sample_list


def build_train_dataset(train_samples, convert_fn):
    train_ds = Dataset.from_dict(dataset_list_to_dict(train_samples))
    train_ds = train_ds.map(lambda x, i: {"sample_index": i}, with_indices=True)
    train_ds = train_ds.map(
        convert_fn, 
        batched=True,
        remove_columns=["input", "output", "system"]
    )
    train_ds.set_format(type="torch", columns=["input_ids", "labels", "sample_index"])

    return train_ds


def build_single_sample_dataset(sample, convert_fn):
    # Build single-sample Dataset
    temp_ds = Dataset.from_dict({
        "input":  [sample["input"]],
        "output": [sample["output"]],
        "system": [sample["system"]],
    })
    temp_ds = temp_ds.map(
        convert_fn, 
        batched=True, 
        remove_columns=["input", "output", "system"]
    )

    return temp_ds


def load_model_and_tokenizer():
    abs_model_path = os.path.join(os.path.dirname(__file__), "sft/scripts/nif-checkpoints/checkpoint-full")
    print(f"Loading model from {abs_model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(abs_model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(
        abs_model_path,
        attn_implementation="eager",
        output_attentions=False,
        use_cache=False,
    )

    model = Qwen2ForCausalLMWithLastAttn.from_pretrained(
        abs_model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).eval()

    # # Freeze params except attention params
    # for param in model.parameters():
    #     param.requires_grad = False

    # unfreeze_target_params(model)

    return model, tokenizer


def _find_subseq_start(row: torch.Tensor, subseq: tuple[int, int, int]) -> int:
    a, b, c = subseq
    for i in range(row.numel() - 1):
        if int(row[i]) == a and int(row[i + 1]) == b:
            return i
    raise ValueError("marker sequence not found")


def _apply_freeze_strategy(model, strategy: str):
    """
    Apply parameter freezing strategy before finetuning.
    
    Strategies:
    - 'all': Full finetune, all parameters trainable
    - 'qk_all': Only Q/K projections in all layers
    - 'qk_last_quarter': Only Q/K projections in last 1/4 layers
    - 'qk_last_half': Only Q/K projections in last 1/2 layers
    - 'qk_last': Only Q/K projections in the last layer
    - 'attn_all': All attention params (Q/K/V/O) in all layers
    - 'attn_last_quarter': All attention params in last 1/4 layers
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    if strategy == 'all':
        # Full finetune: unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        return
    
    num_layers = len(model.model.layers)
    
    if strategy == 'qk_all':
        # Q/K in all layers
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.requires_grad = True
            layer.self_attn.k_proj.weight.requires_grad = True
            if layer.self_attn.q_proj.bias is not None:
                layer.self_attn.q_proj.bias.requires_grad = True
            if layer.self_attn.k_proj.bias is not None:
                layer.self_attn.k_proj.bias.requires_grad = True
                
    elif strategy == 'qk_last_quarter':
        # Q/K in last 1/4 layers
        start_layer = num_layers * 3 // 4
        for i in range(start_layer, num_layers):
            layer = model.model.layers[i]
            layer.self_attn.q_proj.weight.requires_grad = True
            layer.self_attn.k_proj.weight.requires_grad = True
            if layer.self_attn.q_proj.bias is not None:
                layer.self_attn.q_proj.bias.requires_grad = True
            if layer.self_attn.k_proj.bias is not None:
                layer.self_attn.k_proj.bias.requires_grad = True
                
    elif strategy == 'qk_last_half':
        # Q/K in last 1/2 layers
        start_layer = num_layers // 2
        for i in range(start_layer, num_layers):
            layer = model.model.layers[i]
            layer.self_attn.q_proj.weight.requires_grad = True
            layer.self_attn.k_proj.weight.requires_grad = True
            if layer.self_attn.q_proj.bias is not None:
                layer.self_attn.q_proj.bias.requires_grad = True
            if layer.self_attn.k_proj.bias is not None:
                layer.self_attn.k_proj.bias.requires_grad = True
                
    elif strategy == 'qk_last':
        # Q/K in the last layer only
        layer = model.model.layers[-1]
        layer.self_attn.q_proj.weight.requires_grad = True
        layer.self_attn.k_proj.weight.requires_grad = True
        if layer.self_attn.q_proj.bias is not None:
            layer.self_attn.q_proj.bias.requires_grad = True
        if layer.self_attn.k_proj.bias is not None:
            layer.self_attn.k_proj.bias.requires_grad = True
            
    elif strategy == 'attn_all':
        # All attention params (Q/K/V/O) in all layers
        for layer in model.model.layers:
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj_module = getattr(layer.self_attn, proj)
                proj_module.weight.requires_grad = True
                if proj_module.bias is not None:
                    proj_module.bias.requires_grad = True
                    
    elif strategy == 'attn_last_quarter':
        # All attention params in last 1/4 layers
        start_layer = num_layers * 3 // 4
        for i in range(start_layer, num_layers):
            layer = model.model.layers[i]
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj_module = getattr(layer.self_attn, proj)
                proj_module.weight.requires_grad = True
                if proj_module.bias is not None:
                    proj_module.bias.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze strategy: {strategy}. "
                        f"Choose from: 'all', 'qk_all', 'qk_last_quarter', 'qk_last_half', "
                        f"'qk_last', 'attn_all', 'attn_last_quarter'")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze strategy '{strategy}': {trainable:,} / {total:,} params trainable ({100*trainable/total:.2f}%)")


def finetune_on_sample(
    model, 
    tokenizer, 
    question: str = "", 
    answer: str = "", 
    mode: str = "supervised", 
    *, 
    epochs: int = 1, 
    lr: float = 5e-5, 
    input_ids: Tensor | None = None, 
    labels: Tensor | None = None, 
    boost_indices: list[int] | None = None, 
    boost_coef: float = 1.0,
    freeze_strategy: str = "qk_last_quarter",
    first_gen_pos: int | None = None,
):
    """
    Finetune model on a single sample with attention guidance.
    
    Args:
        model: The model to finetune
        tokenizer: Tokenizer for encoding
        question: Question text (used if input_ids not provided)
        answer: Answer text (used if input_ids not provided)
        mode: 'supervised' or 'unsupervised'
        epochs: Number of training epochs
        lr: Learning rate
        input_ids: Pre-encoded input IDs
        labels: Pre-encoded labels
        boost_indices: Token indices that the first generated token should attend to
        boost_coef: Coefficient for attention loss (lambda in: loss = base_loss - lambda * L_attn)
        freeze_strategy: Which parameters to train. Options:
            - 'all': Full finetune
            - 'qk_all': Q/K projections in all layers
            - 'qk_last_quarter': Q/K projections in last 1/4 layers (recommended)
            - 'qk_last_half': Q/K projections in last 1/2 layers
            - 'qk_last': Q/K projections in last layer only
            - 'attn_all': All attention params in all layers
            - 'attn_last_quarter': All attention params in last 1/4 layers
        first_gen_pos: Position of the first generated token. If None, auto-detect from marker.
    """
    device = next(model.parameters()).device
    
    # Apply freeze strategy
    _apply_freeze_strategy(model, freeze_strategy)
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)       

    if mode == "supervised":
        if input_ids is None or labels is None:
            enc_q = tokenizer(question, return_tensors="pt").to(device)
            enc_a = tokenizer(answer, return_tensors="pt").to(device)
            # 1) Concatenate question+answer and mask question tokens in labels
            input_ids = torch.cat([enc_q["input_ids"], enc_a["input_ids"]], dim=1)
            labels = torch.cat([torch.full_like(enc_q["input_ids"], -100), enc_a["input_ids"]], dim=1)

        marker_ids = tuple(
            tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        )
        if len(marker_ids) != 3:
            raise ValueError("expected three-token marker for <|im_start|>assistant\\n")
        for i in range(input_ids.size(0)):
            start = _find_subseq_start(input_ids[i], marker_ids) + 3
            labels[i, :start] = -100
            # Auto-detect first_gen_pos if not provided
            if first_gen_pos is None:
                first_gen_pos = start

    elif mode == "unsupervised":
        if input_ids is None or labels is None:
            text = question + answer
            enc = tokenizer(text, return_tensors="pt").to(device)
            input_ids = labels = enc["input_ids"]
    else:
        raise ValueError("mode must be 'supervised' or 'unsupervised'")
    
    if not isinstance(input_ids, Tensor):
        raise ValueError("Expect input_ids to be Tensor")
    if not isinstance(labels, Tensor):
        raise ValueError("Expect labels to be Tensor")
    input_ids = input_ids.to(device=device)
    labels = labels.to(device=device)

    for epoch in tqdm(range(epochs), desc="Finetuning on sample", leave=False):
        out = model(input_ids=input_ids, labels=labels, save_last_attention=True)

        attn_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        if boost_indices is not None and len(boost_indices) > 0 and first_gen_pos is not None:
            attn = out.attentions[-1]   # [batch, head, q, k]
            if not isinstance(attn, Tensor):
                raise ValueError("Expect attn to be Tensor")
            
            idx = torch.tensor(boost_indices, device=device)
            
            # NEW ATTENTION LOSS:
            # Only look at the first generated token's attention distribution
            # first_gen_pos is the query position (the first token to generate)
            # boost_indices are the key positions to attend to
            #
            # L_attn = mean( log(attn[first_gen_pos, boost_indices]) )
            # Total loss = base_loss - lambda * L_attn
            # (minimizing this maximizes attention on target positions)
            
            first_token_attn = attn[:, :, first_gen_pos, :]  # [batch, head, k]
            
            # Get attention on target positions and take log
            attn_on_target = first_token_attn[:, :, idx]  # [batch, head, len(idx)]
            log_attn = torch.log(attn_on_target + 1e-8)
            
            # L_attn = average log-attention on target positions
            L_attn = log_attn.mean()
            
            # We want to MAXIMIZE L_attn, so attn_loss = -L_attn
            attn_loss = -L_attn

        # Total loss = base_loss + boost_coef * attn_loss
        # Since attn_loss = -L_attn, this is equivalent to: base_loss - boost_coef * L_attn
        loss = out.loss + boost_coef * attn_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Debug print (uncomment for debugging)
        # print(f"Epoch {epoch}: loss={loss.item():.4f}, base_loss={out.loss.item():.4f}, attn_loss={attn_loss.item():.4f}")

        # Explicitly delete variables holding computation graph to prevent OOM
        del loss, out, attn_loss
        if boost_indices is not None and len(boost_indices) > 0 and first_gen_pos is not None:
            del attn, first_token_attn, attn_on_target, log_attn
        torch.cuda.empty_cache()


_TAG_RE = re.compile(r"<ATTN>(.*?)</ATTN>", re.DOTALL)

def extract_attn_segments(text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Return cleaned text and spans of content originally inside <ATTN> tags.

    Spans are character offsets [start, end) in the cleaned text.
    """
    spans: list[tuple[int, int]] = []
    out_parts: list[str] = []
    cursor = 0
    out_len = 0

    # 1) Scan tagged segments in source order
    for m in _TAG_RE.finditer(text):
        # Append text before tag
        pre = text[cursor:m.start()]
        out_parts.append(pre)
        out_len += len(pre)

        # Append tag content and record span in cleaned text
        content = m.group(1)
        start = out_len
        out_parts.append(content)
        out_len += len(content)
        end = out_len
        spans.append((start, end))

        cursor = m.end()

    # 2) Append trailing text after last tag
    out_parts.append(text[cursor:])

    # 3) Join for cleaned text
    cleaned = "".join(out_parts)
    return cleaned, spans


def tokenize_with_marked_tokens(text: str, tokenizer: PreTrainedTokenizer):
    cleaned, spans = extract_attn_segments(text)
    result = tokenizer(cleaned, return_tensors="pt", return_offsets_mapping=True)

    offset_mapping_tensor = result['offset_mapping']
    if not isinstance(offset_mapping_tensor, Tensor):
        raise ValueError("Expect offset_mapping_tensor to be Tensor")
    offset_mapping = offset_mapping_tensor[0].tolist()
    marked_indices = []

    len_spans = len(spans)
    i_span = 0
    in_span = False
    for i, (token_start, token_end) in enumerate(offset_mapping):
        while i_span < len_spans:
            span_start, span_end = spans[i_span]
            if token_end <= span_start:     # move to next token
                break
            elif span_end <= token_start:   # move to next span
                i_span += 1
            else:                           # token_end > span_start and span_end > token_start
                in_span = True
                i_span += 1
        
        if in_span:
            marked_indices.append(i)

        in_span = False

        if i_span >= len_spans:
            break

    return {
        **result,
        "marked_indices": marked_indices
    }


def convert_sample_to_full_text(sample):
    text = f"<|im_start|>system\n{sample['system']}<|im_end|>\n<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"
    text = text.replace('\\t', '\t').replace('\\n', '\n')
    return text


class NewInferenceFunction:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        train_loader: DataLoader | None = None,
        accelerator: Accelerator | None = None,
        param_filter_fn: Callable[[str, Parameter], bool] | None = None,
        ignored_token_ids: Sequence[int] | None = None,
        top_k: int = 10,
        top_k_ratio: float = 0.2,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.accelerator = accelerator

        # Resolve device once to avoid repeated attribute checks
        self.device = (
            accelerator.device if accelerator is not None else next(model.parameters()).device
        )

        # Configure filtering and selection parameters
        self.param_filter_fn = param_filter_fn
        self.top_k = top_k
        self.top_k_ratio = top_k_ratio

        # Normalize ignored IDs to a device-local tensor
        ignored_token_ids = ignored_token_ids or []
        self.ignored_token_ids = torch.tensor(ignored_token_ids, device=self.device)

        self.base_train_results: tuple | None = None
        # BUG FIX: was `self.param_filter_fn: ... = None` which overrode the param passed in __init__
        self.param_snapshot_original: list[Parameter] | None = None
        self.param_snapshot_overfit: list[Parameter] | None = None

    @torch.no_grad()
    def save_model_params(self, to: Literal["original"] | Literal["overfit"] | None = "original"):
        model = self.model
        param_filter_fn = self.param_filter_fn

        snapshot = []
        for name, param in model.named_parameters():
            if param.requires_grad and (param_filter_fn is None or param_filter_fn(name, param)):
                snapshot.append(param.detach().cpu().clone())

        if to == "original":
            self.param_snapshot_original = snapshot
        elif to == "overfit":
            self.param_snapshot_overfit = snapshot

        return snapshot

    @torch.no_grad()
    def restore_model_params(self, param_snapshot: list[Parameter] | None = None):
        model = self.model
        param_filter_fn = self.param_filter_fn

        if param_snapshot is None:
            param_snapshot = self.param_snapshot_original
        if param_snapshot is None:
            return

        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad and (param_filter_fn is None or param_filter_fn(name, param)):
                param.data.copy_(param_snapshot[idx].to(param.device))
                idx += 1

    @torch.no_grad()
    def _apply_gradient_update(self, grads, lr):
        '''
        Vanilla gradient update with no optimizer.
        '''
        model = self.model
        param_filter_fn = self.param_filter_fn

        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad and (param_filter_fn is None or param_filter_fn(name, param)):
                if grads[idx] is not None:
                    grad_device = grads[idx].to(param.device).to(param.dtype)
                    param.data -= lr * grad_device
                idx += 1

    def _build_topk_attention_mask(self, base_attention_mask, attn_scores, target_idx):
        '''
        Compute a new attention mask base on top-k scores.
        '''
        new_mask = base_attention_mask.clone()
        num_prev = target_idx
        if num_prev <= 1:
            return new_mask

        if self.top_k is not None:
            k = min(self.top_k, num_prev)
        else:
            k = max(1, int(num_prev * (self.top_k_ratio or 0.2)))

        topk_indices = torch.topk(attn_scores[:num_prev], k=k, largest=True).indices
        keep = torch.zeros(num_prev, dtype=torch.bool, device=new_mask.device)
        keep[topk_indices.to(new_mask.device)] = True

        new_mask[:, :num_prev] = keep.to(new_mask.dtype)
        return new_mask

    @torch.no_grad()
    def infer(self, batch, target_idx=None, gen_limit: int = 128, compute_saliency: bool = True):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # find target index
        if target_idx is None:
            marker_ids = tuple(
                self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            )
            if len(marker_ids) != 3:
                raise ValueError("expected three-token marker for <|im_start|>assistant\\n")
            starts = [
                _find_subseq_start(input_ids[i], marker_ids) + 3
                for i in range(input_ids.size(0))
            ]
            target_idx = torch.tensor(starts, device=input_ids.device)
        elif not isinstance(target_idx, Tensor):
            target_idx = torch.tensor(target_idx, device=input_ids.device)

        time_start = time()

        # Part 1: single token inference, saliency on ground truth sequence

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        time_part_1_infer = time()
        print(f'Inferring single token costs {(time_part_1_infer - time_start):.3f}s')

        def _alti_saliency_list(sal_batch, starts):
            sal_input_ids = sal_batch["input_ids"].to(self.device)
            sal_attention_mask = sal_batch["attention_mask"].to(self.device)
            out = []
            for i, start in enumerate(starts.tolist()):
                sample_out = []
                sample_len = int(sal_attention_mask[i].sum().item())
                sample_batch = {
                    "input_ids": sal_input_ids[i:i + 1, :sample_len],
                    "attention_mask": sal_attention_mask[i:i + 1, :sample_len],
                }
                for t in range(max(int(start), 1), sample_len):
                    sample_out.append({
                        "index": t,
                        "saliency": compute_alti_saliency_vector(self.model, sample_batch, t),
                    })
                out.append(sample_out)
            return out

        saliency_original = _alti_saliency_list(batch, target_idx) if compute_saliency else [[] for _ in range(input_ids.size(0))]

        time_part_1_saliency = time()
        print(f'Computing saliency of original sample costs {(time_part_1_saliency - time_part_1_infer):.3f}s')

        logits = outputs.logits
        batch_size = logits.size(0)

        prev_ids, prev_text, out_ids, out_text = [], [], [], []
        for i in range(batch_size):
            pos = int(target_idx[i].item())
            original_prev_ids = input_ids[i, :pos].tolist()
            out_token_id = int(torch.argmax(logits[i, pos-1], dim=-1).item())  # shifted

            prev_ids.append(original_prev_ids)
            prev_text.append(self.tokenizer.decode(original_prev_ids))
            out_ids.append(out_token_id)
            out_text.append(self.tokenizer.decode(out_token_id))

        # Trim each sample to target_idx and pad to a common length for generation
        trim_lens = target_idx.to(torch.long).tolist()
        max_len = max(trim_lens)
        trimmed_ids = input_ids.new_full((batch_size, max_len), self.tokenizer.eos_token_id)
        trimmed_mask = attention_mask.new_zeros((batch_size, max_len))

        for i, tlen in enumerate(trim_lens):
            trimmed_ids[i, :tlen] = input_ids[i, :tlen]
            trimmed_mask[i, :tlen] = 1

        if not isinstance(self.model, GenerationMixin):
            raise ValueError("Expect self.model to be GenerationMixin")

        # Part 2: generation sequence inference, saliency on generated sequence

        gen_out = self.model.generate(
            input_ids=trimmed_ids,
            attention_mask=trimmed_mask,
            max_new_tokens=gen_limit,
            do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        if not isinstance(gen_out, GenerateDecoderOnlyOutput):
            raise ValueError("Expect gen_out to be GenerateOutput")
        gen_ids = gen_out.sequences
        gen_scores = torch.stack(gen_out.scores, dim=0)
        topk_vals, topk_ids = torch.topk(gen_scores, k=10, dim=-1)  # [steps, batch, 10]
        topk_map = [
            [
                list(zip(
                    self.tokenizer.convert_ids_to_tokens(ids.tolist()),
                    vals.tolist()
                ))
                for ids, vals in zip(step_ids, step_vals)
            ]
            for step_ids, step_vals in zip(topk_ids, topk_vals)
        ]

        if not isinstance(gen_ids, Tensor):
            raise ValueError("Expect gen_ids to be Tensor")
        if not isinstance(self.tokenizer.pad_token_id, int):
            raise ValueError("Expect pad_token_id to be int")

        time_part_2_generation = time()
        print(f'Generating full sequence of {gen_ids.size(dim=1) - trimmed_ids.size(dim=1)} tokens costs {(time_part_2_generation - time_part_1_saliency):.3f}s')

        # build generation batch and compute saliency on generated continuation
        gen_attention_mask = gen_ids.ne(int(self.tokenizer.pad_token_id)).to(dtype=attention_mask.dtype)
        gen_labels = gen_ids.clone()

        # mask prompt and padding positions for loss
        for i, t in enumerate(target_idx.tolist()):
            gen_labels[i, :t] = -100
        gen_labels = gen_labels.masked_fill(gen_attention_mask == 0, -100)

        gen_batch = {
            **batch,
            "input_ids": gen_ids,
            "attention_mask": gen_attention_mask,
            "labels": gen_labels,
        }
        saliency_generation = _alti_saliency_list(gen_batch, target_idx) if compute_saliency else [[] for _ in range(input_ids.size(0))]

        time_part_2_saliency = time()
        print(f'Computing saliency on the generation result costs {(time_part_2_saliency-time_part_2_generation):.3f}s')

        pred_ids, pred_text, pred_full_text = [], [], []
        full_text, answer_text = [], []
        pred_tokens, pred_full_tokens, full_tokens, answer_tokens = [], [], [], []

        for i in range(batch_size):
            prompt_len = int(target_idx[i].item())
            continuation = gen_ids[i, prompt_len:].tolist()
            full_gen_ids = gen_ids[i].tolist()

            pred_ids.append(continuation)
            pred_tokens.append(self.tokenizer.convert_ids_to_tokens(continuation))
            pred_text.append(self.tokenizer.decode(continuation))

            pred_full_tokens.append(self.tokenizer.convert_ids_to_tokens(full_gen_ids))
            pred_full_text.append(self.tokenizer.decode(full_gen_ids))

            valid_ids = input_ids[i, :int(attention_mask[i].sum().item())].tolist()
            full_tokens.append(self.tokenizer.convert_ids_to_tokens(valid_ids))
            full_text.append(self.tokenizer.decode(valid_ids))

            ans_ids = input_ids[i, prompt_len:int(attention_mask[i].sum().item())].tolist()
            answer_tokens.append(self.tokenizer.convert_ids_to_tokens(ans_ids))
            answer_text.append(self.tokenizer.decode(ans_ids))

        return {
            **batch,
            "logits": logits,
            "attention_mask": attention_mask,
            "target_idx": target_idx.tolist(),
            "prev_ids": prev_ids,
            "prev_text": prev_text,
            "out_ids": out_ids,
            "out_text": out_text,
            "gen_ids": gen_ids,
            "pred_ids": pred_ids,
            "pred_text": pred_text,
            "pred_full_text": pred_full_text,
            "full_text": full_text,
            "answer_text": answer_text,
            "pred_tokens": pred_tokens,
            "pred_full_tokens": pred_full_tokens,
            "full_tokens": full_tokens,
            "answer_tokens": answer_tokens,
            "saliency_original": saliency_original,
            "saliency_generation": saliency_generation,
            "pred_scores": gen_scores.tolist(),
            "pred_token_options": topk_map
        }
    
    def decode_next_token(self, logits, position):
        token_id = int(torch.argmax(logits[0, position], dim=-1).item())
        return self.tokenizer.decode(token_id)

    def _get_train_losses(self):
        all_sum_losses, all_indices = [], []
        tokenwise_dict = {}

        self.model.eval()
        with torch.inference_mode():
            if not isinstance(self.train_loader, DataLoader):
                raise TypeError("Expected `self.train_loader` to be an `DataLoader`.")
            pbar = tqdm(self.train_loader, desc=f"Getting Train Losses", leave=False)
            loss = 1.0
            for batch in pbar:
                pbar.set_postfix(loss=f"{loss:.4f}")
                batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                mean_loss, token_loss = compute_loss_per_sample(
                    self.model,
                    batch_gpu,
                    self.device,
                    self.ignored_token_ids
                )

                shift_labels = batch_gpu["labels"][..., 1:].contiguous()
                indices = batch_gpu["sample_index"].cpu().tolist()

                for i, idx in enumerate(indices):
                    valid_mask = shift_labels[i] != -100
                    start_idx = torch.where(valid_mask)[0][0] if valid_mask.any() else 0
                    tokenwise_dict[idx] = token_loss[i][start_idx:].cpu()

                all_sum_losses.append(mean_loss.detach().cpu())
                all_indices.append(batch_gpu["sample_index"].detach().cpu())

                loss = mean_loss.item()
                del batch_gpu, mean_loss, token_loss, shift_labels  # crucial release

        return torch.cat(all_sum_losses), torch.cat(all_indices), tokenwise_dict

    def influence_overfit_single(
        self,
        query_batch,
        lr=1e-2,
        max_steps=1000,
        loss_threshold=1e-4,
        top_k=10,
        target_idx=None,
    ):
        if self.base_train_results is None:
            self.base_train_results = self._get_train_losses()

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        def _masked_query_batch(batch, target_idx_local):
            masked = {k: v for k, v in batch.items()}
            labels = batch["labels"].clone()
            start = int(target_idx_local[0])
            labels[..., :start] = -100
            masked["labels"] = labels
            return masked, start

        if target_idx is None:
            raise ValueError("target_idx is required to overfit generation starting from target_idx")

        query_batch, start_q = _masked_query_batch(query_batch, target_idx)

        # Compute starting loss
        with torch.no_grad():
            scalar, tokenwise_raw = compute_loss_per_sample(
                self.model, query_batch, self.device, self.ignored_token_ids
            )
            loss_test_start = scalar.item()
            loss_test_tokenwise_start = tokenwise_raw[0][start_q:].cpu()

        self.save_model_params(to="original")

        # Overfitting
        loss_test_curr = loss_test_start
        pbar = tqdm(range(max_steps), desc=f"Overfitting on single sample", leave=False)
        for step in pbar:
            pbar.set_postfix(loss=f"{loss_test_curr:.4f}")

            if loss_test_curr < loss_threshold:
                break

            raise RuntimeError(
                "influence_overfit_single depended on the removed gradient-saliency "
                "path. Use intervention_experiment.py's ALTI correlation matching instead."
            )
            self._apply_gradient_update(grads, lr=lr)
            self.model.zero_grad(set_to_none=True)

            with torch.no_grad():
                scalar, _ = compute_loss_per_sample(
                    self.model, query_batch, self.device, self.ignored_token_ids
                )
                loss_test_curr = scalar.item()

        # Compute ending loss
        with torch.no_grad():
            _, l_test_des_tokenwise_raw = compute_loss_per_sample(
                self.model, query_batch, self.device, self.ignored_token_ids
            )
            loss_test_tokenwise_end = l_test_des_tokenwise_raw[0][start_q:].cpu()

        # Test loss diff
        query_token_diffs = loss_test_tokenwise_end - loss_test_tokenwise_start
        delta_test = loss_test_curr - loss_test_start

        # Train loss diff
        loss_train_des_sum, _, loss_train_des_tokenwise = self._get_train_losses()
        loss_train_base_sum, indices_train, loss_train_base_tokenwise = self.base_train_results

        # Compute similarity scores
        local_scores = []
        local_diffs = {}
        for i, idx in enumerate(indices_train.tolist()):
            rel_delta_train = loss_train_des_sum[i].item() - loss_train_base_sum[i].item()
            # average over all train samples
            denom = loss_train_base_sum[i].item() + 1e-8
            # cosine similarity
            normalized_score = delta_test * (rel_delta_train / denom)
            local_scores.append(normalized_score)
            # optional, record token-level loss changes
            local_diffs[idx] = loss_train_des_tokenwise[idx] - loss_train_base_tokenwise[idx]

        if self.accelerator is None:
            return local_scores, indices_train.tolist(), local_diffs, query_token_diffs

        all_scores = self.accelerator.gather(torch.tensor(local_scores, device=self.device))
        all_indices = self.accelerator.gather(indices_train)

        self.save_model_params(to="overfit")
        self.restore_model_params(self.param_snapshot_original)

        if not isinstance(all_scores, Tensor):
            raise ValueError("Expect all_scores to be Tensor")
        
        if not isinstance(all_indices, Tensor):
            raise ValueError("Expect all_indices to be Tensor")

        return all_scores.tolist(), all_indices.tolist(), local_diffs, query_token_diffs
    
    def _mask_labels_before_target(
        self,
        batch: Mapping[str, Tensor],
        target_idx: int | Sequence[int] | Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        if "labels" not in batch:
            raise KeyError("batch must contain `labels`.")
        labels = batch["labels"].clone()  # labels: [B, T]
        if not isinstance(target_idx, Tensor):
            target_idx = torch.tensor(target_idx, device=labels.device)
        target_idx = target_idx.to(device=labels.device)

        if target_idx.ndim == 0:
            target_idx = target_idx[None]
        if labels.size(0) != target_idx.numel():
            raise ValueError("target_idx size must match batch size.")

        for i, start in enumerate(target_idx.tolist()):
            labels[i, :int(start)] = -100
        masked = dict(batch)
        masked["labels"] = labels
        return masked, target_idx

    def _flatten_grads(
        self,
        grads: Sequence[Tensor],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        # flatten each param grad to 1D then concatenate -> [P]
        flat = [
            g.detach().reshape(-1).to(device=self.device, dtype=dtype)
            for g in grads
            if g is not None
        ]
        if not flat:
            raise RuntimeError("No gradients to flatten.")
        return torch.cat(flat, dim=0)

    def influence_gradient_single(
        self,
        query_batch: Mapping[str, Tensor],
        target_idx: int | Sequence[int] | Tensor,
    ):
        if not isinstance(self.train_loader, DataLoader):
            raise TypeError("Expected `self.train_loader` to be a DataLoader.")
        if target_idx is None:
            raise ValueError("target_idx is required.")

        self.model.eval()
        query_batch = {k: v.to(self.device) for k, v in query_batch.items() if isinstance(v, Tensor)}
        query_batch, _ = self._mask_labels_before_target(query_batch, target_idx)

        # query_vec: [P] flattened gradient over selected params
        query_grads = compute_gradients(
            self.model, query_batch, self.param_filter_fn, self.device, self.ignored_token_ids
        )
        query_vec = self._flatten_grads(query_grads, dtype=torch.float32)
        query_norm = query_vec.norm().clamp_min(1e-12)

        local_scores, local_indices = [], []
        pbar = tqdm(self.train_loader, desc="Gradient influence", leave=False)
        
        # sample level 
        for batch in pbar:
            batch_gpu = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, Tensor)}
            indices = batch_gpu.get("sample_index")
            bsz = batch_gpu["input_ids"].size(0)

            for i in range(bsz):
                single = {
                    "input_ids": batch_gpu["input_ids"][i:i + 1],
                    "attention_mask": batch_gpu["attention_mask"][i:i + 1],
                    "labels": batch_gpu["labels"][i:i + 1],
                }

                # train_vec: [P] flattened gradient for one train sample
                train_grads = compute_gradients(
                    self.model, single, self.param_filter_fn, self.device, self.ignored_token_ids
                )
                train_vec = self._flatten_grads(train_grads, dtype=torch.float32)
                train_norm = train_vec.norm().clamp_min(1e-12)

                sim = torch.dot(query_vec, train_vec) / (query_norm * train_norm)
                local_scores.append(float(sim.detach().cpu()))
                if indices is not None:
                    local_indices.append(int(indices[i].item()))

            del batch_gpu  # crucial release

        if self.accelerator is None:
            return local_scores, local_indices

        scores_t = torch.tensor(local_scores, device=self.device, dtype=torch.float32)
        indices_t = torch.tensor(local_indices, device=self.device, dtype=torch.long)
        all_scores = self.accelerator.gather(scores_t)
        all_indices = self.accelerator.gather(indices_t)

        if not isinstance(all_scores, Tensor):
            raise ValueError("Expect all_scores to be Tensor")
        if not isinstance(all_indices, Tensor):
            raise ValueError("Expect all_indices to be Tensor")

        return all_scores.tolist(), all_indices.tolist()

class DatasetWrapper(TorchDataset):
    '''
    A dataset wrapper converting HuggingFace `Dataset` to Torch `Dataset`.
    '''
    def __init__(self, hf_dataset):
        self._ds = hf_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


def main_compute_new_inference_function():
    '''
    Deprecated. The workflow to overfit on the target test sample.
    '''
    accelerator = Accelerator()
    set_seed(42)

    model, tokenizer = load_model_and_tokenizer()

    convert_to_chatml_with_tokenizer = partial(process_func_chatml, tokenizer=tokenizer)

    # Load Data
    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")

    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")

    train_ds = build_train_dataset(train_samples, convert_to_chatml_with_tokenizer)

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    # Extracts sample index to the batch property
    collator = CustomCollator(base_collator)

    train_loader = DataLoader(
        DatasetWrapper(train_ds.select(range(100))),  # only check the first 100 training samples
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )

    train_loader = accelerator.prepare(train_loader)

    # Assume: model, tokenizer already loaded
    # Assume: query_batch already built with input_ids and attention_mask

    infer = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        accelerator=accelerator,
        param_filter_fn=filter_params,
        top_k=20,
    )

    # Build query batch
    temp_ds = build_single_sample_dataset(test_samples[18], convert_to_chatml_with_tokenizer)    # choose a proper length sample, or it will cuda oom
    query_batch = base_collator([temp_ds[0]])

    for k, v in query_batch.items():
        query_batch[k] = v.to(accelerator.device)

    # 1) Masked inference for one wrong sample
    result = infer.infer(query_batch)
    print_query_and_answer(result["prev_text"][0], result["answer_text"][0], result["pred_text"][0])

    # 2) Rebuild query batch using prediction as new ground truth
    prompt_len = int(result["target_idx"][0])
    prompt_ids = query_batch["input_ids"][0, :prompt_len]
    pred_ids = torch.tensor(
        result["pred_ids"][0],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype
    )
    new_input_ids = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100  # ignore prompt tokens in loss

    query_batch = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "labels": new_labels
    }

    # 3) Empirical influence (overfit on the new ground-truth batch)
    scores, indices, train_diffs, query_diffs = infer.influence_overfit_single(
        query_batch=query_batch,
        lr=5e-3,
        max_steps=10,
        target_idx=result["target_idx"]
    )

    # select 20 largest then filter out short ones, we cannot compute token level saliency for too long trainging samples
    most_related_samples = nlargest(20, enumerate(scores), key=lambda x: x[1])
    pprint(most_related_samples)
    saliency_analysis_samples = []
    for idx, score in most_related_samples:
        if train_ds["input_ids"][idx].shape[0] <= 2000:
            saliency_analysis_samples.append((idx, score))
    saliency_analysis_samples = saliency_analysis_samples[:10]

    # run inference and dump all results
    dumped_json = {
        "related_train_samples": [],
        "target_test_sample": {}
    }
    dumped_json["target_test_sample"]["before"] = {
        "full_tokens": result["full_tokens"][0],
        "start_index": result["target_idx"][0],
        "saliency_list": result["saliency_original"][0]
    }

    infer.restore_model_params(infer.param_snapshot_overfit)

    result = infer.infer(query_batch)
    dumped_json["target_test_sample"]["after"] = {
        "full_tokens": result["full_tokens"][0],
        "start_index": result["target_idx"][0],
        "saliency_list": result["saliency_original"][0]
    }

    infer.restore_model_params(infer.param_snapshot_original)

    for i, sim in tqdm(saliency_analysis_samples, desc='Analyzing training sample saliency'):
        # Build query batch
        temp_ds = build_single_sample_dataset(train_samples[i], convert_to_chatml_with_tokenizer)    # choose a proper length sample, or it will cuda oom
        query_batch = base_collator([temp_ds[0]])

        for k, v in query_batch.items():
            query_batch[k] = v.to(accelerator.device)

        result_0 = infer.infer(query_batch)

        infer.restore_model_params(infer.param_snapshot_overfit)

        result_1 = infer.infer(query_batch)

        infer.restore_model_params(infer.param_snapshot_original)

        dumped_json["related_train_samples"].append({
            "target_idx": result_0["target_idx"][0],
            "before_original": {
                "full_tokens": result_0["full_tokens"][0],
                "start_index": result_0["target_idx"][0],
                "saliency_list": result_0["saliency_original"][0]
            },
            "before_generation": {
                "full_tokens": result_0["pred_full_tokens"][0],
                "start_index": result_0["target_idx"][0],
                "saliency_list": result_0["saliency_generation"][0]
            },
            "after_original": {
                "full_tokens": result_1["full_tokens"][0],
                "start_index": result_1["target_idx"][0],
                "saliency_list": result_1["saliency_original"][0]
            },
            "after_generation": {
                "full_tokens": result_1["pred_full_tokens"][0],
                "start_index": result_1["target_idx"][0],
                "saliency_list": result_1["saliency_generation"][0]
            }
        })
    
    # compress json size
    dumped_json = round_floats(dumped_json, 5)

    with open('./latest_saliency.json', 'w', encoding = 'utf-8') as f:
        json.dump(dumped_json ,f)


def main_compute_gradient_related_samples():
    '''
    The workflow to compute gradient on target test sample, no overfitting.
    '''

    # Part 1: the preparation remain the same
    
    accelerator = Accelerator()
    set_seed(SEED)

    model, tokenizer = load_model_and_tokenizer()

    convert_to_chatml_with_tokenizer = partial(process_func_chatml, tokenizer=tokenizer)

    # Load Data
    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")

    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")

    train_ds = build_train_dataset(train_samples, convert_to_chatml_with_tokenizer)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= SEQUENCE_LENGTH_LIMIT)   # prevent compute_gradients OOM

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # Extracts sample index to the batch property
    collator = CustomCollator(base_collator)

    train_loader = DataLoader(
        DatasetWrapper(train_ds),  # only check the first 100 training samples
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )

    train_loader = accelerator.prepare(train_loader)

    # Assume: model, tokenizer already loaded
    # Assume: query_batch already built with input_ids and attention_mask

    def filter_lm_head(name: str, param: torch.nn.Parameter):
        return "lm_head.weight" in name and param.requires_grad
        
    for name, param in model.named_parameters():
        if "lm_head.weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    inference_function = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        accelerator=accelerator,
        param_filter_fn=filter_lm_head,
        top_k=20,
    )

    # Build query batch
    temp_ds = build_single_sample_dataset(test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_to_chatml_with_tokenizer)    # choose a proper length sample, or it will cuda oom
    query_batch = base_collator([temp_ds[0]])
    if len(temp_ds[0]['input_ids']) > SEQUENCE_LENGTH_LIMIT:
        raise ValueError(f'This test sample (index {SELECTED_TEST_SAMPLE_INDEX}) has too long text')

    for k, v in query_batch.items():
        query_batch[k] = v.to(accelerator.device)

    dumped_json = {
        "related_train_samples": [],
        "overfit_test_results": [],
        "target_test_sample": {}
    }

    # 1) Inference for one wrong sample
    result = inference_function.infer(query_batch)
    print_query_and_answer(result["prev_text"][0], result["answer_text"][0], result["pred_text"][0])

    dumped_json["target_test_sample"]["before"] = {
        "full_tokens": result["full_tokens"][0],
        "start_index": result["target_idx"][0],
        "saliency_list": result["saliency_original"][0]
    }

    dumped_json["target_test_sample"]["after"] = {
        "full_tokens": result["pred_full_tokens"][0],
        "start_index": result["target_idx"][0],
        "saliency_list": result["saliency_generation"][0]
    }

    # 2) Rebuild query batch using prediction as new ground truth
    # 2) Rebuild query batch using prediction as new ground truth
    prompt_len = int(result["target_idx"][0])
    
    # Use the manually specified token index
    target_token_index = TOKEN_INDEX_TO_RETRIEVE

    prompt_ids = query_batch["input_ids"][0, :prompt_len]
    pred_ids = torch.tensor(
        result["pred_ids"][0],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype
    )
    new_input_ids = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100  # ignore prompt tokens in loss
    
    # We only want to calculate loss on the *target_token_index* token.
    # So we mask everything after it.
    new_labels[:, (target_token_index+1):] = -100
    
    print(f"\n{'='*20} DEBUG INFO {'='*20}")
    print(f"Manually selected target_token_index: {target_token_index}")
    print(f"First generated token index (Auto):   {prompt_len}")
    
    # Context window to inspect
    context_start = max(0, target_token_index - 5)
    context_end = min(new_input_ids.size(1), target_token_index + 6)
    
    print(f"Inspecting tokens around target index {target_token_index}:")
    for idx in range(context_start, context_end):
        if idx >= new_input_ids.size(1):
             break
        token_id = new_input_ids[0, idx].item()
        token_str = tokenizer.decode([token_id])
        # Escape newlines for visibility
        token_repr = token_str.replace('\n', '\\n').replace('\t', '\\t')
        
        prefix = "-> " if idx == target_token_index else "   "
        suffix_auto = " (First generated token)" if idx == prompt_len else ""
        print(f"{prefix}Index {idx:<4}: ID={token_id:<6} Token='{token_repr}'{suffix_auto}")
    print(f"{'='*52}\n")

    query_batch = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "labels": new_labels
    }

    # DEBUG
    # new_labels[:, :(prompt_len+3)] = -100  # ignore prompt tokens in loss
    # original_id = new_input_ids[0][2316]
    # new_input_ids[0][2316] = torch.tensor(tokenizer.convert_tokens_to_ids(['\\n']))[0]
    # result = infer.masked_inference(query_batch, target_idx=torch.tensor([2317], dtype=torch.int64))
    # print_query_and_answer(result["prev_text"][0], result["answer_text"][0], result["pred_text"][0])
    # new_input_ids[0][2316] = original_id

    # Part 2: use the new gradient

    # 3) Empirical influence (overfit on the new ground-truth batch)
    # saved to file, and you can comment this part to prevent retrieving again, because it takes ~1h
    scores, indices = inference_function.influence_gradient_single(
        query_batch=query_batch,
        target_idx=target_token_index
    )
    with open(os.path.join(os.path.dirname(__file__), f'../test_{SELECTED_TEST_SAMPLE_INDEX}_{target_token_index}_result.json'), 'w', encoding='utf-8') as f:
        json.dump({"result": list(zip(indices, scores))}, f)

    # read from saved data
    # select 20 largest then filter out short ones, we cannot compute token level saliency for too long training samples
    with open(os.path.join(os.path.dirname(__file__), f'../test_{SELECTED_TEST_SAMPLE_INDEX}_{target_token_index}_result.json'), 'r', encoding='utf-8') as f:
        most_related_samples_json = json.load(f)
    most_related_samples = most_related_samples_json['result']
    # pprint(most_related_samples)

    # lambda x: x[1] -> top bad samples
    # lambda x: -x[1] -> top good samples
    # lambda x: abs(x[1]) -> top related samples
    saliency_analysis_samples = nlargest(10, most_related_samples, lambda x: -x[1])

    # Get the raw sample texts
    raw_sample_texts = [convert_sample_to_full_text(train_samples[i]) for i, s in saliency_analysis_samples]

    print(f"Calling automated GPT annotation for {len(raw_sample_texts)} samples...")
    # Annotate samples automatically
    annotation_results = annotate_samples(raw_sample_texts, tokenizer)

    # Reconstruct marked_code_samples.md for the record and parse token_samples
    token_samples = []
    marked_blocks_str = ""

    for i, res in enumerate(annotation_results):
        marked_text = res.get("marked_text", "")
        token_result = res.get("token_result", {})

        if not marked_text or not token_result:
            print(f"Warning: Annotation failed for sample {i}.")
            continue
        
        # Append to our string to save back to the markdown file
        marked_blocks_str += f"```go\n{marked_text}\n```\n\n"
        
        # Collect token result directly
        token_samples.append(token_result)

    with open(os.path.join(os.path.dirname(__file__), '../marked_code_samples.md'), 'w', encoding='utf-8') as f:
        f.write(marked_blocks_str)

    print(f"Written annotated samples to marked_code_samples.md.")

    for coef in [1, 10, 100, 1000, 10000]:
        inference_function.save_model_params()

        for epoch in range(10):
            for sample in token_samples:
                finetune_on_sample(
                    model,
                    tokenizer,
                    epochs=1,
                    input_ids=sample['input_ids'],
                    labels=sample['input_ids'].clone(),
                    boost_indices=sample['marked_indices'],
                    boost_coef=coef
                )
        
        result = inference_function.infer(query_batch)
        print(f"Finetuned on boost_coef = {coef}")
        print_query_and_answer(result["prev_text"][0], result["answer_text"][0], result["pred_text"][0])

        dumped_json["overfit_test_results"].append({
            "target_idx": result["target_idx"][0],
            "before_original": {
                "full_tokens": result["pred_full_tokens"][0],
                "start_index": result["target_idx"][0],
                "saliency_list": result["saliency_generation"][0]
            },
            "before_generation": {
                "full_tokens": result["pred_full_tokens"][0],
                "start_index": result["target_idx"][0],
                "saliency_list": result["saliency_generation"][0]
            }
        })
        
        inference_function.restore_model_params()

    # Option 2: record training sample saliency

    for i, sim in tqdm(saliency_analysis_samples, desc='Analyzing training sample saliency'):
        # Build query batch
        temp_ds = build_single_sample_dataset(train_samples[i], convert_to_chatml_with_tokenizer)    # choose a proper length sample, or it will cuda oom
        query_batch = base_collator([temp_ds[0]])

        for k, v in query_batch.items():
            query_batch[k] = v.to(accelerator.device)

        result_0 = inference_function.infer(query_batch)

        dumped_json["related_train_samples"].append({
            "target_idx": result_0["target_idx"][0],
            "before_original": {
                "full_tokens": result_0["full_tokens"][0],
                "start_index": result_0["target_idx"][0],
                "saliency_list": result_0["saliency_original"][0]
            },
            "before_generation": {
                "full_tokens": result_0["pred_full_tokens"][0],
                "start_index": result_0["target_idx"][0],
                "saliency_list": result_0["saliency_generation"][0]
            }
        })
    
    # compress json size
    dumped_json = round_floats(dumped_json, 5)

    # this file is displayed in tools/correlation-report
    # filename encodes the experiment parameters so multiple runs don't overwrite each other
    saliency_filename = f'./saliency_test{SELECTED_TEST_SAMPLE_INDEX}_tok{TOKEN_INDEX_TO_RETRIEVE}.json'
    with open(saliency_filename, 'w', encoding = 'utf-8') as f:
        json.dump(dumped_json, f)
    print(f"\nSaliency results → {saliency_filename}")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute gradient-based influence and saliency for a single test sample."
    )
    parser.add_argument(
        "--test-index", type=int, default=None,
        help="Index of the test sample to analyse (overrides SELECTED_TEST_SAMPLE_INDEX)."
    )
    parser.add_argument(
        "--token-index", type=int, default=None,
        help="Token position to retrieve saliency for (overrides TOKEN_INDEX_TO_RETRIEVE)."
    )
    args = parser.parse_args()

    # Override module-level globals when CLI args are provided
    if args.test_index is not None:
        SELECTED_TEST_SAMPLE_INDEX = args.test_index
    if args.token_index is not None:
        TOKEN_INDEX_TO_RETRIEVE = args.token_index

    print(f"[NIF] test_index={SELECTED_TEST_SAMPLE_INDEX}  token_index={TOKEN_INDEX_TO_RETRIEVE}")
    # main_compute_new_inference_function()
    main_compute_gradient_related_samples()
