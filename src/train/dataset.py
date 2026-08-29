import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import transformers
from torch.utils.data import Dataset
import json

# Make the src/ root importable so the data-side edge utilities resolve the same
# way as elsewhere in the repo (``from data.<mod> import ...``).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.edge_augment import augment_edges, node_target_weights  # noqa: E402
from token_klass import is_protected_completion_token  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Chat template constants (must match postprocessing.py) ────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."
CHAT_PREFIX   = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
CHAT_MIDDLE   = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_SUFFIX   = "<|im_end|>\n"

IGNORE_INDEX = -100


class AnnotatedSFTDataset(Dataset):
    """
    Each item:
        input_ids  : LongTensor [seq_len]
        labels     : LongTensor [seq_len]  (IGNORE_INDEX for input tokens)
        annot_pairs: LongTensor [N, 2]     (qi, qj) — may be empty
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int = 8192, language: str | None = None,
                 edge_augment: bool = False, edge_augment_decay: float = 0.5,
                 edge_augment_max_hops: int = 0, edge_augment_node_weight: bool = False,
                 edge_augment_mode: str = "directed",
                 token_select: bool = False, token_select_threshold: float = 2.0,
                 token_select_keep_special: bool = True,
                 annot_skip: bool = False, annot_skip_keep_first: int = 2,
                 annot_skip_keep_special: bool = True,
                 supervise_eos: bool = True,
                 raw_prompt_response: bool = False,
                 raw_max_samples: int = 0):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.language = (language or "go").lower()
        self.items: list[dict] = []
        # If compact labels mask the trailing <|im_end|>/EOS after the mid
        # completion, unmask it so the model is trained to stop. Default ON —
        # HuaWei oneshot compact currently leaves im_end as -100.
        self.supervise_eos = bool(supervise_eos)
        self._stop_token_ids = self._collect_stop_token_ids(tokenizer)
        self._eos_fixed = 0
        if self.supervise_eos:
            logger.info(
                f"supervise_eos ON: unmask trailing stop ids {sorted(self._stop_token_ids)} "
                f"immediately after the last supervised completion token"
            )

        # TEMP debug path: {prompt, response, task_id} JSONL without compact
        # tokenization / annotations. Intended for quick ce_only smoke runs on
        # the raw enterprise-Go FIM dump; discard once proper compact data is back.
        self.raw_prompt_response = bool(raw_prompt_response)
        self.raw_max_samples = int(raw_max_samples or 0)
        self._raw_skipped = 0
        if self.raw_prompt_response:
            logger.warning(
                "raw_prompt_response ON (TEMPORARY DEBUG): tokenizing "
                "{prompt,response} on the fly; no attention_edges. Prefer "
                "loss_mode=ce_only. raw_max_samples=%s",
                self.raw_max_samples or "all",
            )

        # ── Edge-label augmentation (config-gated, default OFF) ────────────────
        self.edge_augment = bool(edge_augment)
        self.edge_augment_decay = float(edge_augment_decay)
        self.edge_augment_max_hops = int(edge_augment_max_hops)
        self.edge_augment_node_weight = bool(edge_augment_node_weight)
        self.edge_augment_mode = str(edge_augment_mode or "directed")
        if self.edge_augment:
            logger.info(
                f"Edge augmentation ON: {self.edge_augment_mode} closure, decay={self.edge_augment_decay}, "
                f"max_hops={self.edge_augment_max_hops or 'inf'}, "
                f"node_weight={self.edge_augment_node_weight}"
            )

        # ── Teacher-gated informative-token selection (config-gated, default OFF)
        self.token_select = bool(token_select)
        self.token_select_threshold = float(token_select_threshold)
        self.token_select_keep_special = bool(token_select_keep_special)
        self._special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self._ts_excluded = 0
        self._ts_total = 0
        if self.token_select:
            logger.info(
                f"Token selection ON: exclude completion tokens with teacher NLL > "
                f"{self.token_select_threshold} (keep_special={self.token_select_keep_special})"
            )

        # ── Annotation-gated skip with structural protection (default OFF) ─────
        # Skip completion tokens that are not the destination of any annotation
        # edge, EXCEPT protected tokens: first ``keep_first`` completion tokens,
        # and keyword / punct / whitespace (only identifier/number may be dropped).
        self.annot_skip = bool(annot_skip)
        self.annot_skip_keep_first = int(annot_skip_keep_first)
        self.annot_skip_keep_special = bool(annot_skip_keep_special)
        self._as_excluded = 0
        self._as_total = 0
        self._left_truncated = 0
        self._edges_dropped_by_trunc = 0
        if self.annot_skip:
            logger.info(
                f"Annot-skip ON: drop unannotated completion tokens "
                f"(keep_first={self.annot_skip_keep_first}, "
                f"keep_special={self.annot_skip_keep_special}, language={self.language})"
            )
        logger.info(
            f"Overlong sequences: left-truncate to max_len={max_len} "
            f"(keep suffix; remap attention_edges; drop edges that fall outside)"
        )

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if self.raw_max_samples > 0 and len(self.items) >= self.raw_max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                # TEMP: raw enterprise-Go FIM rows {prompt, response, task_id}.
                if self.raw_prompt_response and "prompt" in entry and "response" in entry:
                    built = self._build_from_raw_prompt_response(entry)
                    if built is None:
                        self._raw_skipped += 1
                        continue
                    input_ids, labels = built
                    # No annotations on this path; empty edge list.
                    self.items.append(self._make_item(input_ids, labels, []))
                    continue

                if "input_ids" in entry and ("label" in entry or "labels" in entry):
                    input_ids = [int(x) for x in entry["input_ids"]]
                    labels = [int(x) for x in entry.get("label", entry.get("labels", []))]
                    if not input_ids or len(input_ids) != len(labels):
                        continue
                    edges = entry.get("attention_edges", []) or []
                    input_ids, labels, edges = self._left_truncate_align_edges(
                        input_ids, labels, edges, max_len
                    )
                    if not input_ids or all(l == IGNORE_INDEX for l in labels):
                        continue

                    if self.token_select:
                        self._apply_token_select(entry, input_ids, labels)
                    if self.annot_skip:
                        self._apply_annot_skip(input_ids, labels, edges)
                    if self.supervise_eos:
                        self._ensure_eos_supervised(input_ids, labels)

                    self.items.append(self._make_item(input_ids, labels, edges))
                    continue

                if "task_id" not in entry:
                    continue
                task_id = entry["task_id"]
                lang = task_id.split("/")[0]
                if language is not None and lang.lower() != language.lower():
                    continue

                sft_input = entry.get("sft_input", "")
                tokens = entry.get("qwen_tokens", [])
                annotated_edges = entry.get("qwen_annotations", [])

                if not tokens:
                    continue

                input_ids = [t["token_id"] for t in tokens]
                output_char_start = len(CHAT_PREFIX) + len(sft_input) + len(CHAT_MIDDLE)

                output_token_start = len(input_ids)
                for idx, tok in enumerate(tokens):
                    if tok["char_start"] >= output_char_start:
                        output_token_start = idx
                        break

                labels = [IGNORE_INDEX] * output_token_start + input_ids[output_token_start:]
                input_ids, labels, annotated_edges = self._left_truncate_align_edges(
                    input_ids, labels, annotated_edges or [], max_len
                )
                if not input_ids or all(l == IGNORE_INDEX for l in labels):
                    continue

                if self.annot_skip:
                    self._apply_annot_skip(input_ids, labels, annotated_edges)
                if self.supervise_eos:
                    self._ensure_eos_supervised(input_ids, labels)

                self.items.append(self._make_item(input_ids, labels, annotated_edges))

        logger.info(f"Loaded {len(self.items)} samples from {data_path}")
        if self._left_truncated:
            logger.info(
                f"Left-truncated {self._left_truncated} overlong samples to max_len={max_len}; "
                f"dropped {self._edges_dropped_by_trunc} edges that fell outside the kept window"
            )
        if self.raw_prompt_response:
            logger.info(
                f"raw_prompt_response: kept={len(self.items)} skipped={self._raw_skipped} "
                f"(empty/too-long response or overlong after budget)"
            )
        if self.token_select and self._ts_total:
            logger.info(
                f"Token selection: excluded {self._ts_excluded}/{self._ts_total} scored "
                f"completion tokens ({100*self._ts_excluded/self._ts_total:.1f}%) as missing-info "
                f"(teacher NLL > {self.token_select_threshold})"
            )
        if self.annot_skip and self._as_total:
            logger.info(
                f"Annot-skip: excluded {self._as_excluded}/{self._as_total} completion "
                f"tokens ({100*self._as_excluded/self._as_total:.1f}%) as unannotated "
                f"(kept first {self.annot_skip_keep_first} + keyword/punct/whitespace)"
            )
        if self.supervise_eos and self._eos_fixed:
            logger.info(
                f"supervise_eos: unmasked trailing stop token on {self._eos_fixed}/{len(self.items)} samples"
            )

    def _build_from_raw_prompt_response(self, entry) -> tuple[list[int], list[int]] | None:
        """Tokenize a raw FIM row: prompt (context) + response (mid) + EOS.

        Prompt already carries <PRE>/<SUF>/<MID> + ``### Response:`` — no chat
        template is added (matches eval / newgo compact builder). Labels ignore
        the prompt and supervise response + EOS. Prompt is left-truncated if needed.
        """
        prompt = entry.get("prompt", "") or ""
        resp = entry.get("response", "") or ""
        if not prompt or not resp.strip():
            return None
        tok = self.tokenizer
        eos_id = tok.eos_token_id
        if eos_id is None:
            # Qwen chat end as last resort
            eos_id = tok.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(eos_id, int) or eos_id < 0:
            raise RuntimeError("tokenizer has no usable eos_token_id for raw_prompt_response")

        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        resp_ids = tok(resp, add_special_tokens=False).input_ids
        if not resp_ids:
            return None
        budget = self.max_len - len(resp_ids) - 1
        if budget <= 0:
            return None
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
        n_prompt = len(prompt_ids)
        input_ids = prompt_ids + resp_ids + [int(eos_id)]
        labels = [IGNORE_INDEX] * n_prompt + list(resp_ids) + [int(eos_id)]
        return input_ids, labels

    def _left_truncate_align_edges(self, input_ids, labels, edges, max_len):
        """Keep the last ``max_len`` tokens; shift edge endpoints by the drop.

        Edges whose src/dst land outside ``[0, new_len)`` after the shift are
        dropped (typically context→context that lived entirely in the cropped
        prefix). Completion tokens usually sit near the end, so left-truncate
        preserves MID supervision more often than right-truncate / skip.
        """
        n = len(input_ids)
        if n <= max_len:
            return input_ids, labels, list(edges or [])

        drop = n - max_len
        input_ids = input_ids[drop:]
        labels = labels[drop:]
        new_len = len(input_ids)
        kept_edges = []
        n_in = 0
        for e in edges or []:
            n_in += 1
            qi = e.get("src", e.get("source", e.get("token_i_idx", -1)))
            qj = e.get("dst", e.get("target", e.get("token_j_idx", -1)))
            try:
                qi = int(qi) - drop
                qj = int(qj) - drop
            except (TypeError, ValueError):
                continue
            if not (0 <= qi < qj < new_len):
                continue
            ne = dict(e)
            if "src" in ne or "dst" in ne:
                ne["src"] = qi
                ne["dst"] = qj
            if "source" in ne or "target" in ne:
                ne["source"] = qi
                ne["target"] = qj
            if "token_i_idx" in ne or "token_j_idx" in ne:
                ne["token_i_idx"] = qi
                ne["token_j_idx"] = qj
            kept_edges.append(ne)

        self._left_truncated += 1
        self._edges_dropped_by_trunc += max(0, n_in - len(kept_edges))
        return input_ids, labels, kept_edges

    @staticmethod
    def _collect_stop_token_ids(tokenizer) -> set[int]:
        ids: set[int] = set()
        for tok in ("<|im_end|>", getattr(tokenizer, "eos_token", None)):
            if not tok:
                continue
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
                ids.add(int(tid))
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int) and eos_id >= 0:
            ids.add(int(eos_id))
        return ids

    def _ensure_eos_supervised(self, input_ids, labels):
        """Unmask the stop token right after the last supervised completion token.

        Many compact builders put ``<|im_end|>`` in ``input_ids`` but leave its
        label as IGNORE_INDEX, so CE never teaches the model to stop. If the
        token immediately following the last non-IGNORE label is a known stop
        id and currently ignored, set ``labels[j] = input_ids[j]``.
        """
        if not self._stop_token_ids:
            return
        last = -1
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] != IGNORE_INDEX:
                last = i
                break
        if last < 0:
            return
        j = last + 1
        if j >= len(labels):
            return
        if labels[j] != IGNORE_INDEX:
            return
        if int(input_ids[j]) not in self._stop_token_ids:
            return
        labels[j] = int(input_ids[j])
        self._eos_fixed += 1

    def _apply_token_select(self, entry, input_ids, labels):
        """Exclude 'missing-info' completion tokens from the loss in-place."""
        for pos, nll in entry.get("comp_teacher_nll", []) or []:
            pos = int(pos)
            if not (0 <= pos < len(labels)) or labels[pos] == IGNORE_INDEX:
                continue
            self._ts_total += 1
            if float(nll) <= self.token_select_threshold:
                continue
            if self.token_select_keep_special and int(input_ids[pos]) in self._special_ids:
                continue
            labels[pos] = IGNORE_INDEX
            self._ts_excluded += 1

    def _annotated_dst_positions(self, raw_edges, n_tokens: int) -> set[int]:
        """Positions that appear as an annotation edge destination (dst)."""
        dsts: set[int] = set()
        for ann in raw_edges or []:
            qj = ann.get("dst", ann.get("target", ann.get("token_j_idx", -1)))
            try:
                qj = int(qj)
            except (TypeError, ValueError):
                continue
            if 0 <= qj < n_tokens:
                dsts.add(qj)
        return dsts

    def _apply_annot_skip(self, input_ids, labels, raw_edges):
        """Exclude unannotated completion tokens, with structural protections.

        Drop when ALL of:
          - not an annotation destination,
          - relative completion index >= keep_first,
          - surface class is identifier or number,
          - not a special token (when keep_special).
        """
        n = len(labels)
        annotated_dsts = self._annotated_dst_positions(raw_edges, n)
        comp = [i for i, lab in enumerate(labels) if lab != IGNORE_INDEX]
        for rel, pos in enumerate(comp):
            self._as_total += 1
            if pos in annotated_dsts:
                continue
            if self.annot_skip_keep_special and int(input_ids[pos]) in self._special_ids:
                continue
            surface = self.tokenizer.decode([int(input_ids[pos])])
            if is_protected_completion_token(
                relative_pos=rel,
                surface=surface,
                keep_first=self.annot_skip_keep_first,
                language=self.language,
            ):
                continue
            labels[pos] = IGNORE_INDEX
            self._as_excluded += 1

    def _build_annot(self, raw_edges, n_tokens, labels):
        """Return ``(pairs, weights, node_weight)`` for one sample."""
        norm = []
        for ann in raw_edges:
            qi = ann.get("src", ann.get("source", ann.get("token_i_idx", -1)))
            qj = ann.get("dst", ann.get("target", ann.get("token_j_idx", -1)))
            try:
                qi = int(qi)
                qj = int(qj)
            except (TypeError, ValueError):
                continue
            norm.append({"src": qi, "dst": qj, "subtype": ann.get("subtype", "edge")})

        if self.edge_augment:
            edges = augment_edges(
                norm,
                decay=self.edge_augment_decay,
                max_hops=self.edge_augment_max_hops,
                n_tokens=n_tokens,
                mode=self.edge_augment_mode,
            )
        else:
            edges = norm

        pairs, weights = [], []
        for e in edges:
            qi, qj = int(e["src"]), int(e["dst"])
            if 0 <= qi < qj < n_tokens:
                pairs.append((qi, qj))
                weights.append(float(e.get("weight", 1.0)))

        node_weight = None
        if self.edge_augment and self.edge_augment_node_weight:
            tgt = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
            nw = node_target_weights(edges, tgt)
            node_weight = [0.0] * n_tokens
            for p, w in nw.items():
                if 0 <= p < n_tokens:
                    node_weight[p] = float(w)

        if not self.edge_augment:
            weights = None
        return pairs, weights, node_weight

    def _make_item(self, input_ids, labels, raw_edges):
        """Build one dataset item dict, attaching augmentation tensors if on."""
        n = len(input_ids)
        pairs, weights, node_weight = self._build_annot(raw_edges, n, labels)
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "annot_pairs": torch.tensor(pairs, dtype=torch.long)
            if pairs else torch.zeros(0, 2, dtype=torch.long),
        }
        if weights is not None:
            item["annot_weights"] = (torch.tensor(weights, dtype=torch.float)
                                     if weights else torch.zeros(0, dtype=torch.float))
        if node_weight is not None:
            item["node_weight"] = torch.tensor(node_weight, dtype=torch.float)
        return item

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@dataclass
class DataCollatorForAnnotatedSFT:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]
        annot_pairs_list = [inst.get("annot_pairs", torch.zeros(0, 2, dtype=torch.long))
                            for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True,
            padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "annot_pairs": annot_pairs_list,
        }

        if "annot_weights" in instances[0]:
            batch["annot_weights"] = [
                inst.get("annot_weights", torch.zeros(0, dtype=torch.float))
                for inst in instances
            ]
        if "node_weight" in instances[0]:
            node_weight = torch.nn.utils.rnn.pad_sequence(
                [inst["node_weight"] for inst in instances],
                batch_first=True, padding_value=0.0,
            )
            batch["node_weight"] = node_weight
        return batch
