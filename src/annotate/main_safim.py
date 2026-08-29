from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotate.utils import tokenize_code_for_annotation
from src.annotate.neural_annot import AnnotatorAgent
from src.annotate.viz_utils import *
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")

TASK_SUFFIXES = {"multi", "single", "span", "light-span"}
_SAFIM_LANG_MAP = {
    "java":   "Java",
    "cpp":    "CPP",
    "python": "Python",
    "csharp": "C#",
    "c#":     "C#",
    "go":     "Go",
    "c":      "C",
}

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_dict(obj):
    """Serialize dataclass or plain object to dict."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def _flush(entries: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def get_base_id(task_id: str) -> str:
    # Step 1: strip suffix (e.g. "-multi", "-light-span")
    for suffix in TASK_SUFFIXES:
        if task_id.endswith("-" + suffix):
            task_id = task_id[: -(len(suffix) + 1)]
            break

    # Step 2: strip trailing numeric index (e.g. "-0", "-1")
    parts = task_id.rsplit("-", 1)
    if len(parts) == 2 and parts[-1].isdigit():
        return parts[0]

    return task_id


def match_lang(lang: str) -> str | None:
    if "Python" in lang:
        return "Python"
    elif "C#" in lang:
        return "C#"
    elif "CPP" in lang:
        return "CPP"
    elif "Go" in lang:
        return "Go"
    elif "Java" in lang and "JavaScript" not in lang:
        return "Java"
    elif lang == "C":
        return "C"
    return None


def annotate_one(base_id: str, code: str, matched: str, max_rounds: int = 6) -> tuple[str, list, list]:
    """Annotate a single base_id. Returns (base_id, tokens_out, annots_out)."""
    subwords = tokenize_code_for_annotation(code)
    ann = AnnotatorAgent(language=matched, max_rounds=max_rounds)
    neu_sw = ann.annotate(code, subwords)

    html_path = f"./debug_safim/{base_id.replace('/', '_')}.html"
    visualize_correlations(
        neu_sw,
        title="Token Correlation · Attention View",
        code=code,
        subwords=subwords,
        output_path=html_path,
        open_browser=False,
    )

    tokens_out = [_to_dict(sw) for sw in subwords]
    annots_out = [_to_dict(c) for c in neu_sw]
    return base_id, tokens_out, annots_out


def safim_match_lang(lang: str) -> str | None:
    """Normalize SAFIM's lang field (lowercase) to our matched lang."""
    return _SAFIM_LANG_MAP.get(lang.lower())


def safim_reconstruct_code(entry: dict) -> str:
    prompt: str = entry["prompt"]
    ground_truth: str = entry["ground_truth"]

    todo_pattern = re.compile(r'/\*\s*TODO[^*]*\*/', re.DOTALL)
    if todo_pattern.search(prompt):
        return todo_pattern.sub(lambda m: ground_truth, prompt)  # lambda avoids escape processing

    return prompt + "\n" + ground_truth

if __name__ == "__main__":

    JSONL_PATH = "./data/safim/safim-train.jsonl"
    MAX_WORKERS = 8

    data = load_jsonl(JSONL_PATH)

    # ── Pass 1: collect unique task_ids that need annotation ─────────────────
    # SAFIM task_ids are already unique per entry (no suffix variants like MCEval)
    todo: dict[str, tuple[str, str]] = {}
    annotation_cache: dict[str, tuple] = {}

    for entry in data:
        task_id = entry["task_id"]
        lang_raw = entry.get("lang", "")
        matched = safim_match_lang(lang_raw)
        if matched is None:
            continue

        if "tokens" in entry and "annotations" in entry:
            if task_id not in annotation_cache:
                annotation_cache[task_id] = (entry["tokens"], entry["annotations"])
            continue

        if task_id not in annotation_cache and task_id not in todo:
            code = safim_reconstruct_code(entry)
            todo[task_id] = (code, matched)

    print(f"[plan] {len(annotation_cache)} already annotated, {len(todo)} to annotate")

    # ── Pass 2: parallel annotation ───────────────────────────────────────────
    flush_lock = threading.Lock()
    ct = 0

    def _annotate_and_cache(item):
        task_id, (code, matched) = item
        try:
            task_id, tokens_out, annots_out = annotate_one(task_id, code, matched)
            return task_id, tokens_out, annots_out, None
        except Exception as e:
            return task_id, None, None, e

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_annotate_and_cache, item): item[0] for item in todo.items()}

        for future in as_completed(futures):
            task_id, tokens_out, annots_out, err = future.result()
            if err:
                print(f"[error] {task_id}: {err}")
                continue

            annotation_cache[task_id] = (tokens_out, annots_out)
            ct += 1
            print(f"[done] {task_id} ({ct}/{len(todo)})")

            if ct % 10 == 0:
                with flush_lock:
                    for entry in data:
                        tid = entry["task_id"]
                        if tid in annotation_cache and "tokens" not in entry:
                            entry["tokens"], entry["annotations"] = annotation_cache[tid]
                    _flush(data, JSONL_PATH)
                    print(f"[flush] saved checkpoint at {ct} annotations")

    # ── Pass 3: apply cache to all entries and final flush ────────────────────
    lang_ct: dict[str, int] = {}
    for entry in data:
        task_id = entry["task_id"]
        lang_raw = entry.get("lang", "")
        matched = safim_match_lang(lang_raw)
        if matched is None:
            continue

        if task_id in annotation_cache and "tokens" not in entry:
            entry["tokens"], entry["annotations"] = annotation_cache[task_id]
        if "tokens" in entry:
            lang_ct[matched] = lang_ct.get(matched, 0) + 1

    _flush(data, JSONL_PATH)
    print(f"Done. {ct} entries annotated → {JSONL_PATH}")
    print(f"Per-lang counts: {lang_ct}")