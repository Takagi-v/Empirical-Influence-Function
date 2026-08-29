"""Unified evaluation entry point — config-driven, one interface for all sets.

    # evaluate an experiment's trained checkpoint on its configured datasets
    python src/eval/run_eval.py --config configs/experiments/<exp>.yaml

    # or ad-hoc: any model on any registry datasets
    python src/eval/run_eval.py --model <ckpt_or_base> --name <tag> \
        --datasets csn_test,mceval,humaneval_x [--num_samples 5]

Policy (per dataset, from the registry):
    has_tests == true   -> pass@1, pass@<num_samples>, CodeBLEU
    has_tests == false  -> CodeBLEU only

All generation eval uses the dataset's CHATML representation (clean
prefix/suffix) via the execution judge in scripts/benchmark/benchmark_eval.py.
Outputs a uniform schema:
    outputs/<name>/eval/<dataset>.json     per-dataset detail (+ raw/ subdir)
    outputs/<name>/eval/summary.json       one row per dataset, comparable

Note: benchmark_eval reports the sampled-pass under the legacy key "pass@10"
whose VALUE is "any-of-num_samples passes". run_eval relabels it to
"pass@<num_samples>" so the reported k matches what was actually generated.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
SRC = THIS.parents[1]
REPO = THIS.parents[2]
sys.path.insert(0, str(SRC))

from common import ExpPaths, load_experiment   # noqa: E402
from data.registry import get_dataset, default_model   # noqa: E402

BENCH = REPO / "scripts" / "benchmark" / "benchmark_eval.py"


def _find_run_meta(model_path: str) -> dict | None:
    """Walk up from a checkpoint/model path to the training run's meta.json
    (written by run_train.py at outputs/<name>/meta.json), so eval results can
    inherit the run's free-text `description`."""
    p = Path(str(model_path))
    for cand in [p, *p.parents]:
        m = cand / "meta.json"
        if m.exists():
            try:
                return json.loads(m.read_text())
            except Exception:
                return None
        if cand.name == "outputs":
            break
    return None


def run_one(model: str, spec, eval_dir: Path, num_samples: int, codebleu: bool,
            max_rows: int = 0, judge_timeout_sec: int = 10, judge_workers: int = 8) -> dict:
    """Run benchmark_eval on one dataset; return the unified per-dataset record."""
    chatml = spec.resolve("chatml")
    raw_dir = eval_dir / "raw" / spec.name
    raw_dir.mkdir(parents=True, exist_ok=True)
    # has_tests=false -> greedy only (pass is unsupported anyway), still get CodeBLEU.
    ns = num_samples if spec.has_tests else 1
    argv = [
        sys.executable, str(BENCH),
        "--model_path", str(model),
        "--eval_path", str(chatml),
        "--output_dir", str(raw_dir),
        "--baseline_name", spec.name,
        "--languages", spec.language,
        "--num_samples", str(ns),
        "--temperature", "0.2", "--top_p", "0.95",
        "--infer_batch_size", os.environ.get("INFER_BS", "8"),
        "--sample_infer_batch_size", os.environ.get("SAMPLE_BS", "4"),
        "--judge_workers", str(judge_workers), "--judge_timeout_sec", str(judge_timeout_sec),
        "--predictions_out", str(raw_dir / "predictions.jsonl"),
    ]
    if spec.source_dataset:
        argv += ["--source_datasets", spec.source_dataset]
    if codebleu:
        argv += ["--compute_codebleu"]
    if max_rows:
        argv += ["--max_rows", str(max_rows)]

    log = raw_dir / "run.log"
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO))
    rec = {"dataset": spec.name, "has_tests": spec.has_tests, "exit": proc.returncode}

    bj = raw_dir / f"{spec.name}_benchmark_eval.json"
    if not bj.exists():
        rec["error"] = f"benchmark json missing (see {log})"
        return rec
    d = json.loads(bj.read_text())
    overall = d.get("overall", {})
    rec["n_total"] = overall.get("n_total")
    rec["n_judged"] = overall.get("n_judged")
    if spec.has_tests:
        rec["pass@1"] = overall.get("pass@1")
        # legacy key "pass@10" VALUE == any-of-num_samples; relabel to true k.
        rec[f"pass@{num_samples}"] = overall.get("pass@10")
    if codebleu and "codebleu" in d:
        rec["codebleu"] = d["codebleu"].get("overall", {}).get("codebleu")
    # copy the cleaned per-dataset json next to summary
    (eval_dir / f"{spec.name}.json").write_text(json.dumps(
        {"unified": rec, "raw_overall": overall,
         "codebleu": d.get("codebleu", {}).get("overall") if codebleu else None}, indent=2))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified config-driven evaluation.")
    ap.add_argument("--config", help="experiment YAML (model+datasets from it)")
    ap.add_argument("--model", help="ad-hoc: checkpoint dir or 'base'")
    ap.add_argument("--name", help="ad-hoc: output experiment name")
    ap.add_argument("--datasets", help="comma-separated registry dataset names; "
                    "in --config mode this OVERRIDES the config's eval.datasets "
                    "(e.g. 'mceval,humaneval_x' to skip the slow csn_test)")
    ap.add_argument("--num_samples", type=int, default=5, help="pass@k k (default 5)")
    ap.add_argument("--no_codebleu", action="store_true")
    ap.add_argument("--max_rows", type=int, default=0, help="debug: cap rows per dataset")
    ap.add_argument("--judge_timeout_sec", type=int, default=10,
                    help="per-test execution-judge timeout (s); raise for compiled langs "
                    "(e.g. go) on a loaded node to avoid false-zero timeouts")
    ap.add_argument("--judge_workers", type=int, default=8,
                    help="parallel judge workers; lower to reduce go build-cache contention")
    ap.add_argument("--description", help="free-text run intention to record in "
                    "summary.json (overrides config/meta description)")
    args = ap.parse_args()

    desc = None
    if args.config:
        cfg = load_experiment(args.config)
        name = cfg["name"]
        desc = cfg.get("description")
        paths = ExpPaths.for_experiment(name).mkdirs()
        ev = cfg.get("eval", {}) or {}
        # Eval ALWAYS targets the trained checkpoint (outputs/<name>/checkpoints).
        # The top-level `model:` is the TRAINING base model, NOT the eval target —
        # using it here would (wrongly) evaluate the untrained base. Allow an
        # explicit override only via eval.model.
        model = ev.get("model") or str(paths.checkpoints)
        ds_names = ev.get("datasets", [])
        num_samples = int(ev.get("num_samples", args.num_samples))
        codebleu = bool(ev.get("codebleu", True)) and not args.no_codebleu
        # --datasets on the CLI overrides the config's dataset list (e.g. to skip
        # the slow csn_test and only run mceval,humaneval_x).
        if args.datasets:
            ds_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        if not (args.model and args.name and args.datasets):
            ap.error("ad-hoc mode needs --model, --name and --datasets")
        name = args.name
        paths = ExpPaths.for_experiment(name).mkdirs()
        model = args.model
        ds_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
        num_samples = args.num_samples
        codebleu = not args.no_codebleu

    # 'base' is a convenience alias for the registry default base model.
    if str(model) == "base":
        model = default_model()

    # description priority: --description > config's description > source run meta.json
    desc = args.description or desc
    if not desc:
        desc = (_find_run_meta(model) or {}).get("description")

    print(f"[run_eval] {name}  model={model}")
    print(f"[run_eval] datasets={ds_names}  pass@k k={num_samples}  codebleu={codebleu}")
    if desc:
        print(f"[run_eval] description: {desc}")

    summary = {"name": name, "model": str(model), "num_samples": num_samples,
               "description": desc, "results": []}
    for dn in ds_names:
        spec = get_dataset(dn)
        print(f"[run_eval] -> {dn} (has_tests={spec.has_tests}) ...", flush=True)
        rec = run_one(model, spec, paths.eval, num_samples, codebleu, args.max_rows,
                      judge_timeout_sec=args.judge_timeout_sec, judge_workers=args.judge_workers)
        summary["results"].append(rec)
        print(f"           {json.dumps({k: v for k, v in rec.items() if k not in ('dataset',)})}")

    # Merge with any previously-evaluated datasets (don't drop results for sets
    # not included in this run, e.g. a slow csn_test done earlier).
    prev_path = paths.eval / "summary.json"
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text())
            have = {r["dataset"] for r in summary["results"]}
            for r in prev.get("results", []):
                if r.get("dataset") not in have:
                    summary["results"].append(r)
        except Exception:
            pass
    summary["results"].sort(key=lambda r: r.get("dataset", ""))
    prev_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[run_eval] summary -> {paths.eval / 'summary.json'}")
    # pretty table
    print(f"\n{'dataset':16s} | {'pass@1':>8s} {'pass@'+str(num_samples):>8s} {'codebleu':>9s}")
    for r in summary["results"]:
        p1 = r.get("pass@1"); pk = r.get(f"pass@{num_samples}"); cb = r.get("codebleu")
        print(f"{r['dataset']:16s} | {('%.4f'%p1) if p1 is not None else '   -   ':>8s} "
              f"{('%.4f'%pk) if pk is not None else '   -   ':>8s} "
              f"{('%.4f'%cb) if cb is not None else '   -   ':>9s}")


if __name__ == "__main__":
    main()
