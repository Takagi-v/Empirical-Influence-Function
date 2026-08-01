import argparse
import json
import os
import re
from urllib import request as urllib_request

import numpy as np
from sklearn.decomposition import PCA


DEFAULT_TTAV_UPLOAD_URL = "http://1.94.115.154/registerEIFBundle"
_UMAP_MIN_POINTS = 6  # fall back to PCA for very small token sequences


# Trailing tags that record how the attribution analysis was parameterised, not
# which sample it is. They don't change the tokens or the embeddings, and the
# generator strips them when naming bundles, so keeping them here would send
# every lookup to a directory that doesn't exist. Other suffixes (e.g. `_new`)
# do distinguish samples and are kept.
_ANALYSIS_PARAM_SUFFIX = re.compile(r"^salr[\d\-]+$", re.IGNORECASE)


def infer_sample_id(report_json_path: str) -> str:
    stem = os.path.splitext(os.path.basename(report_json_path))[0]
    match = re.match(r"^correlation_matching_results_(.+?)_all_tokens(?:_(.+))?$", stem)
    if not match:
        return stem
    prefix = match.group(1)
    suffix = match.group(2)
    if not suffix or _ANALYSIS_PARAM_SUFFIX.match(suffix):
        return prefix
    return f"{prefix}_{suffix}"


def decode_token(token: str) -> str:
    return token.replace("Ċ", "\n").replace("Ġ", " ").replace("ĉ", "  ")


def normalize_token_for_display(token: str) -> str:
    decoded = decode_token(token).replace("\r", "")
    if decoded == "\n":
        return "↵"
    if decoded == "\t":
        return "⇥"
    if len(decoded) == 0:
        return "·"

    visible = decoded.replace("\n", "↵").replace("\t", "⇥")
    visible = re.sub(r" {2,}", lambda m: "␠" * len(m.group(0)), visible)

    if len(visible.strip()) == 0:
        return visible.replace(" ", "␠") or "·"
    return visible


def build_short_token_label(token: str, idx: int, role: str) -> str:
    role_tag = "P" if role == "prompt" else "O"
    normalized = normalize_token_for_display(token)
    shortened = normalized[:15] + "…" if len(normalized) > 18 else normalized
    return f"{role_tag}{idx}: {shortened}"


def compute_projection(embeddings: np.ndarray, use_umap: bool = False) -> np.ndarray:
    num_points = embeddings.shape[0]
    if num_points == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if num_points == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if embeddings.shape[1] == 1:
        return np.concatenate([embeddings.astype(np.float32), np.zeros((num_points, 1), dtype=np.float32)], axis=1)

    if use_umap and num_points >= _UMAP_MIN_POINTS:
        try:
            import umap
            n_neighbors = min(15, num_points - 1)
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            proj = reducer.fit_transform(embeddings)
            return proj.astype(np.float32)
        except Exception:
            pass  # fall through to PCA

    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(embeddings)
    return proj.astype(np.float32)


def build_static_token_features(report: dict) -> np.ndarray:
    report_tokens = report["test_sample_baseline"]["full_tokens"]
    prompt_len = int(report["test_sample_baseline"]["prompt_len"])
    per_token_results = report.get("per_token_results", [])
    num_tokens = len(report_tokens)
    if num_tokens == 0:
        return np.zeros((0, 6), dtype=np.float32)

    max_source_saliency = np.zeros(num_tokens, dtype=np.float32)
    source_ref_count = np.zeros(num_tokens, dtype=np.float32)
    target_pair_score = np.zeros(num_tokens, dtype=np.float32)
    analyzed_target = np.zeros(num_tokens, dtype=np.float32)

    for token_result in per_token_results:
        tgt_idx = int(token_result["target_token_index"])
        if 0 <= tgt_idx < num_tokens:
            analyzed_target[tgt_idx] = 1.0
            best_pair = max((float(pair.get("cos_sim", 0.0)) for pair in token_result.get("correlation_pairs", [])), default=0.0)
            target_pair_score[tgt_idx] = max(target_pair_score[tgt_idx], best_pair)

        for corr in token_result.get("top_correlations", []):
            src_idx = int(corr["source_token_index"])
            if 0 <= src_idx < num_tokens:
                saliency = float(corr.get("saliency_score", 0.0))
                max_source_saliency[src_idx] = max(max_source_saliency[src_idx], saliency)
                source_ref_count[src_idx] += 1.0

    max_count = float(source_ref_count.max()) if source_ref_count.size else 0.0
    if max_count > 0:
        source_ref_count /= max_count

    token_lengths = np.array([min(len(tok), 16) / 16.0 for tok in report_tokens], dtype=np.float32)
    positions = np.linspace(0.0, 1.0, num_tokens, dtype=np.float32)
    is_output = np.array([1.0 if idx >= prompt_len else 0.0 for idx in range(num_tokens)], dtype=np.float32)

    return np.stack([
        positions,
        is_output,
        analyzed_target,
        max_source_saliency,
        target_pair_score,
        token_lengths,
    ], axis=1).astype(np.float32)


def build_bundle_payload(
    report_json_path: str,
    test_data_path: str | None,
    model_path: str | None,
    sample_id: str | None = None,
) -> dict:
    with open(report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    sample_id = sample_id or infer_sample_id(report_json_path)
    test_index = int(report["experiment_meta"]["test_sample_index"])
    report_tokens = report["test_sample_baseline"]["full_tokens"]
    prompt_len = int(report["test_sample_baseline"]["prompt_len"])
    embeddings = build_static_token_features(report)

    labels = [0 if i < prompt_len else 1 for i in range(len(report_tokens))]
    text_list = []
    token_list = []
    for idx, token in enumerate(report_tokens):
        role = "prompt" if idx < prompt_len else "output"
        text_list.append(build_short_token_label(token, idx, role))
        token_list.append(normalize_token_for_display(token))

    projection = compute_projection(embeddings)

    return {
        "sample_id": sample_id,
        "vis_method": "TimeVis",
        "vis_id": "1",
        "overwrite": True,
        "bundle": {
            "model": "EIFStaticTokenBundle",
            "classes": ["prompt", "output"],
            "sample_index": test_index,
            "prompt_len": prompt_len,
            "labels": labels,
            "text_list": text_list,
            "text_data": text_list,
            "token_list": token_list,
            "index": {"train": list(range(len(report_tokens))), "test": []},
            "embeddings": embeddings.astype(np.float32).tolist(),
            "projection": projection.tolist(),
        },
    }


def upload_bundle(upload_url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        upload_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Export an EIF sample as a TTAV token bundle and upload it.")
    parser.add_argument("--report-json", required=True, help="Path to correlation_matching_results_*_all_tokens*.json")
    parser.add_argument("--test-data", default="sft_test.jsonl", help="Path to the EIF test JSONL file.")
    parser.add_argument("--model-path", default=None, help="Optional model checkpoint path for EIF inference.")
    parser.add_argument("--sample-id", default=None, help="Override the inferred sample id used for TTAV bundle naming.")
    parser.add_argument("--ttav-upload-url", default=DEFAULT_TTAV_UPLOAD_URL, help="TTAV backend upload endpoint.")
    args = parser.parse_args()

    payload = build_bundle_payload(
        report_json_path=args.report_json,
        test_data_path=args.test_data,
        model_path=args.model_path,
        sample_id=args.sample_id,
    )
    result = upload_bundle(args.ttav_upload_url, payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
