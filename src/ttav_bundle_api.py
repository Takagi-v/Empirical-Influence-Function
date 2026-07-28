import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import time
from urllib.parse import parse_qs, urlparse

import numpy as np

from src.export_real_ttav_bundle import DEFAULT_MODEL_PATH, build_real_bundle_payload
from src.export_train_probe_bundle import build_train_probe_bundle_payload
from src.export_ttav_bundle import build_bundle_payload, infer_sample_id, upload_bundle


REPO_ROOT = Path(__file__).resolve().parent.parent
CORR_RESULTS_DIR = REPO_ROOT / "correlation_matching_results"
EIF_BUNDLE_CACHE_ROOT = REPO_ROOT / "ttav_bundles"
PREGENERATED_REAL_BUNDLE_ROOT = REPO_ROOT / "ttav_bundles_real"
PREPARE_STATUS_LOCK = Lock()
PREPARE_STATUS: dict[str, dict] = {}


def _set_prepare_status(sample_id: str, stage: str, message: str, *, active: bool, error: bool = False):
    payload = {
        "sampleId": sample_id,
        "stage": stage,
        "message": message,
        "active": active,
        "error": error,
        "updatedAt": int(time() * 1000),
    }
    with PREPARE_STATUS_LOCK:
        PREPARE_STATUS[sample_id] = payload
    print(f"[{sample_id}] {stage}: {message}", flush=True)


def _get_prepare_status(sample_id: str) -> dict:
    with PREPARE_STATUS_LOCK:
        payload = PREPARE_STATUS.get(sample_id)
    if payload is None:
        return {
            "sampleId": sample_id,
            "stage": "idle",
            "message": "No active prepare job.",
            "active": False,
            "error": False,
            "updatedAt": int(time() * 1000),
        }
    return dict(payload)


def _cache_dir(sample_id: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return EIF_BUNDLE_CACHE_ROOT / sample_id


def _cache_payload_path(sample_id: str, explicit_path: str | None = None) -> Path:
    return _cache_dir(sample_id, explicit_path) / "bundle_payload.json"


def _resolve_cache_payload_path(sample_id: str, explicit_path: str | None = None) -> Path:
    payload_path = _cache_payload_path(sample_id, explicit_path)
    if payload_path.exists():
        return payload_path

    # A caller-supplied explicit_path (from the frontend's "EIF Bundle Cache
    # Path" field / localStorage) can go stale or point at a path that only
    # ever existed on a different machine. Rather than treat that as a hard
    # cache miss and fall through to a live model load -- which requires
    # loading a ~7GB checkpoint on this 7.1GB-RAM box and reliably OOMs, see
    # HANDOFF_2026-07-13_EIF_TTAV_OOM.md -- fall back to the server's own
    # default cache locations first.
    if explicit_path:
        default_path = _cache_payload_path(sample_id, None)
        if default_path.exists():
            print(f"[prepare] sampleId={sample_id} cache=default_fallback (explicit path missing: {explicit_path})", flush=True)
            return default_path

    pregenerated_path = PREGENERATED_REAL_BUNDLE_ROOT / sample_id / "bundle_payload.json"
    if pregenerated_path.exists():
        print(f"[prepare] sampleId={sample_id} cache=pregenerated_real_bundle", flush=True)
        return pregenerated_path

    return payload_path


def write_local_bundle_cache(sample_id: str, payload: dict, explicit_path: str | None = None):
    cache_dir = _cache_dir(sample_id, explicit_path)
    bundle = payload["bundle"]
    vis_method = str(payload.get("vis_method", "TimeVis"))
    vis_id = str(payload.get("vis_id", "1"))
    num_points = len(bundle["labels"])

    dataset_dir = cache_dir / "dataset"
    epoch_dir = cache_dir / "epochs" / "epoch_1"
    vis_info_dir = cache_dir / "visualize" / f"{vis_method}_{vis_id}"
    vis_dir = vis_info_dir / "epochs" / "epoch_1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    (_cache_payload_path(sample_id, explicit_path)).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_info = {
        "model": bundle.get("model", "EIFStaticTokenBundle"),
        "classes": bundle.get("classes", ["prompt", "output"]),
        "eif_bundle": True,
        "sample_id": sample_id,
        "prompt_len": bundle.get("prompt_len"),
        "cache_source": "EIF",
    }
    (dataset_dir / "info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(dataset_dir / "labels.npy", np.asarray(bundle["labels"], dtype=np.int64))
    (dataset_dir / "index.json").write_text(
        json.dumps(bundle.get("index", {"train": list(range(num_points)), "test": []}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dataset_dir / "text.txt").write_text("\n".join(str(x) for x in bundle["text_list"]), encoding="utf-8")
    (dataset_dir / "token_list.json").write_text(json.dumps(bundle.get("token_list", []), ensure_ascii=False), encoding="utf-8")
    (dataset_dir / "text_data.json").write_text(json.dumps(bundle.get("text_data", []), ensure_ascii=False), encoding="utf-8")

    np.save(epoch_dir / "embeddings.npy", np.asarray(bundle["embeddings"], dtype=np.float32))
    np.save(vis_dir / "projection.npy", np.asarray(bundle["projection"], dtype=np.float32))

    vis_info = {
        "content_path": str(cache_dir),
        "vis_method": vis_method,
        "vis_id": vis_id,
        "data_type": "Text",
        "task_type": "Alignment",
        "vis_config": {"gpu_id": -1},
        "sample_id": sample_id,
        "eif_bundle": True,
        "cache_source": "EIF",
    }
    (vis_info_dir / "info.json").write_text(json.dumps(vis_info, ensure_ascii=False, indent=2), encoding="utf-8")


def load_local_bundle_cache(sample_id: str, explicit_path: str | None = None) -> dict:
    payload_path = _resolve_cache_payload_path(sample_id, explicit_path)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _make_probe_sample_id(
    base_sample_id: str,
    train_sample_id: int,
    probe_pairs: list[dict],
    context_radius: int,
    include_full_train: bool,
    focus_train_indices: list[int] | None,
) -> str:
    signature = json.dumps(
        {
            "trainSampleId": train_sample_id,
            "contextRadius": context_radius,
            "includeFullTrain": include_full_train,
            "focusTrainIndices": [int(idx) for idx in (focus_train_indices or [])],
            "pairs": [
                {
                    "id": pair.get("id"),
                    "trainSourceIndex": pair.get("trainSourceIndex"),
                    "trainTargetIndex": pair.get("trainTargetIndex"),
                    "testSourceIndex": pair.get("testSourceIndex"),
                    "testTargetIndex": pair.get("testTargetIndex"),
                }
                for pair in probe_pairs
            ],
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return f"{base_sample_id}_train{train_sample_id}_probe_{digest}"


def _default_probe_cache_path(base_sample_id: str, train_sample_id: int, probe_sample_id: str) -> str:
    return str(EIF_BUNDLE_CACHE_ROOT / "probes" / base_sample_id / f"train_{train_sample_id}" / probe_sample_id)


def _payload_matches_request(payload: dict, bundle_mode: str, embedding_type: str, model_path: str | None) -> bool:
    bundle = payload.get("bundle") if isinstance(payload, dict) else None
    if not isinstance(bundle, dict):
        return False

    payload_embedding_type = str(bundle.get("embedding_type", "")).strip().lower()
    payload_model_path = str(bundle.get("checkpoint_path", "")).strip()

    if bundle_mode == "real":
        if payload_embedding_type != embedding_type:
            return False
        if model_path and payload_model_path and payload_model_path != model_path:
            return False
        return True

    return payload_embedding_type in {"", "static"}


def _build_requested_payload(
    report_json_path: Path,
    test_data: str | None,
    model_path: str | None,
    sample_id: str | None,
    bundle_mode: str,
    embedding_type: str,
    hidden_layer: int,
    vis_method: str,
    vis_id: str,
    progress_callback=None,
) -> dict:
    if bundle_mode == "real":
        resolved_model_path = model_path or str(DEFAULT_MODEL_PATH)
        return build_real_bundle_payload(
            report_json_path=str(report_json_path),
            model_path=resolved_model_path,
            sample_id=sample_id,
            embedding_type=embedding_type,
            hidden_layer=hidden_layer,
            vis_method=vis_method,
            vis_id=vis_id,
            progress_callback=progress_callback,
        )

    return build_bundle_payload(
        report_json_path=str(report_json_path),
        test_data_path=test_data,
        model_path=model_path,
        sample_id=sample_id,
    )


class TTAVBundleRequestHandler(BaseHTTPRequestHandler):
    server_version = "EIFTTAVBundleAPI/0.1"

    def log_message(self, format: str, *args):
        print(f"[http] {self.address_string()} - {format % args}", flush=True)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/prepare-ttav-bundle-status":
            self._send_json(404, {"status": "error", "message": "Not found"})
            return

        sample_id = parse_qs(parsed.query).get("sampleId", [""])[0].strip()
        if not sample_id:
            self._send_json(400, {"status": "error", "message": "sampleId is required"})
            return

        print(f"[status] sampleId={sample_id}", flush=True)
        self._send_json(200, {"status": "success", **_get_prepare_status(sample_id)})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/prepare-ttav-train-probe":
            self._handle_prepare_train_probe()
            return
        if parsed.path != "/api/prepare-ttav-bundle":
            self._send_json(404, {"status": "error", "message": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        report_file_name = str(req.get("reportFileName", "")).strip()
        if not report_file_name:
            self._send_json(400, {"status": "error", "message": "reportFileName is required"})
            return

        report_json_path = CORR_RESULTS_DIR / report_file_name
        if not report_json_path.exists():
            self._send_json(404, {"status": "error", "message": f"Report JSON not found: {report_file_name}"})
            return

        test_data = str(req.get("testData", "sft_test.jsonl")).strip() or "sft_test.jsonl"
        raw_model_path = req.get("modelPath")
        model_path = str(raw_model_path).strip() if raw_model_path else None
        sample_id = req.get("sampleId")
        ttav_upload_url = str(req.get("ttavUploadUrl", "")).strip()
        ttav_url = str(req.get("ttavUrl", "")).strip()
        vis_method = str(req.get("visMethod", "TimeVis")).strip() or "TimeVis"
        vis_id = str(req.get("visId", "1")).strip() or "1"
        selected_indices = req.get("selectedIndices", [])
        target_index = req.get("targetIndex")
        require_cached = bool(req.get("requireCached", False))
        explicit_cache_path = str(req.get("eifBundleCachePath", "")).strip() or None
        bundle_mode = str(req.get("bundleMode", "real")).strip().lower() or "real"
        embedding_type = str(req.get("embeddingType", "contextual")).strip().lower() or "contextual"
        hidden_layer = int(req.get("hiddenLayer", -1))
        overwrite_req = req.get("overwrite")
        overwrite_remote = bool(overwrite_req) if overwrite_req is not None else (bundle_mode == "real")
        resolved_sample_id = str(sample_id).strip() if sample_id else infer_sample_id(str(report_json_path))
        started_at = time()
        print(
            f"[prepare] sampleId={resolved_sample_id} requireCached={require_cached} bundleMode={bundle_mode} "
            f"embeddingType={embedding_type} vis={vis_method}/{vis_id}",
            flush=True,
        )

        try:
            _set_prepare_status(resolved_sample_id, "checking_cache", "Checking EIF local bundle cache", active=True)
            payload_path = _resolve_cache_payload_path(resolved_sample_id, explicit_cache_path)
            cache_hit = False

            if payload_path.exists():
                cached_payload = json.loads(payload_path.read_text(encoding="utf-8"))
                if _payload_matches_request(cached_payload, bundle_mode, embedding_type, model_path):
                    payload = cached_payload
                    cache_hit = True
                    _set_prepare_status(resolved_sample_id, "cache_hit", "Matching EIF local bundle cache found", active=True)
                    print(f"[prepare] sampleId={resolved_sample_id} cache=hit", flush=True)
                else:
                    payload = cached_payload
                    _set_prepare_status(resolved_sample_id, "cache_miss", "Existing cache does not match the requested real bundle", active=True)
                    print(f"[prepare] sampleId={resolved_sample_id} cache=mismatch", flush=True)
            if not cache_hit:
                if require_cached:
                    _set_prepare_status(resolved_sample_id, "error", "EIF local bundle cache not found. Prepare sample first.", active=False, error=True)
                    self._send_json(404, {
                        "status": "error",
                        "message": f"EIF local bundle cache not found for sample: {resolved_sample_id}. Please prepare the sample first.",
                    })
                    return
                _set_prepare_status(resolved_sample_id, "building_bundle", "Building TTAV bundle from EIF report", active=True)
                payload = _build_requested_payload(
                    report_json_path=report_json_path,
                    test_data=test_data,
                    model_path=model_path,
                    sample_id=resolved_sample_id,
                    bundle_mode=bundle_mode,
                    embedding_type=embedding_type,
                    hidden_layer=hidden_layer,
                    vis_method=vis_method,
                    vis_id=vis_id,
                    progress_callback=lambda stage, message: _set_prepare_status(resolved_sample_id, stage, message, active=True),
                )
                num_points = len(payload.get("bundle", {}).get("labels", []))
                print(f"[prepare] sampleId={resolved_sample_id} bundle_points={num_points}", flush=True)

            payload["vis_method"] = vis_method
            payload["vis_id"] = vis_id
            payload["overwrite"] = overwrite_remote
            payload["build_trainable_session"] = True
            payload["wait_until_ready"] = False
            payload["data_type"] = "Text"
            payload["task_type"] = "Alignment"
            payload["vis_config"] = {
                "gpu_id": -1,
                "n_neighbors": 10,
                "max_epochs": 10,
                "patient": 3,
                "s_n_epochs": 500,
                "b_n_epochs": 0,
                "t_n_epochs": 5,
                "lambda": 1.0,
                "refine_hd_k": 15,
            }
            _set_prepare_status(resolved_sample_id, "writing_local_cache", "Writing EIF local bundle cache", active=True)
            write_local_bundle_cache(resolved_sample_id, payload, explicit_path=explicit_cache_path)
            real_bundle_dir = REPO_ROOT / "ttav_bundles_real" / resolved_sample_id
            write_local_bundle_cache(resolved_sample_id, payload, explicit_path=str(real_bundle_dir))
            _set_prepare_status(resolved_sample_id, "uploading_to_ttav", "Uploading bundle to TTAV", active=True)
            upload_result = upload_bundle(ttav_upload_url, payload)
            elapsed = time() - started_at
            ttav_cached = upload_result.get("cached") is True
            print(f"[prepare] sampleId={resolved_sample_id} ttav_cached={ttav_cached} elapsed={elapsed:.2f}s", flush=True)
            _set_prepare_status(resolved_sample_id, "completed", f"Prepare sample completed in {elapsed:.1f}s", active=False)
        except Exception as exc:
            elapsed = time() - started_at
            print(f"[prepare] sampleId={resolved_sample_id} error after {elapsed:.2f}s: {exc}", flush=True)
            _set_prepare_status(resolved_sample_id, "error", str(exc), active=False, error=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return

        content_path = upload_result.get("content_path")
        response = {
            "status": "success",
            "sampleId": upload_result.get("sample_id") or payload.get("sample_id"),
            "contentPath": content_path,
            "visMethod": upload_result.get("vis_method", vis_method),
            "visId": upload_result.get("vis_id", vis_id),
            "ttavUrl": ttav_url,
            "selectedIndices": selected_indices,
            "targetIndex": target_index,
            "eifBundleCachePath": str(_cache_dir(resolved_sample_id, explicit_cache_path)),
            "eifCacheHit": cache_hit,
            "bundleMode": bundle_mode,
            "embeddingType": embedding_type,
            "overwrite": overwrite_remote,
            "uploadResult": upload_result,
            "refineReady": upload_result.get("refineReady", False),
            "trainableSessionStatus": upload_result.get("trainableSessionStatus", "registered"),
            "statusMessage": upload_result.get("statusMessage"),
        }
        self._send_json(200, response)

    def _handle_prepare_train_probe(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        report_file_name = str(req.get("reportFileName", "")).strip()
        if not report_file_name:
            self._send_json(400, {"status": "error", "message": "reportFileName is required"})
            return

        report_json_path = CORR_RESULTS_DIR / report_file_name
        if not report_json_path.exists():
            self._send_json(404, {"status": "error", "message": f"Report JSON not found: {report_file_name}"})
            return

        train_sample_id = req.get("trainSampleId")
        probe_pairs = req.get("probePairs")
        if not isinstance(train_sample_id, int):
            self._send_json(400, {"status": "error", "message": "trainSampleId is required"})
            return
        if not isinstance(probe_pairs, list) or not probe_pairs:
            self._send_json(400, {"status": "error", "message": "probePairs is required"})
            return

        base_sample_id = str(req.get("sampleId", "")).strip() or infer_sample_id(str(report_json_path))
        context_radius = int(req.get("contextRadius", 1))
        include_full_train = bool(req.get("includeFullTrain", False))
        raw_focus_train_indices = req.get("focusTrainIndices", [])
        focus_train_indices = raw_focus_train_indices if isinstance(raw_focus_train_indices, list) else []
        model_path = str(req.get("modelPath", "")).strip() or str(DEFAULT_MODEL_PATH)
        ttav_upload_url = str(req.get("ttavUploadUrl", "")).strip()
        ttav_url = str(req.get("ttavUrl", "")).strip()
        vis_method = str(req.get("visMethod", "TimeVis")).strip() or "TimeVis"
        vis_id = str(req.get("visId", "1")).strip() or "1"
        probe_sample_id = _make_probe_sample_id(
            base_sample_id,
            train_sample_id,
            probe_pairs,
            context_radius,
            include_full_train,
            focus_train_indices,
        )
        explicit_cache_path = str(req.get("probeCachePath", "")).strip() or _default_probe_cache_path(base_sample_id, train_sample_id, probe_sample_id)
        status_key = probe_sample_id
        started_at = time()

        print(
            f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} pairs={len(probe_pairs)} vis={vis_method}/{vis_id}",
            flush=True,
        )

        try:
            _set_prepare_status(status_key, "building_probe", "Building train-sample embedding probe", active=True)
            payload = build_train_probe_bundle_payload(
                report_json_path=str(report_json_path),
                model_path=model_path,
                train_sample_id=train_sample_id,
                probe_pairs=probe_pairs,
                sample_id=base_sample_id,
                embedding_type="contextual",
                hidden_layer=-1,
                vis_method=vis_method,
                vis_id=vis_id,
                context_radius=context_radius,
                include_full_train=include_full_train,
                focus_train_indices=focus_train_indices,
                progress_callback=lambda stage, message: _set_prepare_status(status_key, stage, message, active=True),
            )
            payload["sample_id"] = probe_sample_id
            payload["vis_method"] = vis_method
            payload["vis_id"] = vis_id
            payload["overwrite"] = True
            _set_prepare_status(status_key, "writing_local_cache", "Writing train probe bundle cache", active=True)
            write_local_bundle_cache(probe_sample_id, payload, explicit_path=explicit_cache_path)

            upload_result = None
            upload_error = None
            browser_upload_required = False
            _set_prepare_status(status_key, "uploading_to_ttav", "Uploading train probe to TTAV", active=True)
            try:
                upload_result = upload_bundle(ttav_upload_url, payload)
            except Exception as upload_exc:
                upload_error = str(upload_exc)
                browser_upload_required = True
                print(
                    f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} server_upload_failed: {upload_error}",
                    flush=True,
                )
                _set_prepare_status(
                    status_key,
                    "browser_upload_required",
                    "Server-side TTAV upload failed; browser upload fallback required",
                    active=False,
                )

            elapsed = time() - started_at
            print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} elapsed={elapsed:.2f}s", flush=True)
            if not browser_upload_required:
                _set_prepare_status(status_key, "completed", f"Train probe completed in {elapsed:.1f}s", active=False)
        except Exception as exc:
            elapsed = time() - started_at
            print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} error after {elapsed:.2f}s: {exc}", flush=True)
            _set_prepare_status(status_key, "error", str(exc), active=False, error=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return

        self._send_json(200, {
            "status": "success",
            "sampleId": (upload_result or {}).get("sample_id") or probe_sample_id,
            "contentPath": (upload_result or {}).get("content_path"),
            "visMethod": (upload_result or {}).get("vis_method", vis_method),
            "visId": (upload_result or {}).get("vis_id", vis_id),
            "ttavUrl": ttav_url,
            "trainSampleId": train_sample_id,
            "selectedIndices": payload.get("selected_indices", []),
            "targetIndex": payload.get("target_index"),
            "promptLen": payload.get("bundle", {}).get("prompt_len", 0),
            "probeCachePath": explicit_cache_path,
            "comparisonSummary": payload.get("comparison_summary"),
            "uploadResult": upload_result,
            "browserUploadRequired": browser_upload_required,
            "uploadError": upload_error,
            "bundlePayload": payload if browser_upload_required else None,
        })


def main():
    parser = argparse.ArgumentParser(description="EIF API for preparing and uploading TTAV bundles.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), TTAVBundleRequestHandler)
    print(f"EIF TTAV bundle API listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
