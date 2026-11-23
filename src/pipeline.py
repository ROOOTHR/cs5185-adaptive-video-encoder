"""Per-candidate feedback loop; keeps every attempted output in the run."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from . import config, encode, evaluate
from .metadata import budget


POLICIES = ("proportional", "center", "secant")
AIM_FRACTION = 0.985          # middle of the -3%..0 acceptance window
SLOPE_BOUNDS = (0.5, 2.0)     # secant slope must be within these multiples of actual/requested


def next_rate(policy: str, history: list[tuple[int, int]], target_bytes: int) -> float:
    """Next requested kbps from [(requested_kbps, actual_bytes), ...], oldest first (before clamping)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown feedback policy: {policy}")
    kbps, actual = history[-1]
    actual = max(actual, 1)
    if policy == "proportional":
        margin = config.OVERHEAD if actual > target_bytes else 1.0
        return kbps * target_bytes / actual * margin
    aim = target_bytes * AIM_FRACTION
    rate = kbps * aim / actual
    if policy == "secant" and len(history) >= 2:
        (k0, b0), (k1, b1) = history[-2], history[-1]
        if k1 != k0:
            slope, proportional_slope = (b1 - b0) / (k1 - k0), b1 / k1
            if SLOPE_BOUNDS[0] * proportional_slope <= slope <= SLOPE_BOUNDS[1] * proportional_slope:
                rate = k1 + (aim - b1) / slope
    return rate


def pick_best(attempts: list[dict]) -> dict | None:
    eligible = [row for row in attempts if row.get("status") == "ok"]
    return max(eligible, key=lambda row: row["measurements"]["actual_bytes"], default=None)


def run_candidate(video_id: str, ratio: float, group: str, params: dict,
                  feature: dict, meta: dict, run_dir: Path,
                  initial_kbps: int | None = None, reasons=(), policy: str = "proportional") -> dict:
    started = time.perf_counter()
    bud0 = budget(meta, ratio)
    kbps = int(initial_kbps or bud0["video_kbps"])
    video_dir = run_dir / "videos" / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    attempts, measurements, history = [], {}, []
    for number in range(1, config.MAX_ATTEMPTS + 1):
        name = f"{group}_{int(ratio * 100):02d}pct_a{number}"
        out = video_dir / f"{name}.mp4"
        job = str(video_dir / name)
        plan = {"params": dict(params), "motion": "n/a", "detail": "n/a"}
        encode_started = time.perf_counter()
        encoded = encode.two_pass(config.RAW_DIR / meta["file"], out, plan,
                                  dict(bud0, video_kbps=kbps), job)
        encode_time = time.perf_counter() - encode_started
        codes = [item["returncode"] for item in encoded["passes"]]
        record = {"job_id": name, "attempt_id": number, "video_id": video_id,
                  "group": group, "target_ratio": ratio, "target_bytes": bud0["target_bytes"],
                  "planned_kbps": kbps, "features": feature["stats"],
                  "parameters": dict(params), "reasons": list(reasons),
                  "ffmpeg_commands": [item["cmd"] for item in encoded["passes"]],
                  "returncodes": codes, "encode_time_s": encode_time, "feedback_policy": policy}
        if len(codes) != 2 or any(code != 0 for code in codes) or not out.is_file():
            tail = encoded["passes"][-1]["stderr_tail"] if encoded["passes"] else "no FFmpeg pass completed"
            record.update(status="failed", failure_reason=f"encode failed: {tail}", next_action="stop")
        else:
            eval_started = time.perf_counter()
            measured = evaluate.evaluate(out, config.RAW_DIR / meta["file"], meta, bud0["target_bytes"])
            eval_time = time.perf_counter() - eval_started
            record["eval_time_s"] = eval_time
            measurements[number] = measured
            record["measurements"] = {key: measured.get(key) for key in (
                "actual_bytes", "size_error", "duration_s", "frame_count", "decode_ok", "file_kbps",
                "psnr_y_db", "ms_ssim", "ms_ssim_db", "ms_ssim_min", "actual_video_kbps")}
            record.update(status="ok" if measured["valid"] else "rejected",
                           failure_reason=measured["failure_reason"] or None,
                           output_path=str(out))
            if measured["valid"]:
                record["next_action"] = "accept"
            elif any(not (e.startswith("actual_bytes exceeds") or
                          e.startswith("actual_bytes is more than 3% below"))
                     for e in measured["integrity_errors"]):
                record["next_action"] = "stop_integrity_failure"
            elif number == config.MAX_ATTEMPTS:
                record["next_action"] = "exhausted"
            else:
                history.append((kbps, measured["actual_bytes"]))
                raw_rate = next_rate(policy, history, bud0["target_bytes"])
                low, high = kbps * (1 - config.MAX_ADJUST), kbps * (1 + config.MAX_ADJUST)
                record.update(next_action="retry",
                              next_kbps=max(50, int(round(min(max(raw_rate, low), high)))),
                              reason=f"size_error={measured['size_error']:+.4%}")
        if out.is_file():
            digest = hashlib.sha256()
            with out.open("rb") as artifact:
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
            record["output_sha256"] = digest.hexdigest()
        with attempts_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
        attempts.append(record)
        if record["next_action"] != "retry":
            break
        kbps = record["next_kbps"]

    best = pick_best(attempts)
    selected = measurements.get(best["attempt_id"]) if best else None
    if selected:
        frame_data = {key: selected[key] for key in ("ms_ssim_per_frame", "psnr_y_per_frame")}
        frame_data.update(video_id=video_id, group=group, ratio=ratio,
                          frame_count=len(frame_data["ms_ssim_per_frame"]),
                          selected_attempt=best["attempt_id"])
        (run_dir / "perframe" / f"{video_id}_{group}_{int(ratio*100):02d}pct.json").write_text(
            json.dumps(frame_data, allow_nan=True), encoding="utf-8")
    summary = {"video_id": video_id, "target_ratio": ratio, "group": group,
               "status": "ok" if best else "failed",
               "failure_reason": None if best else "; ".join(
                   row.get("failure_reason") or row.get("next_action", "") for row in attempts),
               "attempts": len(attempts), "selected_attempt": best["attempt_id"] if best else None,
               "total_time_s": time.perf_counter() - started,
               "encode_time_s": sum(row["encode_time_s"] for row in attempts),
               "eval_time_s": sum(row.get("eval_time_s", 0) for row in attempts),
               "agent_time_s": 0,
               "target_bytes": bud0["target_bytes"], "target_kbps": bud0["video_kbps"],
               "bpp": bud0["video_kbps"] * 1000 / (meta["width"] * meta["height"] * meta["fps"]),
               "parameters": json.dumps(params, sort_keys=True), "reasons": "; ".join(reasons)}
    if best:
        summary.update({key: best["measurements"].get(key) for key in (
            "actual_bytes", "size_error", "duration_s", "frame_count", "decode_ok", "file_kbps",
            "psnr_y_db", "ms_ssim", "ms_ssim_db", "ms_ssim_min", "actual_video_kbps")})
        summary["output_path"] = best["output_path"]
        digest = hashlib.sha256(Path(best["output_path"]).read_bytes()).hexdigest()
        summary["output_sha256"] = digest
    elif attempts and attempts[-1].get("measurements"):
        summary.update(attempts[-1]["measurements"])
        summary["output_path"] = attempts[-1].get("output_path")
    return summary
