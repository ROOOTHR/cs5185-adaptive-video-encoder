"""五步闭环（项目第 1-5 步的编排）。

对应附录 A 的控制流：分析 -> 决策 -> 编码 -> 检查 -> 调整。
每次尝试的候选都保留，最后按"不超目标且最接近"挑一个，不会因为后续失败丢掉前一轮的可用结果。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config, decide, encode, evaluate, features
from . import metadata as meta_mod

TOL = 0.03          # 体积误差容忍 3%
MAX_ADJUST = 0.30   # 单轮码率调整不超过 ±30%，防止震荡


def _fmt_ratio(r: float) -> str:
    return f"{int(round(r * 100)):02d}pct"


def pick_best(attempts: list[dict]) -> dict | None:
    """优先选不超目标且最接近的；全都超则选最小的。"""
    done = [a for a in attempts if a.get("status") == "ok"]
    if not done:
        return None
    under = [a for a in done if a["actual_bytes"] <= a["target_bytes"]]
    pool = under or done
    return min(pool, key=lambda a: abs(a["actual_bytes"] - a["target_bytes"]))


def run_task(video_id: str, ratio: float, verbose: bool = True,
             mode: str = "adaptive") -> dict:
    """mode='adaptive' 走规则表；mode='fixed' 是统一基线对照组。"""
    t0 = time.perf_counter()
    config.ensure_dirs()

    src = config.RAW_DIR / f"{video_id}.mp4"
    meta = meta_mod.load_meta(video_id)
    feat = features.load(video_id)
    bud0 = meta_mod.budget(meta, ratio)
    plan = decide.decide(feat, bud0, mode=mode)

    out_root = config.OUT_DIR if mode == "adaptive" else config.BASELINE_DIR
    tag = f"{video_id}_{_fmt_ratio(ratio)}" + ("" if mode == "adaptive" else "_base")
    log_path = config.LOG_DIR / f"{tag}.jsonl"
    attempts: list[dict] = []
    kbps = bud0["video_kbps"]
    prev_bytes: int | None = None

    for n in range(1, config.MAX_ATTEMPTS + 1):
        out = out_root / video_id / f"{_fmt_ratio(ratio)}_a{n}.mp4"
        job = f"{tag}_a{n}"
        bud = dict(bud0, video_kbps=kbps, video_bps=kbps * 1000)

        te = time.perf_counter()
        enc = encode.two_pass(src, out, plan, bud, job)
        enc_time = round(time.perf_counter() - te, 2)

        rec = {
            "job_id": job,
            "attempt_id": n,
            "video_id": video_id,
            "target_ratio": ratio,
            "target_bytes": bud0["target_bytes"],
            "planned_kbps": kbps,
            "motion": plan["motion"],
            "detail": plan["detail"],
            "features": {
                "mafd": feat["stats"]["mafd"]["mean"],
                "edge_ratio": feat["stats"]["edge_ratio"]["mean"],
                "scene_cuts": len(feat.get("scene_cuts", [])),
            },
            "parameters": plan["params"],
            "encode_time_s": enc_time,
            "ffmpeg_cmd": enc["passes"][-1]["cmd"] if enc["passes"] else [],
            "returncode": enc["passes"][-1]["returncode"] if enc["passes"] else -1,
        }

        failed = rec["returncode"] != 0 or not out.exists()
        if failed:
            rec.update(status="encode_failed",
                       failure_reason=(enc["passes"][-1]["stderr_tail"] if enc["passes"] else "no run"))
            attempts.append(rec)
        else:
            ev = evaluate.evaluate(out, src, meta["bytes"], bud0["target_bytes"])
            rec.update(status="ok", **ev)
            err = ev["size_error"] or 0

            # 撞到内容上限：已经低于目标，但码率再加也不再涨（06_ShakeNDry 就是这样，
            # 内容太容易编码，x264 到最低 QP 后无法再花掉比特）。继续调只是白跑。
            capped = (prev_bytes is not None
                      and ev["actual_bytes"] < bud0["target_bytes"] * 0.95
                      and ev["actual_bytes"] <= prev_bytes * 1.03)
            prev_bytes = ev["actual_bytes"]

            if capped:
                rec["next_action"] = "capped"
                rec["reason"] = "码率提高后体积几乎不变，判定已达内容上限"
            elif abs(err) <= TOL:
                rec["next_action"] = "accept"
            elif n < config.MAX_ATTEMPTS:
                rec["next_action"] = "adjust"
                raw = kbps * bud0["target_bytes"] / ev["actual_bytes"] * config.OVERHEAD
                lo, hi = kbps * (1 - MAX_ADJUST), kbps * (1 + MAX_ADJUST)
                nxt = max(50, int(round(min(max(raw, lo), hi))))
                rec["next_kbps"] = nxt
                rec["reason"] = (f"实际 {ev['actual_MB']}MB / 目标 {ev['target_MB']}MB，"
                                 f"误差 {err:+.1%}，码率 {kbps} -> {nxt} kbps")
            else:
                rec["next_action"] = "exhausted"
            attempts.append(rec)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if verbose:
            st = rec["status"]
            if st == "ok":
                print(f"  [{video_id} {ratio:.0%}] 轮{n} {kbps}k -> "
                      f"{rec['actual_MB']:.2f}MB (目标 {rec['target_MB']:.2f}) "
                      f"err={rec['size_error']:+.1%} PSNR={rec.get('psnr_y_db')} "
                      f"{rec['next_action']} {enc_time}s")
            else:
                print(f"  [{video_id} {ratio:.0%}] 轮{n} 编码失败 rc={rec['returncode']}")

        stop = (failed or n >= config.MAX_ATTEMPTS
                or rec.get("next_action") in ("accept", "capped", "exhausted"))
        if stop:
            break
        kbps = rec["next_kbps"]

    best = pick_best(attempts)
    total = round(time.perf_counter() - t0, 2)

    summary = {
        "video_id": video_id,
        "target_ratio": ratio,
        "status": "ok" if best else "failed",
        "attempts": len(attempts),
        "total_time_s": total,
        "mode": mode,
        "motion": plan["motion"],
        "detail": plan["detail"],
        "params": plan["params"],
        "target_bytes": bud0["target_bytes"],
        "selected_attempt": None,
    }
    if best:
        summary.update({
            "selected_attempt": best["attempt_id"],
            "actual_bytes": best["actual_bytes"],
            "actual_MB": best["actual_MB"],
            "target_MB": best["target_MB"],
            "size_error": best["size_error"],
            "ratio_achieved": best["ratio_achieved"],
            "decode_ok": best.get("decode_ok"),
            "psnr_y_db": best.get("psnr_y_db"),
            "duration_s": best.get("duration_s"),
            "frame_count": best.get("frame_count"),
            "video_kbps": best.get("video_kbps"),
            "output_path": best.get("output_path"),
        })
    return summary
