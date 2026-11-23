"""Run the pre-registered feedback-policy comparison (see docs/feedback_experiment.md).

    python tools/feedback_experiment.py run   [--workers 8]
    python tools/feedback_experiment.py report

Only DEV and VAL videos are used for the comparison. One TEST cell (10_Stockholm 5% A) is run as a
labelled diagnostic and never feeds a decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, decide, experiment, pipeline  # noqa: E402
from src import metadata as meta_mod  # noqa: E402

OUT = config.RUN_DIR / "feedback_exp"
POLICIES = pipeline.POLICIES
REPEATS = (1, 2)
DIAG_REPEATS = (1, 2, 3)
MAIN_IDS = config.DEV_IDS + config.VAL_IDS
GROUPS = ("B0", "B1", "A")
DIAG = ("10_Stockholm", 0.05, "A")


def _features(video_id: str) -> dict:
    path = config.RUN_DIR / "final_v2" / "features" / f"{video_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _job(spec: dict) -> list[dict]:
    policy, rep, vid, diag = spec["policy"], spec["rep"], spec["video_id"], spec["diag"]
    run_dir = OUT / (f"diag_{policy}_r{rep}_{vid}" if diag else f"{policy}_r{rep}_{vid}")
    (run_dir / "perframe").mkdir(parents=True, exist_ok=True)
    meta, feat, rules = meta_mod.load_meta(vid), _features(vid), experiment._load_rules()
    rows = []
    for ratio in config.RATIOS:
        for group in GROUPS:
            if diag and (vid, ratio, group) != DIAG:
                continue
            bud = meta_mod.budget(meta, ratio)
            bpp = bud["video_kbps"] * 1000 / (meta["width"] * meta["height"] * meta["fps"])
            plan = decide.decide(feat, bud, group, rules, bpp)
            result = pipeline.run_candidate(vid, ratio, group, plan["params"], feat, meta, run_dir,
                                            reasons=plan["reasons"], policy=policy)
            result.update(policy=policy, rep=rep, video_id=vid, ratio=ratio, group=group, diagnostic=diag)
            rows.append(result)
    return rows


def run(workers: int) -> None:
    config.ensure_dirs()
    jobs = [dict(policy=p, rep=r, video_id=v, diag=False) for r in REPEATS for p in POLICIES for v in MAIN_IDS]
    jobs += [dict(policy=p, rep=r, video_id=DIAG[0], diag=True) for r in DIAG_REPEATS for p in POLICIES]
    done, rows = 0, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, spec): spec for spec in jobs}
        for future in as_completed(futures):
            spec = futures[future]
            rows += future.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {spec['policy']} r{spec['rep']} {spec['video_id']}"
                  f"{' (diagnostic)' if spec['diag'] else ''}", flush=True)
    keep = ("policy", "rep", "diagnostic", "video_id", "ratio", "group", "status", "attempts", "selected_attempt",
            "target_bytes", "actual_bytes", "size_error", "psnr_y_db", "ms_ssim_db", "failure_reason")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keep, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["diagnostic"], r["policy"], r["rep"], r["video_id"],
                                                     r["ratio"], r["group"])))
    print(f"wrote {OUT / 'summary.csv'}")


def _read() -> list[dict]:
    with (OUT / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _num(row: dict, key: str):
    try:
        return float(row[key]) if row[key] not in ("", "None") else None
    except (KeyError, ValueError):
        return None


def report() -> None:
    rows = _read()
    main = [r for r in rows if r["diagnostic"] == "False"]
    diag = [r for r in rows if r["diagnostic"] == "True"]

    def stats(sub):
        n = len(sub)
        ok = [r for r in sub if r["status"] == "ok"]
        att = [int(r["attempts"]) for r in sub]
        err = [_num(r, "size_error") for r in ok if _num(r, "size_error") is not None]
        return n, len(ok), sum(att) / n, sum(a >= 3 for a in att), (sum(err) / len(err) if err else float("nan"))

    print("## Main comparison: DEV + VAL, 48 tasks x 2 repeats per policy")
    print("| policy | runs | accepted | rate | mean attempts | >=3 attempts | mean size_error (accepted) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for policy in POLICIES:
        n, ok, att, slow, err = stats([r for r in main if r["policy"] == policy])
        print(f"| {policy} | {n} | {ok} | {ok / n:.1%} | {att:.2f} | {slow} | {err:+.2%} |")
    print("\n## Per repeat (shows run-to-run variability)")
    print("| policy | rep | accepted / 48 | mean attempts |")
    print("|---|---:|---:|---:|")
    for policy in POLICIES:
        for rep in REPEATS:
            n, ok, att, _, _ = stats([r for r in main if r["policy"] == policy and int(r["rep"]) == rep])
            print(f"| {policy} | {rep} | {ok} / {n} | {att:.2f} |")
    print("\n## Per group (accepted / runs)")
    print("| policy | " + " | ".join(GROUPS) + " |")
    print("|---|" + "---:|" * len(GROUPS))
    for policy in POLICIES:
        cells = []
        for g in GROUPS:
            sub = [r for r in main if r["policy"] == policy and r["group"] == g]
            cells.append(f"{sum(r['status'] == 'ok' for r in sub)} / {len(sub)}")
        print(f"| {policy} | " + " | ".join(cells) + " |")

    print("\n## Quality on tasks accepted by both (mean delta vs proportional)")
    by = {(r["policy"], int(r["rep"]), r["video_id"], r["ratio"], r["group"]): r for r in main}
    print("| policy | paired tasks | dMS-SSIM(dB) | dPSNR-Y(dB) |")
    print("|---|---:|---:|---:|")
    for policy in ("center", "secant"):
        dm, dp = [], []
        for key, row in by.items():
            if key[0] != policy or row["status"] != "ok":
                continue
            base = by.get(("proportional",) + key[1:])
            if base and base["status"] == "ok":
                dm.append(_num(row, "ms_ssim_db") - _num(base, "ms_ssim_db"))
                dp.append(_num(row, "psnr_y_db") - _num(base, "psnr_y_db"))
        if dm:
            print(f"| {policy} | {len(dm)} | {sum(dm) / len(dm):+.3f} | {sum(dp) / len(dp):+.3f} |")

    print("\n## Diagnostic cell: 10_Stockholm 5% A (TEST video, not used for any decision)")
    print("| policy | rep | status | attempts | size_error |")
    print("|---|---:|---|---:|---:|")
    for r in sorted(diag, key=lambda r: (r["policy"], int(r["rep"]))):
        err = _num(r, "size_error")
        print(f"| {r['policy']} | {r['rep']} | {r['status']} | {r['attempts']} | "
              f"{'' if err is None else format(err, '+.2%')} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = parser.parse_args()
    run(args.workers) if args.cmd == "run" else report()
