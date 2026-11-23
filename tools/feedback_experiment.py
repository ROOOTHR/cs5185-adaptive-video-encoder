"""Size-only comparison of the feedback correction rules (see docs/feedback_experiment.md).

    python tools/feedback_experiment.py run   [--workers 8]
    python tools/feedback_experiment.py report

Only the correction step is under test, so a task is just: encode two-pass at the requested rate, read the
file size, ask `pipeline.next_rate` for the next rate. No quality metrics are computed. DEV and VAL videos
form the comparison; one TEST cell (10_Stockholm 5% A) is a labelled diagnostic that never feeds a decision.
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

from src import config, decide, encode, experiment, pipeline  # noqa: E402
from src import metadata as meta_mod  # noqa: E402

OUT = config.RUN_DIR / "feedback_exp"
POLICIES = pipeline.POLICIES
REPEATS = (1, 2, 3)
DIAG_REPEATS = (1, 2, 3, 4, 5)
MAIN_IDS = config.DEV_IDS + config.VAL_IDS
GROUPS = ("B0", "B1", "A")
DIAG = ("10_Stockholm", 0.05, "A")
WINDOW = (-0.03, 0.0)


def _features(video_id: str) -> dict:
    path = config.RUN_DIR / "final_v2" / "features" / f"{video_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _loop(src: Path, video_dir: Path, name: str, params: dict, bud0: dict, policy: str) -> dict:
    """Same control flow as pipeline.run_candidate, minus every measurement except file size."""
    kbps, history, trace = int(bud0["video_kbps"]), [], []
    target = bud0["target_bytes"]
    for number in range(1, config.MAX_ATTEMPTS + 1):
        out = video_dir / f"{name}_a{number}.mp4"
        plan = {"params": dict(params), "motion": "n/a", "detail": "n/a"}
        encoded = encode.two_pass(src, out, plan, dict(bud0, video_kbps=kbps), str(video_dir / f"{name}_a{number}"))
        if any(item["returncode"] != 0 for item in encoded["passes"]) or len(encoded["passes"]) != 2 or not out.is_file():
            return dict(status="encode_failed", attempts=number, size_error=None, trace=trace)
        size = out.stat().st_size
        error = (size - target) / target
        trace.append([kbps, size, round(error, 5)])
        if WINDOW[0] <= error <= WINDOW[1]:
            return dict(status="ok", attempts=number, size_error=error, trace=trace)
        history.append((kbps, size))
        low, high = kbps * (1 - config.MAX_ADJUST), kbps * (1 + config.MAX_ADJUST)
        kbps = max(50, int(round(min(max(pipeline.next_rate(policy, history, target), low), high))))
    return dict(status="exhausted", attempts=config.MAX_ATTEMPTS, size_error=error, trace=trace)


def _job(spec: dict) -> list[dict]:
    policy, rep, vid, diag = spec["policy"], spec["rep"], spec["video_id"], spec["diag"]
    video_dir = OUT / "videos" / f"{'diag_' if diag else ''}{policy}_r{rep}" / vid
    video_dir.mkdir(parents=True, exist_ok=True)
    meta, feat, rules = meta_mod.load_meta(vid), _features(vid), experiment._load_rules()
    rows = []
    for ratio in config.RATIOS:
        for group in GROUPS:
            if diag and (vid, ratio, group) != DIAG:
                continue
            bud = meta_mod.budget(meta, ratio)
            bpp = bud["video_kbps"] * 1000 / (meta["width"] * meta["height"] * meta["fps"])
            plan = decide.decide(feat, bud, group, rules, bpp)
            result = _loop(config.RAW_DIR / meta["file"], video_dir, f"{group}_{int(ratio * 100):02d}pct",
                           plan["params"], bud, policy)
            rows.append(dict(policy=policy, rep=rep, diagnostic=diag, video_id=vid, ratio=ratio, group=group,
                             target_bytes=bud["target_bytes"], **result))
    return rows


def run(workers: int) -> None:
    config.ensure_dirs()
    jobs = [dict(policy=p, rep=r, video_id=v, diag=False) for r in REPEATS for p in POLICIES for v in MAIN_IDS]
    jobs += [dict(policy=p, rep=r, video_id=DIAG[0], diag=True) for r in DIAG_REPEATS for p in POLICIES]
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, spec): spec for spec in jobs}
        for future in as_completed(futures):
            spec = futures[future]
            rows += future.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {spec['policy']} r{spec['rep']} {spec['video_id']}"
                  f"{' (diagnostic)' if spec['diag'] else ''}", flush=True)
    rows.sort(key=lambda r: (r["diagnostic"], r["policy"], r["rep"], r["video_id"], r["ratio"], r["group"]))
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["trace"] = json.dumps(r["trace"])
    keep = ("policy", "rep", "diagnostic", "video_id", "ratio", "group", "status", "attempts", "target_bytes",
            "size_error", "trace")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keep)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT / 'summary.csv'}")


def _read() -> list[dict]:
    with (OUT / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _err(row: dict):
    return None if row["size_error"] in ("", "None") else float(row["size_error"])


def report() -> None:
    rows = _read()
    main = [r for r in rows if r["diagnostic"] == "False"]
    diag = [r for r in rows if r["diagnostic"] == "True"]

    def stats(sub):
        ok = [r for r in sub if r["status"] == "ok"]
        att = [int(r["attempts"]) for r in sub]
        err = [_err(r) for r in ok]
        return len(sub), len(ok), sum(att) / len(sub), sum(a >= 3 for a in att), (sum(err) / len(err) if err else float("nan"))

    print(f"## Main comparison: DEV + VAL, {len(MAIN_IDS) * 2 * len(GROUPS)} tasks x {len(REPEATS)} repeats per policy")
    print("| policy | runs | accepted | rate | mean attempts | >=3 attempts | mean size_error (accepted) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for policy in POLICIES:
        n, ok, att, slow, err = stats([r for r in main if r["policy"] == policy])
        print(f"| {policy} | {n} | {ok} | {ok / n:.1%} | {att:.2f} | {slow} | {err:+.2%} |")
    print("\n## Per repeat (run-to-run variability)")
    print("| policy | " + " | ".join(f"r{r} accepted / mean attempts" for r in REPEATS) + " |")
    print("|---|" + "---:|" * len(REPEATS))
    for policy in POLICIES:
        cells = []
        for rep in REPEATS:
            n, ok, att, _, _ = stats([r for r in main if r["policy"] == policy and int(r["rep"]) == rep])
            cells.append(f"{ok} / {n}, {att:.2f}")
        print(f"| {policy} | " + " | ".join(cells) + " |")
    print("\n## Per group (accepted / runs)")
    print("| policy | " + " | ".join(GROUPS) + " |")
    print("|---|" + "---:|" * len(GROUPS))
    for policy in POLICIES:
        cells = []
        for g in GROUPS:
            sub = [r for r in main if r["policy"] == policy and r["group"] == g]
            cells.append(f"{sum(r['status'] == 'ok' for r in sub)} / {len(sub)}")
        print(f"| {policy} | " + " | ".join(cells) + " |")
    print("\n## Per ratio (accepted / runs)")
    print("| policy | 5% | 2% |")
    print("|---|---:|---:|")
    for policy in POLICIES:
        cells = []
        for ratio in ("0.05", "0.02"):
            sub = [r for r in main if r["policy"] == policy and float(r["ratio"]) == float(ratio)]
            cells.append(f"{sum(r['status'] == 'ok' for r in sub)} / {len(sub)}")
        print(f"| {policy} | " + " | ".join(cells) + " |")

    print("\n## Tasks that fail under at least one policy (accepted count out of repeats)")
    print("| video | ratio | group | " + " | ".join(POLICIES) + " |")
    print("|---|---:|---|" + "---:|" * len(POLICIES))
    keys = sorted({(r["video_id"], r["ratio"], r["group"]) for r in main})
    for key in keys:
        cells = []
        for policy in POLICIES:
            sub = [r for r in main if r["policy"] == policy and (r["video_id"], r["ratio"], r["group"]) == key]
            cells.append(sum(r["status"] == "ok" for r in sub))
        if min(cells) < len(REPEATS):
            print(f"| {key[0]} | {float(key[1]):.0%} | {key[2]} | " + " | ".join(f"{c}/{len(REPEATS)}" for c in cells) + " |")

    print("\n## Diagnostic cell: 10_Stockholm 5% A (TEST video, not used for any decision)")
    print("| policy | accepted | mean attempts | traces (requested kbps -> size_error) |")
    print("|---|---:|---:|---|")
    for policy in POLICIES:
        sub = sorted([r for r in diag if r["policy"] == policy], key=lambda r: int(r["rep"]))
        traces = "; ".join(" > ".join(f"{k}:{e:+.1%}" for k, _, e in json.loads(r["trace"])) for r in sub)
        print(f"| {policy} | {sum(r['status'] == 'ok' for r in sub)} / {len(sub)} | "
              f"{sum(int(r['attempts']) for r in sub) / len(sub):.2f} | {traces} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = parser.parse_args()
    run(args.workers) if args.cmd == "run" else report()
