"""Reproducible v2 experiment phases with explicit split guards."""
from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config, decide, features, pipeline
from . import metadata as meta_mod
from .encode import _exe
from .ms_ssim import ms_ssim, sample_y_frames

# Pre-registered before scanning: choose by MS-SSIM(dB); a setting is meaningfully
# better only at +0.10 dB or more while losing no more than 0.30 dB PSNR-Y.
SIGNIFICANCE_MSSSIM_DB = 0.10
SIGNIFICANCE_PSNR_DB = -0.30
PRE_REGISTERED_FEATURES = ("edge_ratio", "mafd", "luma_std", "bpp")
MAX_RULES = 3
ANCHOR = dict(decide.ANCHOR)


def require_development_ids(ids):
    ids = list(ids)
    forbidden = sorted(set(ids) - set(config.DEV_IDS))
    if forbidden:
        raise ValueError(f"sweep/derive accepts DEV IDs only; rejected: {', '.join(forbidden)} (TEST is frozen)")
    if not ids:
        raise ValueError("at least one DEV video is required")
    return ids


def _run(command: list[str], check=True) -> str:
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n{result.stderr[-1000:]}")
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, check=True) -> str:
    return _run(["git", "-C", str(config.ROOT), *args], check=check)


def create_run(run_name: str) -> Path:
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("run name may contain only letters, digits, underscore, and hyphen")
    path = config.RUN_DIR / run_name
    if path.exists():
        raise FileExistsError(f"run directory already exists; use a new name: {path}")
    for part in ("features", "videos", "perframe", "figs", "frames", "clips"):
        (path / part).mkdir(parents=True, exist_ok=True)
    return path


def write_run_config(run_dir: Path, phase: str) -> dict:
    ffmpeg = _exe("ffmpeg")
    ffmpeg_version = _run([ffmpeg, "-version"]).splitlines()[0]
    status = _git("status", "--porcelain", "--untracked-files=normal").splitlines()
    sources = {}
    valid = True
    for vid in config.ALL_IDS:
        meta = meta_mod.load_meta(vid)
        video = config.RAW_DIR / meta["file"]
        metadata_path = config.META_DIR / f"{vid}.json"
        actual = _sha(video)
        expected = meta.get("sha256")
        match = not expected or actual.lower() == expected.lower()
        sources[vid] = {"video_sha256": actual, "metadata_sha256": _sha(metadata_path),
                        "expected_video_sha256": expected, "matches_metadata": match}
        valid &= match
    packages = {}
    for name in ("numpy", "opencv-python", "matplotlib", "imageio-ffmpeg",
                 "torch", "pytorch-msssim"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "phase": phase, "run_name": run_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(status),
        "git_status": status, "ffmpeg_version": ffmpeg_version,
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(), "encode_threads": config.ENCODE_THREADS,
        "parallelism": 1, "python": sys.version, "packages": packages,
        "decide_sha256": _sha(config.ROOT / "src" / "decide.py"),
        "dataset_root": str(config.RAW_DIR), "dataset_sha256_valid": bool(valid),
        "dataset_sha256": sources,
    }
    (run_dir / "run_config.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    if not valid:
        raise ValueError("one or more dataset video hashes do not match the supplied metadata")
    return record


def extract_features(ids, run_dir: Path) -> dict:
    return features.run_all(list(ids), output_dir=run_dir / "features")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _variant_rows():
    definitions = [
        ("bf_5", "bf", 5, {"bf": 5}), ("bf_8", "bf", 8, {"bf": 8}),
        ("refs_6", "refs", 6, {"refs": 6}),
        ("aq_mode_2", "aq-mode", 2, {"aq": 2}), ("aq_mode_3", "aq-mode", 3, {"aq": 3}),
        ("aq_strength_0.6", "aq-strength", 0.6, {"aq_str": 0.6}),
        ("aq_strength_1.4", "aq-strength", 1.4, {"aq_str": 1.4}),
        ("deblock_-1_-1", "deblock", [-1, -1], {"deblock": (-1, -1)}),
        ("deblock_1_1", "deblock", [1, 1], {"deblock": (1, 1)}),
        ("psy_rd_0.5", "psy-rd", 0.5, {"psy_rd": 0.5}),
        ("psy_rd_1.5", "psy-rd", 1.5, {"psy_rd": 1.5}),
        ("preset_slow", "preset", "slow", {"preset": "slow"}),
        ("g_60", "g", 60, {"g": 60}),
    ]
    return [(name, dimension, value, dict(ANCHOR, **overrides))
            for name, dimension, value, overrides in definitions]


def _base_row(vid: str, ratio: float, candidate: str, dimension: str,
              value, feat: dict, summary: dict) -> dict:
    return {"video_id": vid, "ratio": ratio, "candidate": candidate,
            "dimension": dimension, "setting": json.dumps(value, ensure_ascii=False),
            "mafd": feat["stats"]["mafd"]["mean"],
            "edge_ratio": feat["stats"]["edge_ratio"]["mean"],
            "luma_std": feat["stats"]["luma_std"]["mean"],
            "bpp": summary["bpp"], **summary}


def _heatmap(rows: list[dict], metric: str, output: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    candidates = ["anchor"] + [item[0] for item in _variant_rows()]
    keys = sorted({(r["video_id"], float(r["ratio"])) for r in rows})
    lookup = {(r["video_id"], float(r["ratio"]), r["candidate"]): r for r in rows}
    anchor_values = {(vid, ratio): _num(lookup.get((vid, ratio, "anchor"), {}).get(metric)) for vid, ratio in keys}
    matrix = np.full((len(keys), len(candidates)), np.nan)
    for i, key in enumerate(keys):
        base = anchor_values[key]
        for j, cand in enumerate(candidates):
            value = _num(lookup.get((*key, cand), {}).get(metric))
            if value is not None and base is not None:
                matrix[i, j] = value - base
    fig, ax = plt.subplots(figsize=(max(12, 0.65 * len(candidates)), max(6, 0.28 * len(keys))))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-max(0.2, np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 0.2),
                      vmax=max(0.2, np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 0.2))
    ax.set_xticks(range(len(candidates)), candidates, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)), [f"{v} {r:.0%}" for v, r in keys], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="variant - anchor (dB)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _num(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _index(run_dir: Path, title: str) -> None:
    links = []
    for filename in ("summary.csv", "sweep_summary.csv", "oracle_summary.csv", "rules.csv", "best_values.csv",
                     "loocv.csv", "val_check.csv", "comparisons.csv", "subjective_notes.csv", "attempts.jsonl", "run_config.json"):
        path = run_dir / filename
        if path.exists():
            links.append(f'<li><a href="{filename}">{filename}</a></li>')
    groups = []
    for folder in ("figs", "frames", "clips"):
        directory = run_dir / folder
        if directory.exists():
            for path in sorted(directory.glob("*")):
                rel = path.relative_to(run_dir).as_posix()
                if path.suffix.lower() in (".png", ".jpg", ".svg"):
                    groups.append(f'<figure><a href="{html.escape(rel)}"><img src="{html.escape(rel)}" style="max-width:900px"></a><figcaption>{html.escape(path.name)}</figcaption></figure>')
                elif path.suffix.lower() == ".mp4":
                    groups.append(f'<figure><video controls preload="none" src="{html.escape(rel)}"></video><figcaption>{html.escape(path.name)}</figcaption></figure>')
    text = f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><ul>{"".join(links)}</ul>{"".join(groups)}'
    (run_dir / "index.html").write_text(text, encoding="utf-8")


def run_features(run_name: str = "features_v2") -> Path:
    run_dir = create_run(run_name)
    write_run_config(run_dir, "features")
    extract_features(config.ALL_IDS, run_dir)
    _index(run_dir, f"{run_name}: dataset features")
    return run_dir


def sweep(run_name: str) -> Path:
    ids = require_development_ids(config.DEV_IDS)
    run_dir = create_run(run_name)
    write_run_config(run_dir, "development sweep")
    feature_map = extract_features(ids, run_dir)
    candidates = [("anchor", "anchor", "anchor", dict(ANCHOR))] + [
        (name, dimension, value, params) for name, dimension, value, params in _variant_rows()]
    summary_rows, attempt_rows = [], []
    total_jobs = len(ids) * len(config.RATIOS) * len(candidates)
    job_index = 0
    for vid in ids:
        feat = feature_map[vid]
        meta = meta_mod.load_meta(vid)
        for ratio in config.RATIOS:
            for candidate, dimension, value, params in candidates:
                job_index += 1
                print(f"[{job_index}/{total_jobs}] {vid} {ratio:.0%} {candidate}", flush=True)
                summary = pipeline.run_candidate(vid, ratio, candidate, params, feat, meta, run_dir)
                summary_rows.append(_base_row(vid, ratio, candidate, dimension, value, feat, summary))
    lookup = {(r["video_id"], float(r["ratio"]), r["candidate"]): r for r in summary_rows}
    sweep_rows = []
    for row in summary_rows:
        key = (row["video_id"], float(row["ratio"]))
        anchor = lookup.get((*key, "anchor"), {})
        psnr_delta = _num(row.get("psnr_y_db")) - _num(anchor.get("psnr_y_db")) if _num(row.get("psnr_y_db")) is not None and _num(anchor.get("psnr_y_db")) is not None else None
        ms_delta = _num(row.get("ms_ssim_db")) - _num(anchor.get("ms_ssim_db")) if _num(row.get("ms_ssim_db")) is not None and _num(anchor.get("ms_ssim_db")) is not None else None
        row.update(delta_psnr_y_db=psnr_delta, delta_ms_ssim_db=ms_delta,
                   significantly_better=bool(ms_delta is not None and psnr_delta is not None and
                       ms_delta >= SIGNIFICANCE_MSSSIM_DB and psnr_delta >= SIGNIFICANCE_PSNR_DB))
        sweep_rows.append(row)
    _write_csv(run_dir / "summary.csv", summary_rows)
    _write_csv(run_dir / "sweep_summary.csv", sweep_rows)
    figs = run_dir / "figs"
    _heatmap(sweep_rows, "ms_ssim_db", figs / "sweep_heatmap_msssim.png", "DEV sweep: ΔMS-SSIM(dB) from anchor")
    _heatmap(sweep_rows, "psnr_y_db", figs / "sweep_heatmap_psnr.png", "DEV sweep: ΔPSNR-Y(dB) from anchor")
    _index(run_dir, f"{run_name}: DEV sweep")
    return run_dir


def _feature(row: dict, name: str) -> float:
    return float(row[name])


def _separator(samples: list[dict], labels: list[bool], name: str):
    values = sorted(set(_feature(row, name) for row in samples))
    for left, right in zip(values, values[1:]):
        threshold = (left + right) / 2
        high = [_feature(row, name) > threshold for row in samples]
        if high == labels:
            return {"feature": name, "operator": "gt", "threshold": threshold}
        low = [_feature(row, name) <= threshold for row in samples]
        if low == labels:
            return {"feature": name, "operator": "le", "threshold": threshold}
    return None


def derive_rules(rows: list[dict], max_rules: int = MAX_RULES) -> list[dict]:
    samples = sorted({(r["video_id"], float(r["ratio"])) for r in rows})
    by_key = {(r["video_id"], float(r["ratio"]), r["candidate"]): r for r in rows}
    qualified = []
    for dimension in (x[1] for x in _variant_rows()):
        if dimension in {x[1] for x in qualified}:
            continue
        parameter = decide.PARAMETERS[dimension]
        settings = [(name, value) for name, dim, value, _ in _variant_rows() if dim == dimension]
        for candidate, value in settings:
            trials = [by_key.get((vid, ratio, candidate)) for vid, ratio in samples]
            anchors = [by_key.get((vid, ratio, "anchor")) for vid, ratio in samples]
            if any(t is None or a is None or _num(t.get("ms_ssim_db")) is None or
                   _num(a.get("ms_ssim_db")) is None or _num(t.get("psnr_y_db")) is None or
                   _num(a.get("psnr_y_db")) is None for t, a in zip(trials, anchors)):
                continue
            labels = [(float(t["ms_ssim_db"]) - float(a["ms_ssim_db"]) >= SIGNIFICANCE_MSSSIM_DB and
                       float(t["psnr_y_db"]) - float(a["psnr_y_db"]) >= SIGNIFICANCE_PSNR_DB)
                      for t, a in zip(trials, anchors)]
            if not any(labels) or all(labels):
                continue
            sample_rows = [dict(t) for t in trials]
            for feature_name in PRE_REGISTERED_FEATURES:
                separator = _separator(sample_rows, labels, feature_name)
                if not separator:
                    continue
                positive = sum(labels)
                mean_delta = float(np.mean([float(t["delta_ms_ssim_db"]) for t, label in zip(trials, labels) if label]))
                qualified.append({"dimension": dimension, "feature": separator["feature"],
                    "operator": separator["operator"], "threshold": separator["threshold"],
                    "parameter": parameter, "value": value, "fallback": ANCHOR[parameter],
                    "source": "sweep_summary.csv", "significant_samples": positive,
                    "mean_positive_delta_ms_ssim_db": mean_delta, "candidate": candidate})
    qualified.sort(key=lambda rule: (-rule["significant_samples"],
                                     -rule["mean_positive_delta_ms_ssim_db"], rule["dimension"]))
    chosen = []
    for rule in qualified:
        if rule["dimension"] not in {r["dimension"] for r in chosen}:
            chosen.append(rule)
        if len(chosen) == max_rules:
            break
    return chosen


def _same_value(a, b):
    if isinstance(a, list):
        a = tuple(a)
    if isinstance(b, list):
        b = tuple(b)
    return a == b


def _apply_rules(feat_row: dict, rules: list[dict], dimension: str, bpp: float):
    parameter = decide.PARAMETERS[dimension]
    value = ANCHOR[parameter]
    for rule in rules:
        if rule["dimension"] != dimension:
            continue
        observed = bpp if rule["feature"] == "bpp" else float(feat_row[rule["feature"]])
        match = observed > float(rule["threshold"]) if rule["operator"] == "gt" else observed <= float(rule["threshold"])
        value = rule["value"] if match else rule["fallback"]
    return value


def derive(run_name: str) -> dict:
    run_dir = config.RUN_DIR / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    raw = _read_csv(run_dir / "sweep_summary.csv")
    # Derivation consumes only DEV rows; the guard is repeated at the data boundary.
    require_development_ids(sorted({row["video_id"] for row in raw}))
    rules = derive_rules(raw)
    rules_path = config.ROOT / "src" / "rules_v2.json"
    rules_path.write_text(json.dumps({"pre_registered": {
        "metric": "delta_ms_ssim_db", "significance_ms_ssim_db": SIGNIFICANCE_MSSSIM_DB,
        "minimum_delta_psnr_y_db": SIGNIFICANCE_PSNR_DB, "features": list(PRE_REGISTERED_FEATURES),
        "max_rules": MAX_RULES}, "rules": rules}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(run_dir / "rules.csv", rules)
    oracle_rows = []
    by_pair = {}
    for row in raw:
        if row["status"] == "ok" and _num(row.get("ms_ssim_db")) is not None:
            by_pair.setdefault((row["video_id"], float(row["ratio"])), []).append(row)
    for (vid, ratio), candidates in sorted(by_pair.items()):
        oracle = max(candidates, key=lambda row: (float(row["ms_ssim_db"]), float(row.get("psnr_y_db") or -1e9)))
        oracle_rows.append({"video_id": vid, "ratio": ratio, "group": "Oracle",
                            "chosen_sweep_candidate": oracle["candidate"],
                            "chosen_dimension": oracle["dimension"], "chosen_setting": oracle["setting"],
                            "ms_ssim_db": oracle["ms_ssim_db"], "psnr_y_db": oracle["psnr_y_db"],
                            "size_error": oracle["size_error"], "output_path": oracle["output_path"]})
    _write_csv(run_dir / "oracle_summary.csv", oracle_rows)
    _write_best_values_and_loocv(raw, run_dir, rules)
    _index(run_dir, f"{run_name}: DEV scan and frozen derivation")
    return {"rules": rules, "path": rules_path}


def _write_best_values_and_loocv(rows: list[dict], run_dir: Path, full_rules: list[dict]) -> None:
    dimensions = list(dict.fromkeys(dim for _, dim, _, _ in _variant_rows()))
    by_key = {(r["video_id"], float(r["ratio"]), r["candidate"]): r for r in rows}
    keys = sorted({(r["video_id"], float(r["ratio"])) for r in rows})
    best_rows = []
    for vid, ratio in keys:
        for dimension in dimensions:
            candidates = [("anchor", "anchor", ANCHOR)] + [
                (name, dim, dict(ANCHOR, **overrides))
                for name, dim, _, overrides in _variant_rows() if dim == dimension]
            available = [by_key[(vid, ratio, name)] for name, _, _ in candidates
                         if (vid, ratio, name) in by_key and _num(by_key[(vid, ratio, name)].get("ms_ssim_db")) is not None]
            if not available:
                continue
            best = max(available, key=lambda row: (float(row["ms_ssim_db"]), float(row.get("psnr_y_db") or -1e9)))
            chosen_params = next((params for name, _, params in candidates if name == best["candidate"]), ANCHOR)
            best_rows.append({"video_id": vid, "ratio": ratio, "dimension": dimension,
                              "best_candidate": best["candidate"], "best_value": json.dumps(chosen_params[decide.PARAMETERS[dimension]]),
                              "ms_ssim_db": best["ms_ssim_db"], "psnr_y_db": best["psnr_y_db"],
                              **{f: best[f] for f in PRE_REGISTERED_FEATURES}})
    _write_csv(run_dir / "best_values.csv", best_rows)
    loocv = []
    ids = sorted({r["video_id"] for r in rows})
    for held_out in ids:
        train = [r for r in rows if r["video_id"] != held_out]
        require_development_ids({r["video_id"] for r in train})
        fold_rules = derive_rules(train)
        for sample in (r for r in rows if r["video_id"] == held_out and r["candidate"] == "anchor"):
            ratio = float(sample["ratio"])
            for dimension in dimensions:
                parameter = decide.PARAMETERS[dimension]
                predicted = _apply_rules(sample, fold_rules, dimension, float(sample["bpp"]))
                options = [(r, json.loads(r["parameters"]).get(parameter)) for r in rows
                           if r["video_id"] == held_out and float(r["ratio"]) == ratio and
                           (r["dimension"] == dimension or r["candidate"] == "anchor") and
                           _num(r.get("ms_ssim_db")) is not None]
                if not options:
                    continue
                oracle_row, oracle_value = max(options, key=lambda pair: (float(pair[0]["ms_ssim_db"]), float(pair[0].get("psnr_y_db") or -1e9)))
                predicted_row = next((r for r, value in options if _same_value(value, predicted)), None)
                loocv.append({"held_out_video": held_out, "ratio": ratio, "dimension": dimension,
                              "predicted_value": json.dumps(predicted), "oracle_value": json.dumps(oracle_value),
                              "prediction_equal_oracle": _same_value(predicted, oracle_value),
                              "predicted_ms_ssim_db": predicted_row["ms_ssim_db"] if predicted_row else "",
                              "oracle_ms_ssim_db": oracle_row["ms_ssim_db"],
                              "ms_ssim_loss_db": float(oracle_row["ms_ssim_db"]) - float(predicted_row["ms_ssim_db"]) if predicted_row else ""})
    _write_csv(run_dir / "loocv.csv", loocv)
    _plot_best_values(best_rows, run_dir / "figs")


def _plot_best_values(rows: list[dict], figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    for dimension in dict.fromkeys(r["dimension"] for r in rows):
        subset = [r for r in rows if r["dimension"] == dimension]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, feature_name in zip(axes.ravel(), PRE_REGISTERED_FEATURES):
            x = [float(r[feature_name]) for r in subset]
            y = [float(r["ms_ssim_db"]) for r in subset]
            ax.scatter(x, y, alpha=0.8)
            for row, xx, yy in zip(subset, x, y):
                ax.annotate(f"{row['video_id'][-2:]} {float(row['ratio']):.0%}", (xx, yy), fontsize=6)
            ax.set_xlabel(feature_name)
            ax.set_ylabel("oracle MS-SSIM (dB)")
            ax.grid(alpha=0.2)
        fig.suptitle(f"DEV oracle setting vs pre-registered features: {dimension}")
        fig.tight_layout()
        fig.savefig(figdir / f"oracle_{dimension.replace('-', '_')}.png", dpi=150)
        plt.close(fig)


def _load_rules(path: Path | None = None) -> list[dict]:
    path = path or config.ROOT / "src" / "rules_v2.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rules"]


def validate(run_name: str, rules_path: Path | None = None) -> Path:
    run_dir = create_run(run_name)
    write_run_config(run_dir, "VAL rule generalization check")
    rule_list = _load_rules(rules_path)
    feature_map = extract_features(config.VAL_IDS, run_dir)
    rows = []
    for vid in config.VAL_IDS:
        feat, meta = feature_map[vid], meta_mod.load_meta(vid)
        for ratio in config.RATIOS:
            bud = meta_mod.budget(meta, ratio)
            bpp = bud["video_kbps"] * 1000 / (meta["width"] * meta["height"] * meta["fps"])
            for group in ("B1", "A"):
                plan = decide.decide(feat, bud, group, rule_list, bpp)
                result = pipeline.run_candidate(vid, ratio, group, plan["params"], feat, meta, run_dir)
                result.update(video_id=vid, ratio=ratio, group=group, reasons="; ".join(plan["reasons"]))
                rows.append(result)
    paired = {(r["video_id"], float(r["target_ratio"]), r["group"]): r for r in rows}
    for row in rows:
        baseline = paired.get((row["video_id"], float(row["target_ratio"]), "B1"), {})
        if row["group"] == "A" and row.get("status") == "ok" and baseline.get("status") == "ok":
            row["delta_psnr_y_db"] = float(row["psnr_y_db"]) - float(baseline["psnr_y_db"])
            row["delta_ms_ssim_db"] = float(row["ms_ssim_db"]) - float(baseline["ms_ssim_db"])
    _write_csv(run_dir / "val_check.csv", rows)
    _write_csv(run_dir / "summary.csv", rows)
    _index(run_dir, f"{run_name}: VAL-only rule check")
    return run_dir


def final_guard() -> None:
    tag = _git("rev-parse", "--verify", "refs/tags/frozen-v2", check=False)
    if not tag:
        raise RuntimeError("final requires the frozen-v2 tag")
    if tag != _git("rev-parse", "HEAD"):
        raise RuntimeError("final requires HEAD to be exactly the frozen-v2 commit")
    status = _git("status", "--porcelain", "--untracked-files=all").splitlines()
    if status:
        raise RuntimeError("final requires a clean worktree: " + "; ".join(status))


def final(run_name: str = "final_v2") -> Path:
    final_guard()
    run_dir = create_run(run_name)
    write_run_config(run_dir, "frozen final experiment")
    rule_list = _load_rules()
    feature_map = extract_features(config.ALL_IDS, run_dir)
    rows = []
    total = len(config.ALL_IDS) * len(config.RATIOS) * 3
    index = 0
    for vid in config.ALL_IDS:
        feat, meta = feature_map[vid], meta_mod.load_meta(vid)
        for ratio in config.RATIOS:
            bud = meta_mod.budget(meta, ratio)
            bpp = bud["video_kbps"] * 1000 / (meta["width"] * meta["height"] * meta["fps"])
            for group in ("B0", "B1", "A"):
                index += 1
                plan = decide.decide(feat, bud, group, rule_list, bpp)
                print(f"[{index}/{total}] {vid} {ratio:.0%} {group}", flush=True)
                result = pipeline.run_candidate(vid, ratio, group, plan["params"], feat, meta,
                                                run_dir, reasons=plan["reasons"])
                result.update(video_id=vid, ratio=ratio, group=group,
                              mafd=feat["stats"]["mafd"]["mean"],
                              edge_ratio=feat["stats"]["edge_ratio"]["mean"],
                              luma_std=feat["stats"]["luma_std"]["mean"],
                              reasons="; ".join(plan["reasons"]))
                rows.append(result)
    _write_csv(run_dir / "summary.csv", rows)
    _index(run_dir, f"{run_name}: frozen final experiment")
    return run_dir


def alignment_check(sweep_run: str) -> Path:
    source_run = config.RUN_DIR / sweep_run
    if not (source_run / "sweep_summary.csv").is_file():
        raise FileNotFoundError(source_run / "sweep_summary.csv")
    check_dir = config.CHECK_DIR / "msssim_alignment"
    check_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    rng = np.random.default_rng(5185)
    reference = rng.integers(16, 240, size=(720, 1280), dtype=np.uint8)
    noise = rng.normal(0, 1, reference.shape)
    self_score = ms_ssim(reference, reference)
    noisy_scores = [ms_ssim(reference, np.clip(reference.astype(float) + noise * level, 0, 255).astype(np.uint8))
                    for level in (1, 2, 5, 10)]
    assert self_score == 1.0, f"self comparison is {self_score}"
    assert all(a > b for a, b in zip([self_score] + noisy_scores, noisy_scores)), noisy_scores
    selected = [("01_FourPeople", 0.05), ("02_Beauty", 0.05), ("03_Bosphorus", 0.05)]
    alignment_rows = []
    for vid, ratio in selected:
        summary_path = source_run / "summary.csv"
        summaries = _read_csv(summary_path)
        row = next((item for item in summaries if item["video_id"] == vid and
                    float(item["ratio"]) == ratio and item["candidate"] == "anchor" and item["status"] == "ok"), None)
        if row is None:
            raise ValueError(f"no successful anchor candidate for {vid} {ratio:.0%}")
        meta = meta_mod.load_meta(vid)
        indices, original = sample_y_frames(config.RAW_DIR / meta["file"], meta["width"], meta["height"], meta["frames"], 10)
        _, encoded = sample_y_frames(row["output_path"], meta["width"], meta["height"], meta["frames"], 10)
        ours = [ms_ssim(a, b) for a, b in zip(original, encoded)]
        npz = check_dir / f"{vid}.npz"
        np.savez(npz, reference=np.stack(original), distorted=np.stack(encoded))
        torch_result = check_dir / f"{vid}_torch.json"
        helper = config.ROOT / "tools" / "msssim_reference.py"
        python = Path(r"D:\gcg\Multimedia\project\_venv_msssim\Scripts\python.exe")
        _run([str(python), str(helper), str(npz), str(torch_result)])
        reference_result = json.loads(torch_result.read_text(encoding="utf-8"))
        for idx, ours_value, reference_value in zip(indices, ours, reference_result["values"]):
            alignment_rows.append({"video_id": vid, "ratio": ratio, "frame_index": idx,
                "ours": ours_value, "pytorch_msssim": reference_value,
                "absolute_difference": abs(ours_value - reference_value),
                "torch_version": reference_result["torch"],
                "pytorch_msssim_version": reference_result["pytorch_msssim"]})
    alignment_path = config.CHECK_DIR / "msssim_alignment.csv"
    _write_csv(alignment_path, alignment_rows)
    max_difference = max(row["absolute_difference"] for row in alignment_rows)
    (config.CHECK_DIR / "msssim_alignment_basic.json").write_text(json.dumps({
        "self_comparison": self_score, "noise_levels": [1, 2, 5, 10],
        "noise_scores": [1.0] + noisy_scores,
        "strictly_monotonic_decrease": all(a > b for a, b in zip([1.0] + noisy_scores, noisy_scores[1:] + [noisy_scores[-1] - 1e-12])),
        "alignment_samples": len(alignment_rows), "max_absolute_difference": max_difference,
        "torch_version": alignment_rows[0]["torch_version"],
        "pytorch_msssim_version": alignment_rows[0]["pytorch_msssim_version"]}, indent=2), encoding="utf-8")
    if max_difference > 1e-4:
        raise ValueError(f"MS-SSIM reference alignment exceeds 1e-4: {max_difference}")
    return alignment_path


def over_target_check() -> Path:
    check_root = config.CHECK_DIR / "over_target_retry_v3"
    check_root.mkdir(parents=True, exist_ok=True)
    vid, ratio = config.DEV_IDS[0], 0.05
    meta = meta_mod.load_meta(vid)
    feat = features.extract(config.RAW_DIR / meta["file"])
    base = meta_mod.budget(meta, ratio)
    attempts = []
    triggered = None
    for multiplier in (1.15, 1.30, 1.60, 2.00):
        trial_dir = check_root / f"rate_{round(multiplier * 100)}pct"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "perframe").mkdir(exist_ok=True)
        plan = decide.decide(feat, base, "B1", [])
        result = pipeline.run_candidate(vid, ratio, f"E3_{round(multiplier*100)}", plan["params"], feat,
                                        meta, trial_dir, initial_kbps=round(base["video_kbps"] * multiplier),
                                        reasons=plan["reasons"])
        log = trial_dir / "attempts.jsonl"
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        first = records[0]
        attempts.append({"video_id": vid, "ratio": ratio, "initial_multiplier": multiplier,
                         "initial_kbps": first["planned_kbps"],
                         "first_actual_bytes": first.get("measurements", {}).get("actual_bytes"),
                         "first_size_error": first.get("measurements", {}).get("size_error"),
                         "first_next_action": first["next_action"],
                         "retry_triggered": first["next_action"] == "retry",
                         "attempt_count": result["attempts"], "final_status": result["status"]})
        if first["next_action"] == "retry" and first.get("measurements", {}).get("size_error", -1) > 0:
            triggered = attempts[-1]
            break
    _write_csv(check_root / "over_target_retry.csv", attempts)
    if not triggered:
        raise RuntimeError("E3 check did not produce an over-target first pass; see check CSV")
    return check_root / "over_target_retry.csv"
