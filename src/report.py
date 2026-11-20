"""Figures and subjective synchronization materials for one immutable run."""
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config, experiment, features, metadata
from .encode import _exe


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(row, key):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _features(run: Path, vid: str) -> dict:
    return json.loads((run / "features" / f"{vid}.json").read_text(encoding="utf-8"))


def _frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open visual comparison video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read visual comparison frame {index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _choose_region(original, b0, adaptive):
    gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.int16)
            for image in (original, b0, adaptive)]
    difference = np.maximum(np.abs(gray[0] - gray[1]), np.abs(gray[0] - gray[2])).astype(np.float64)
    patch, step = 128, 16
    integral = cv2.integral(difference)
    sums = (integral[patch:, patch:] - integral[:-patch, patch:]
            - integral[patch:, :-patch] + integral[:-patch, :-patch])
    grid = sums[::step, ::step]
    gy, gx = np.unravel_index(np.argmax(grid), grid.shape)
    return gx * step, gy * step, patch


def _comparison_materials(run: Path, rows: list[dict]) -> list[dict]:
    by_key = {(r["video_id"], float(r["ratio"]), r["group"]): r for r in rows}
    notes, clips = [], run / "clips"
    clips.mkdir(exist_ok=True)
    frames_dir = run / "frames"
    frames_dir.mkdir(exist_ok=True)
    ff = _exe("ffmpeg")
    for vid in config.TEST_IDS:
        feat = _features(run, vid)
        meta = metadata.load_meta(vid)
        for ratio in config.RATIOS:
            adaptive = by_key.get((vid, ratio, "A"))
            baseline = by_key.get((vid, ratio, "B0"))
            if not adaptive or not baseline or adaptive.get("status") != "ok" or baseline.get("status") != "ok":
                for category in ("运动流畅度", "模糊", "块效应", "重影", "细节损失"):
                    notes.append({"video_id": vid, "ratio": ratio, "category": category,
                                  "observation": "未确认：B0/A 未同时通过硬性验收", "frame": "", "region": ""})
                continue
            perframe_path = run / "perframe" / f"{vid}_A_{int(ratio*100):02d}pct.json"
            perframe = json.loads(perframe_path.read_text(encoding="utf-8"))
            quality = perframe["ms_ssim_per_frame"]
            baseline_perframe_path = run / "perframe" / f"{vid}_B0_{int(ratio*100):02d}pct.json"
            baseline_perframe = json.loads(baseline_perframe_path.read_text(encoding="utf-8"))
            low_frame = int(np.argmin(quality))
            edge_peak = int(np.argmax(feat["per_frame"]["edge_ratio"]))
            original_path = str(config.RAW_DIR / meta["file"])
            if edge_peak == low_frame:
                edge_values = np.asarray(feat["per_frame"]["edge_ratio"], dtype=float)
                edge_peak = int(np.argsort(edge_values)[-2])
            selected = (low_frame, edge_peak)
            for index in selected:
                original = _frame(original_path, index)
                b0 = _frame(baseline["output_path"], index)
                adaptive_frame = _frame(adaptive["output_path"], index)
                x, y, size = _choose_region(original, b0, adaptive_frame)
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
                from matplotlib.patches import Rectangle
                for ax, image, label in zip(axes[:3], (original, b0, adaptive_frame),
                                            ("Original", "B0", "A")):
                    ax.imshow(image)
                    ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="yellow", linewidth=1.5))
                    ax.set_title(label)
                    ax.axis("off")
                crop = adaptive_frame[y:y+size, x:x+size]
                axes[3].imshow(cv2.resize(crop, (size*2, size*2), interpolation=cv2.INTER_NEAREST))
                axes[3].set_title(f"A crop 2x ({x},{y})")
                axes[3].axis("off")
                fps = meta["fps"]
                title = (f"{vid} {ratio:.0%} | frame {index} @ {index/fps:.3f}s | "
                         f"file kbps B0/A: {_float(baseline, 'file_kbps') or 0:.0f}/"
                         f"{_float(adaptive, 'file_kbps') or 0:.0f} | "
                         f"PSNR-Y B0/A: {_float(baseline, 'psnr_y_db') or 0:.3f}/"
                         f"{_float(adaptive, 'psnr_y_db') or 0:.3f} dB | "
                         f"MS-SSIM dB B0/A: {_float(baseline, 'ms_ssim_db') or 0:.3f}/"
                         f"{_float(adaptive, 'ms_ssim_db') or 0:.3f} | frame MS-SSIM B0/A: "
                         f"{baseline_perframe['ms_ssim_per_frame'][index]:.5f}/"
                         f"{perframe['ms_ssim_per_frame'][index]:.5f}")
                fig.suptitle(title, fontsize=9)
                fig.tight_layout()
                suffix = "lowest_msssim" if index == low_frame else "edge_peak"
                fig.savefig(frames_dir / f"{vid}_{int(ratio*100):02d}pct_{suffix}.png", dpi=150)
                plt.close(fig)
                region = f"x={x},y={y},size={size}x{size}"
                for category in ("運動流暢度", "模糊", "塊效應", "重影", "細節損失"):
                    notes.append({"video_id": vid, "ratio": ratio, "category": category,
                                  "observation": "未確認：需逐像素/播放主觀复核",
                                  "frame": index, "region": region})
            clip = clips / f"{vid}_{int(ratio*100):02d}pct_B0_vs_A.mp4"
            command = [ff, "-y", "-i", original_path, "-i", baseline["output_path"],
                       "-i", adaptive["output_path"], "-filter_complex",
                       "[0:v]scale=640:-2[o];[1:v]scale=640:-2[b];[2:v]scale=640:-2[a];"
                       "[o][b][a]hstack=inputs=3:shortest=1[v]", "-map", "[v]", "-an",
                       "-c:v", "libx264", "-threads", str(config.ENCODE_THREADS),
                       "-preset", "medium", "-crf", "23", str(clip)]
            subprocess.run(command, check=True, capture_output=True)
    _write(run / "subjective_notes.csv", notes)
    return notes


def _comparison(rows: list[dict], run: Path) -> list[dict]:
    by_key = {(r["video_id"], float(r["target_ratio"]), r["group"]): r for r in rows}
    detail = []
    for vid in config.ALL_IDS:
        split = "DEV" if vid in config.DEV_IDS else "VAL" if vid in config.VAL_IDS else "TEST"
        for ratio in config.RATIOS:
            adaptive = by_key.get((vid, ratio, "A"), {})
            for baseline_group in ("B1", "B0"):
                baseline = by_key.get((vid, ratio, baseline_group), {})
                pa, pb = _float(adaptive, "psnr_y_db"), _float(baseline, "psnr_y_db")
                ma, mb = _float(adaptive, "ms_ssim_db"), _float(baseline, "ms_ssim_db")
                delta_psnr = pa - pb if pa is not None and pb is not None else None
                delta_ms = ma - mb if ma is not None and mb is not None else None
                detail.append({"split": split, "video_id": vid, "ratio": ratio,
                    "comparison": f"A-{baseline_group}", "delta_psnr_y_db": delta_psnr,
                    "delta_ms_ssim_db": delta_ms,
                    "result_ms_ssim": "win" if delta_ms is not None and delta_ms > 0 else
                        "loss" if delta_ms is not None and delta_ms < 0 else
                        "tie" if delta_ms == 0 else "unavailable",
                    "result_psnr": "win" if delta_psnr is not None and delta_psnr > 0 else
                        "loss" if delta_psnr is not None and delta_psnr < 0 else
                        "tie" if delta_psnr == 0 else "unavailable",
                    "A_status": adaptive.get("status", "missing"),
                    "baseline_status": baseline.get("status", "missing"),
                    "A_size_error": _float(adaptive, "size_error"),
                    "baseline_size_error": _float(baseline, "size_error")})
    aggregates = []
    for split in ("TEST", "DEV", "VAL"):
        for comparison in ("A-B1", "A-B0"):
            subset = [row for row in detail if row["split"] == split and row["comparison"] == comparison]
            for metric, outcome in (("delta_psnr_y_db", "result_psnr"), ("delta_ms_ssim_db", "result_ms_ssim")):
                vals = [_float(row, metric) for row in subset if _float(row, metric) is not None]
                aggregates.append({"split": split, "comparison": comparison, "metric": metric,
                    "paired_samples": len(vals), "mean_delta": float(np.mean(vals)) if vals else "",
                    "wins": sum(1 for row in subset if row[outcome] == "win"),
                    "losses": sum(1 for row in subset if row[outcome] == "loss"),
                    "ties": sum(1 for row in subset if row[outcome] == "tie")})
    _write(run / "comparisons.csv", detail + aggregates)
    return detail


def _plots(run: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = run / "figs"
    figdir.mkdir(exist_ok=True)
    import shutil
    for source_run in config.RUN_DIR.iterdir():
        if not source_run.is_dir() or source_run == run:
            continue
        for name in ("sweep_heatmap_msssim.png", "sweep_heatmap_psnr.png"):
            source = source_run / "figs" / name
            if source.is_file():
                shutil.copy2(source, figdir / name)
    ids = config.ALL_IDS
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, vid in zip(axes.ravel(), ids):
        feat = _features(run, vid)
        values = feat["per_frame"]["mafd"]
        time_axis = np.arange(len(values)) / feat["fps"]
        ax.plot(time_axis, values, color="#276fbf", linewidth=.8)
        for cut in feat["scene_cuts"]:
            ax.axvline(cut / feat["fps"], color="#c33", alpha=.6, linewidth=.7)
        ax.set_title(vid, fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Source motion feature timelines (red: scene-cut detections)")
    fig.tight_layout()
    fig.savefig(figdir / "feature_timelines.png", dpi=150)
    plt.close(fig)

    rule_set = experiment._load_rules()
    fig, ax = plt.subplots(figsize=(9, 6))
    for vid in config.ALL_IDS:
        feat = _features(run, vid)
        split = "DEV" if vid in config.DEV_IDS else "VAL" if vid in config.VAL_IDS else "TEST"
        ax.scatter(feat["stats"]["mafd"]["mean"], feat["stats"]["edge_ratio"]["mean"], label=f"{split} {vid}")
    for rule in rule_set:
        if rule["feature"] == "mafd":
            ax.axvline(float(rule["threshold"]), linestyle="--", label=f"DEV mafd threshold {rule['threshold']:.4g}")
        if rule["feature"] == "edge_ratio":
            ax.axhline(float(rule["threshold"]), linestyle="--", label=f"DEV edge threshold {rule['threshold']:.4g}")
    ax.set_xlabel("mafd mean")
    ax.set_ylabel("edge_ratio mean")
    ax.set_title("Frozen rules (thresholds derived from DEV only)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "decision_map.png", dpi=150)
    plt.close(fig)

    if rule_set:
        fig, axes = plt.subplots(len(rule_set), 1, figsize=(9, max(3, 2.8 * len(rule_set))), squeeze=False)
        for ax, rule in zip(axes.ravel(), rule_set):
            name = rule["feature"]
            values, colors, labels = [], [], []
            for vid in config.DEV_IDS + config.VAL_IDS:
                feat = _features(run, vid)
                if name == "bpp":
                    meta = metadata.load_meta(vid)
                    for ratio in config.RATIOS:
                        bud = metadata.budget(meta, ratio)
                        values.append(bud["video_kbps"] * 1000 / (meta["width"] * meta["height"] * meta["fps"]))
                        labels.append(f"{vid[-2:]} {ratio:.0%}")
                        colors.append("#176b45" if vid in config.DEV_IDS else "#ae6b21")
                else:
                    values.append(float(feat["stats"][name]["mean"]))
                    labels.append(vid[-2:])
                    colors.append("#176b45" if vid in config.DEV_IDS else "#ae6b21")
            xpos = np.arange(len(values))
            ax.scatter(values, xpos, c=colors)
            ax.axvline(float(rule["threshold"]), color="#a22", linestyle="--")
            ax.set_yticks(xpos, labels)
            ax.set_xlabel(name)
            ax.set_title(f"{rule['dimension']}: {rule['operator']} {rule['threshold']:.6g} -> {rule['value']}")
            ax.grid(axis="x", alpha=.2)
        fig.suptitle("Frozen DEV thresholds (green=DEV, amber=VAL check)")
        fig.tight_layout()
        fig.savefig(figdir / "decision_thresholds.png", dpi=150)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(.5, .5, "No content-dependent parameter rule met the pre-registered DEV criterion; A uses the anchor.",
                ha="center", va="center", wrap=True)
        ax.axis("off")
        fig.savefig(figdir / "decision_thresholds.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    test = [row for row in rows if row["video_id"] in config.TEST_IDS and row.get("status") == "ok"]
    for metric, label, name in (("psnr_y_db", "PSNR-Y (dB)", "group_quality_psnr.png"),
                                ("ms_ssim_db", "MS-SSIM (dB)", "group_quality_msssim.png")):
        groups = ("B0", "B1", "A")
        means = [np.mean([float(r[metric]) for r in test if r["group"] == group and _float(r, metric) is not None])
                 if any(r["group"] == group and _float(r, metric) is not None for r in test) else np.nan for group in groups]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(groups, means, color=("#888", "#4382bf", "#38a17a"))
        ax.set_ylabel(label)
        ax.set_title(f"TEST group means: {label}")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(figdir / name, dpi=150)
        plt.close(fig)

    comparisons = _read(run / "comparisons.csv")
    rows_delta = [r for r in comparisons if r.get("split") == "TEST" and r.get("metric") in ("delta_ms_ssim_db", "delta_psnr_y_db")]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, msssim, psnr = [], [], []
    for comparison in ("A-B1", "A-B0"):
        pair = {r["metric"]: r for r in rows_delta if r["comparison"] == comparison}
        labels.append(comparison)
        msssim.append(_float(pair.get("delta_ms_ssim_db", {}), "mean_delta") or 0)
        psnr.append(_float(pair.get("delta_psnr_y_db", {}), "mean_delta") or 0)
    x = np.arange(len(labels))
    ax.bar(x - .18, msssim, .36, label="ΔMS-SSIM (dB)")
    ax.bar(x + .18, psnr, .36, label="ΔPSNR-Y (dB)")
    ax.axhline(0, color="#333", linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_title("TEST adaptive deltas")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(figdir / "adaptive_deltas.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for group in ("B0", "B1", "A"):
        subset = [r for r in test if r["group"] == group and _float(r, "actual_video_kbps") is not None and _float(r, "ms_ssim_db") is not None]
        ax.scatter([float(r["actual_video_kbps"]) for r in subset],
                   [float(r["ms_ssim_db"]) for r in subset], label=group, alpha=.8)
    ax.set_xlabel("video bitrate (kbps)")
    ax.set_ylabel("MS-SSIM (dB)")
    ax.set_title("TEST rate–distortion")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figdir / "rd_scatter.png", dpi=150)
    plt.close(fig)

    case = "05_Jockey"
    feat = _features(run, case)
    perframe = {}
    for group in ("A", "B0"):
        path = run / "perframe" / f"{case}_{group}_05pct.json"
        if path.exists():
            perframe[group] = json.loads(path.read_text(encoding="utf-8"))
    case_rows = {(r["group"]): r for r in rows if r["video_id"] == case and float(r["target_ratio"]) == .05}
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    t = np.arange(feat["frames"]) / feat["fps"]
    for ax, key, label in zip(axes[:3], ("mafd", "edge_ratio", "luma_std"), ("mafd", "edge ratio", "luma std")):
        ax.plot(t, feat["per_frame"][key], label=label)
        for cut in feat["scene_cuts"]:
            ax.axvline(cut / feat["fps"], color="#c33", linestyle=":")
        ax.legend(loc="upper right")
        ax.grid(alpha=.2)
    for group, color in (("A", "#16875b"), ("B0", "#777")):
        if group in perframe:
            vals = perframe[group]["ms_ssim_per_frame"]
            axes[3].plot(np.arange(len(vals)) / feat["fps"], vals, label=f"{group} MS-SSIM", color=color)
    axes[3].legend()
    axes[3].set_ylabel("MS-SSIM")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(alpha=.2)
    a_params = case_rows.get("A", {}).get("parameters", "anchor")
    fig.suptitle(f"Case: {case} 5% | features -> frozen parameters {a_params}")
    fig.tight_layout()
    fig.savefig(figdir / "case_feature_to_parameters.png", dpi=150)
    plt.close(fig)


def _html(run: Path, rows: list[dict]) -> None:
    def esc(value):
        import html
        return html.escape(str(value if value is not None else ""))
    visible = ("video_id", "target_ratio", "group", "status", "target_bytes", "actual_bytes",
               "size_error", "psnr_y_db", "ms_ssim", "ms_ssim_db", "decode_ok", "frame_count", "output_path")
    html_rows = []
    for row in rows:
        color = ' style="background:#ffd9d9"' if _float(row, "size_error") is not None and _float(row, "size_error") > 0 else ""
        html_rows.append(f"<tr{color}>" + "".join(f"<td>{esc(row.get(k))}</td>" for k in visible) +
                         '<td><a href="attempts.jsonl#L1">attempts</a></td></tr>')
    links = [f'<li><a href="{name}">{name}</a></li>' for name in (
        "comparisons.csv", "subjective_notes.csv", "attempts.jsonl", "run_config.json", "run_report.json")
        if (run / name).exists()]
    assets = []
    for folder in ("figs", "frames", "clips"):
        for path in sorted((run / folder).glob("*")):
            rel = path.relative_to(run).as_posix()
            if path.suffix == ".png":
                assets.append(f'<figure><a href="{esc(rel)}"><img loading="lazy" src="{esc(rel)}" style="max-width:100%"></a><figcaption>{esc(path.name)}</figcaption></figure>')
            elif path.suffix == ".mp4":
                assets.append(f'<figure><video controls preload="none" src="{esc(rel)}"></video><figcaption>{esc(path.name)}</figcaption></figure>')
    table_head = "".join(f"<th>{esc(key)}</th>" for key in visible) + "<th>log</th>"
    document = ("<!doctype html><meta charset=utf-8><title>CS5185 v2 results</title>"
        "<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;font-size:11px}td,th{border:1px solid #ccc;padding:4px}figure{margin:1rem 0}video{max-width:100%}</style>"
        "<h1>CS5185 v2 experiment</h1><ul>" + "".join(links) + "</ul>"
        "<h2>Run summary (failed and over-target outputs highlighted)</h2><div style=\"overflow:auto\"><table><thead><tr>" + table_head + "</tr></thead><tbody>" + "".join(html_rows) + "</tbody></table></div><h2>Figures and subjective materials</h2>" + "".join(assets))
    (run / "index.html").write_text(document, encoding="utf-8")


def generate(run_name: str) -> Path:
    run = config.RUN_DIR / run_name
    if not run.is_dir():
        raise FileNotFoundError(run)
    rows = _read(run / "summary.csv")
    _comparison(rows, run)
    notes = _comparison_materials(run, rows)
    _plots(run, rows)
    failures = [row for row in rows if row.get("status") != "ok"]
    over = [row for row in rows if _float(row, "size_error") is not None and _float(row, "size_error") > 0]
    test = [row for row in rows if row["video_id"] in config.TEST_IDS]
    report = {"test_rows": len(test), "test_failures": len([r for r in test if r.get("status") != "ok"]),
              "test_over_target_rows": len([r for r in test if _float(r, "size_error") is not None and _float(r, "size_error") > 0]),
              "size_error_by_group": {group: [_float(r, "size_error") for r in rows if r.get("group") == group and _float(r, "size_error") is not None]
                                      for group in ("B0", "B1", "A")},
              "all_failed_rows": len(failures), "all_over_target_rows": len(over),
              "rules": experiment._load_rules(), "subjective_note_rows": len(notes),
              "commands_run": ["features", "sweep", "derive", "validate", "final", "report"]}
    (run / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _html(run, rows)
    return run / "index.html"
