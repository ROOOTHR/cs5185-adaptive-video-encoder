"""Rule-driven x264 decisions and command construction."""
from __future__ import annotations

import json
from pathlib import Path

ANCHOR = dict(preset="medium", g=150, bf=3, refs=3, aq=1, aq_str=1.0,
              deblock=(0, 0), psy_rd=1.0)
B0 = dict(ANCHOR, g=60)

FEATURES = ("edge_ratio", "mafd", "luma_std", "bpp")
RULES_PATH = Path(__file__).with_name("rules_v2.json")
RULES = json.loads(RULES_PATH.read_text(encoding="utf-8"))["rules"] if RULES_PATH.exists() else []

PARAMETERS = {"bf": "bf", "refs": "refs", "aq-mode": "aq",
              "aq-strength": "aq_str", "deblock": "deblock",
              "psy-rd": "psy_rd", "preset": "preset", "g": "g"}


def feature_value(feat: dict, name: str, bpp: float | None = None) -> float:
    if name == "bpp":
        if bpp is None:
            raise ValueError("bpp is required by this rule")
        return float(bpp)
    return float(feat["stats"][name]["mean"])


def decide(feat: dict, bud: dict, mode: str = "A", rules=None,
           bpp: float | None = None) -> dict:
    if mode not in {"A", "B0", "B1"}:
        raise ValueError(f"unknown group: {mode}")
    params = dict(B0 if mode == "B0" else ANCHOR)
    reasons = [f"{mode}: " + ("统一基线 g=60" if mode == "B0" else
                              "统一校准锚点" if mode == "B1" else
                              "冻结规则按扫描结果选择参数")]
    if mode == "A":
        for rule in rules:
            name = rule["feature"]
            observed = feature_value(feat, name, bpp)
            threshold = float(rule["threshold"])
            matched = observed > threshold if rule["operator"] == "gt" else observed <= threshold
            parameter = rule["parameter"]
            params[parameter] = rule["value"] if matched else rule["fallback"]
            observed_relation = ">" if observed > threshold else "<="
            branch = "命中" if matched else "未命中"
            reasons.append(
                f"检测到 {name}={observed:.6g} {observed_relation} {threshold:.6g}，{branch}规则 -> "
                f"{parameter}={params[parameter]} "
                f"({rule.get('source', 'validated rule')})")
    return {"mode": mode, "params": params, "reasons": reasons,
            "motion": "low" if feat["stats"]["mafd"]["mean"] < 5 else "high",
            "detail": "low" if feat["stats"]["edge_ratio"]["mean"] < 0.1 else "high"}


def ffmpeg_args(plan: dict, bud: dict, job: str, pass_no: int,
                out_path: str, threads: int = 4) -> list[str]:
    p = plan["params"]
    kbps = int(bud["video_kbps"])
    x264 = f"deblock={p['deblock'][0]},{p['deblock'][1]}:psy-rd={p['psy_rd']},0"
    common = ["-c:v", "libx264", "-threads", str(threads), "-preset", str(p["preset"]),
              "-pix_fmt", "yuv420p", "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k",
              "-bufsize", f"{kbps}k", "-g", str(p["g"]), "-bf", str(p["bf"]),
              "-refs", str(p["refs"]), "-aq-mode", str(p["aq"]),
              "-aq-strength", str(p["aq_str"]), "-x264-params", x264]
    if pass_no == 1:
        return common + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null", "NUL"]
    return common + ["-pass", "2", "-passlogfile", job, "-an",
                     "-movflags", "+faststart", out_path]
