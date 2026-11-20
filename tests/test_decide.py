import pytest

from src.decide import decide
from src.experiment import require_development_ids


def feature(mafd, edge=0.1):
    return {"stats": {"mafd": {"mean": mafd}, "edge_ratio": {"mean": edge},
                      "luma_std": {"mean": 40.0}}, "scene_cuts": []}


def test_content_rule_changes_parameter_when_feature_crosses_threshold():
    rule = {"feature": "mafd", "threshold": 5.0, "operator": "gt",
            "parameter": "bf", "value": 5, "fallback": 3,
            "dimension": "bf"}
    budget = {"video_kbps": 1200}
    low = decide(feature(2.0), budget, "A", [rule])
    high = decide(feature(8.0), budget, "A", [rule])
    assert low["params"]["bf"] == 3
    assert high["params"]["bf"] == 5


def test_bpp_rule_changes_parameter_across_threshold():
    rule = {"feature": "bpp", "threshold": 0.02, "operator": "gt",
            "parameter": "aq", "value": 2, "fallback": 1,
            "dimension": "aq-mode"}
    budget = {"video_kbps": 1200}
    low = decide(feature(2.0), budget, "A", [rule], bpp=0.01)
    high = decide(feature(2.0), budget, "A", [rule], bpp=0.03)
    assert low["params"]["aq"] == 1
    assert high["params"]["aq"] == 2


def test_sweep_rejects_any_test_video():
    with pytest.raises(ValueError, match="TEST"):
        require_development_ids(["01_FourPeople", "09_Parkrun"])
