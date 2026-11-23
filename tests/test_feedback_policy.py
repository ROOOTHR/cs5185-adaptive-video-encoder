import pytest

from src import config, pipeline

TARGET = 1_000_000


def test_proportional_keeps_the_original_rule():
    # under target: no margin; over target: extra 0.98
    assert pipeline.next_rate("proportional", [(1000, 900_000)], TARGET) == pytest.approx(1000 * TARGET / 900_000)
    assert pipeline.next_rate("proportional", [(1000, 1_100_000)], TARGET) == pytest.approx(
        1000 * TARGET / 1_100_000 * config.OVERHEAD)


def test_center_aims_at_the_middle_of_the_window_on_both_sides():
    aim = TARGET * pipeline.AIM_FRACTION
    assert pipeline.next_rate("center", [(1000, 900_000)], TARGET) == pytest.approx(1000 * aim / 900_000)
    assert pipeline.next_rate("center", [(1000, 1_100_000)], TARGET) == pytest.approx(1000 * aim / 1_100_000)


def test_secant_uses_two_points_when_the_slope_is_sane():
    aim = TARGET * pipeline.AIM_FRACTION
    history = [(1000, 900_000), (1100, 1_000_000)]   # slope 1000 B/kbps vs proportional 909
    assert pipeline.next_rate("secant", history, TARGET) == pytest.approx(1100 + (aim - 1_000_000) / 1000)


def test_secant_falls_back_to_center_on_an_unusable_slope():
    history = [(1000, 900_000), (1100, 890_000)]     # negative slope
    assert pipeline.next_rate("secant", history, TARGET) == pytest.approx(
        pipeline.next_rate("center", history, TARGET))
    assert pipeline.next_rate("secant", history[-1:], TARGET) == pytest.approx(
        pipeline.next_rate("center", history[-1:], TARGET))


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown feedback policy"):
        pipeline.next_rate("nope", [(1000, 900_000)], TARGET)


def test_default_policy_is_the_original_one():
    assert config.FEEDBACK_POLICY == "proportional"
