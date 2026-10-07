"""Guards on the constants every other module depends on."""

import rul
from rul import config


def test_package_imports():
    assert rul.__version__


def test_thirteen_retained_sensors():
    assert len(config.SENSORS) == 13
    assert len(set(config.SENSORS)) == 13
    assert "s14" not in config.SENSORS  # redundant with s9


def test_feature_layout_is_sensor_major():
    assert len(config.FEATURE_NAMES) == 65
    # The notebooks build features sensor by sensor: s2_mean ... s2_delta, s3_mean ...
    assert config.FEATURE_NAMES[:5] == [
        "s2_mean",
        "s2_std",
        "s2_slope",
        "s2_last",
        "s2_delta",
    ]
    assert config.FEATURE_NAMES[-1] == "s21_delta"


def test_target_constants():
    assert config.R_MAX == 125
    assert config.MAX_HISTORY == 362
