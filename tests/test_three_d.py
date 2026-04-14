"""Tests for 3D configuration model."""

import pytest

from pyradtran.models.three_d import ThreeDConfig


class TestThreeDConfig:
    def test_atmosphere_file(self):
        c = ThreeDConfig(atmosphere_file="/data/atm3d.nc")
        lines = c.to_uvspec_lines()
        assert "atmosphere_file /data/atm3d.nc" in lines

    def test_empty_config(self):
        c = ThreeDConfig()
        lines = c.to_uvspec_lines()
        assert lines == []

    def test_extra_field_forbidden(self):
        with pytest.raises(ValueError):
            ThreeDConfig(nonexistent=1)
