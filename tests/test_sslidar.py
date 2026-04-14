"""Tests for SSLidar configuration model."""

import pytest
from pyradtran.models.sslidar import SslidarConfig


class TestSslidarConfig:
    def test_area(self):
        s = SslidarConfig(area=2.0)
        lines = s.to_uvspec_lines()
        assert "sslidar area 2.0" in lines

    def test_all_params(self):
        s = SslidarConfig(area=1.5, E0=0.2, efficiency=0.8, position=0.5, range_bin=0.2)
        lines = s.to_uvspec_lines()
        assert "sslidar area 1.5" in lines
        assert "sslidar E0 0.2" in lines
        assert "sslidar eff 0.8" in lines
        assert "sslidar position 0.5" in lines
        assert "sslidar range 0.2" in lines

    def test_nranges(self):
        s = SslidarConfig(area=1.0, n_ranges=200)
        lines = s.to_uvspec_lines()
        assert "sslidar_nranges 200" in lines

    def test_polarisation(self):
        s = SslidarConfig(area=1.0, polarisation=True)
        lines = s.to_uvspec_lines()
        assert "sslidar_polarisation" in lines

    def test_area_valid_range(self):
        with pytest.raises(Exception):
            SslidarConfig(area=-1.0)

    def test_empty_sslidar(self):
        s = SslidarConfig()
        lines = s.to_uvspec_lines()
        assert lines == []

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            SslidarConfig(invalid=1)
