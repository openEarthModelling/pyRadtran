"""Tests for 3D configuration model."""

import pytest
from pyradtran.models.three_d import ThreeDConfig


class TestThreeDConfig:
    def test_atmosphere_file_3d(self):
        c = ThreeDConfig(atmosphere_file="/data/atm3d.nc")
        lines = c.to_uvspec_lines()
        assert "atmosphere_file_3D /data/atm3d.nc" in lines

    def test_cloud_file_3d(self):
        c = ThreeDConfig(cloud_file="/data/cloud3d.nc")
        lines = c.to_uvspec_lines()
        assert "cloud_file_3D /data/cloud3d.nc" in lines

    def test_output_3d(self):
        c = ThreeDConfig(output_3d=True)
        lines = c.to_uvspec_lines()
        assert "output3D" in lines

    def test_processing_3d(self):
        c = ThreeDConfig(processing_3d=True)
        lines = c.to_uvspec_lines()
        assert "processing3D" in lines

    def test_ipa_3d(self):
        c = ThreeDConfig(ipa_3d=True)
        lines = c.to_uvspec_lines()
        assert "ipa_3d" in lines

    def test_all_fields(self):
        c = ThreeDConfig(
            atmosphere_file="/data/atm3d.nc",
            cloud_file="/data/cloud3d.nc",
            output_3d=True,
            processing_3d=True,
        )
        lines = c.to_uvspec_lines()
        assert len(lines) == 4

    def test_empty_config(self):
        c = ThreeDConfig()
        lines = c.to_uvspec_lines()
        assert lines == []

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            ThreeDConfig(nonexistent=1)