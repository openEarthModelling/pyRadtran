"""Tests for temporary file lifecycle management."""

import os

import numpy as np

from pyradtran.core.tempfile_manager import TempFileManager


class TestTempFileManager:
    def test_creates_and_returns_path(self, tmp_path):
        mgr = TempFileManager(temp_dir=str(tmp_path))
        path = mgr.write_array("extinction.dat", np.array([1.0, 2.0, 3.0]))
        assert os.path.exists(path)
        data = np.loadtxt(path)
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0])

    def test_creates_unique_files(self, tmp_path):
        mgr = TempFileManager(temp_dir=str(tmp_path))
        p1 = mgr.write_array("extinction.dat", np.array([1.0]))
        p2 = mgr.write_array("extinction.dat", np.array([2.0]))
        assert p1 != p2
        assert os.path.exists(p1)
        assert os.path.exists(p2)

    def test_cleanup_removes_files(self, tmp_path):
        mgr = TempFileManager(temp_dir=str(tmp_path))
        p1 = mgr.write_array("test.dat", np.array([1.0]))
        p2 = mgr.write_array("test.dat", np.array([2.0]))
        assert os.path.exists(p1)
        mgr.cleanup()
        assert not os.path.exists(p1)
        assert not os.path.exists(p2)

    def test_keep_temp_preserves_files(self, tmp_path):
        mgr = TempFileManager(temp_dir=str(tmp_path), keep_temp=True)
        p1 = mgr.write_array("test.dat", np.array([1.0]))
        mgr.cleanup()
        assert os.path.exists(p1)

    def test_context_manager(self, tmp_path):
        with TempFileManager(temp_dir=str(tmp_path)) as mgr:
            p1 = mgr.write_array("test.dat", np.array([1.0]))
            assert os.path.exists(p1)
        assert not os.path.exists(p1)

    def test_write_text(self, tmp_path):
        mgr = TempFileManager(temp_dir=str(tmp_path))
        path = mgr.write_text("config.txt", "hello\nworld\n")
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "hello\nworld\n"

    def test_tracked_files(self, tmp_path):
        mgr = TempFileManager(temp_dir=str(tmp_path))
        p1 = mgr.write_array("a.dat", np.array([1.0]))
        p2 = mgr.write_array("b.dat", np.array([2.0]))
        assert len(mgr.files) == 2
        assert p1 in mgr.files
        assert p2 in mgr.files
