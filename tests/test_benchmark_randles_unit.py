"""Offline unit tests for the Randles 2013 benchmark (no uvspec needed)."""

import json

import numpy as np
import pytest
import xarray as xr

from pyradtran.benchmarks import CASES, RandlesAerosol, load_reference, run_randles2013
from pyradtran.benchmarks.randles2013 import (
    AEROSOL_WAVELENGTH_GRID_UM,
    ALTITUDE_GRID_KM,
    BANDS,
    N_LEGENDRE,
    NORMALIZATION_CONSTANTS,
    SZAS,
    build_scene,
    taper_layer_weights,
)

ALT_GRID = np.array(ALTITUDE_GRID_KM)


class TestTaperWeights:
    def test_weights_on_benchmark_grid(self):
        w = taper_layer_weights(ALT_GRID)
        np.testing.assert_allclose(w, [0.0, 0.25, 0.75])

    def test_weights_sum_to_one(self):
        assert taper_layer_weights(ALT_GRID).sum() == pytest.approx(1.0, abs=1e-15)

    def test_taller_grid_same_result(self):
        """Aerosol-free layers above 2 km get zero weight."""
        w = taper_layer_weights(np.array([12.0, 3.0, 2.0, 1.0, 0.0]))
        np.testing.assert_allclose(w, [0.0, 0.0, 0.25, 0.75])


class TestRandlesAerosol:
    def _piece(self, ssa=1.0):
        return RandlesAerosol(ssa=ssa)

    def test_tau_column_at_550(self):
        lo = self._piece().to_layer_optics(np.array([0.55]), ALT_GRID, n_legendre=N_LEGENDRE)
        assert lo.tau.shape == (1, 3)
        assert lo.tau.sum() == pytest.approx(0.2, abs=1e-12)
        assert lo.tau[0, 0] == 0.0  # layer [3,2] km: above the aerosol top
        assert lo.tau[0, 1] == pytest.approx(0.2 * 0.25)  # layer [2,1] km
        assert lo.tau[0, 2] == pytest.approx(0.2 * 0.75)  # layer [1,0] km

    def test_angstrom_exponent_one(self):
        piece = self._piece()
        tau30 = piece.to_layer_optics(np.array([0.30]), ALT_GRID).tau.sum()
        tau55 = piece.to_layer_optics(np.array([0.55]), ALT_GRID).tau.sum()
        assert tau30 / tau55 == pytest.approx(0.55 / 0.30, rel=1e-12)

    def test_ssa_constant_per_case(self):
        wl = np.array([0.3, 0.55, 2.0])
        for ssa in (1.0, 0.8):
            lo = RandlesAerosol(ssa=ssa).to_layer_optics(wl, ALT_GRID)
            assert lo.ssa.shape == (3, 3)
            np.testing.assert_allclose(lo.ssa, ssa)

    def test_case2b_ssa_is_08(self):
        scene = build_scene("case2b", "saw", 30, "bb")
        assert scene.aerosol.pieces[0].ssa == 0.8

    def test_hg_moments(self):
        n_leg = 16
        lo = self._piece().to_layer_optics(np.array([0.55]), ALT_GRID, n_legendre=n_leg)
        assert lo.legendre_moments.shape == (1, 3, n_leg)
        for l in range(8):
            np.testing.assert_allclose(lo.legendre_moments[:, :, l], 0.7**l)

    def test_piece_protocol_duck_typing(self):
        """RandlesAerosol is accepted as a CompositeAerosol piece."""
        from pyradtran.models.aerosol_composite import CompositeAerosol

        comp = CompositeAerosol(
            pieces=[self._piece()],
            wavelength_grid_um=AEROSOL_WAVELENGTH_GRID_UM,
            altitude_grid_km=ALTITUDE_GRID_KM,
        )
        mixed = comp.evaluate(np.array([0.55]), ALT_GRID)
        assert mixed.tau.sum() == pytest.approx(0.2, abs=1e-12)


class TestBuildScene:
    def test_case1_has_no_aerosol(self):
        scene = build_scene("case1", "saw", 30, "bb")
        assert scene.aerosol is None
        text = scene.build_input()
        assert "aerosol" not in text
        assert "atmosphere_file subarctic_winter" in text
        assert "mol_abs_param reptran" in text
        assert "sza 30.0" in text
        assert "albedo 0.2" in text
        assert "rte_solver disort" in text
        assert "number_of_streams 16" in text
        assert "pseudospherical" in text
        assert "disort_intcor moments" in text
        assert "output_process integrate" in text
        assert "output_user lambda edir edn eup" in text
        assert "output_format ascii" in text

    def test_case2a_has_aerosol_piece(self, tmp_path):
        scene = build_scene("case2a", "trop", 75, "uvvis", aerosol_output_dir=tmp_path)
        assert scene.aerosol is not None
        assert isinstance(scene.aerosol.pieces[0], RandlesAerosol)
        assert scene.aerosol.pieces[0].ssa == 1.0
        assert scene.aerosol.altitude_grid_km == ALTITUDE_GRID_KM
        text = scene.build_input()
        assert "wavelength 200.0 700.0" in text
        assert "aerosol_file explicit" in text
        master = [ln for ln in text.splitlines() if ln.startswith("aerosol_file explicit")][0]
        assert tmp_path.name in master

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError, match="case"):
            build_scene("case9", "saw", 30, "bb")
        with pytest.raises(ValueError, match="atmosphere"):
            build_scene("case1", "mlw", 30, "bb")
        with pytest.raises(ValueError, match="sza"):
            build_scene("case1", "saw", 60, "bb")
        with pytest.raises(ValueError, match="band"):
            build_scene("case1", "saw", 30, "swir")


def _synthetic_dataset(edir_toa=1000.0, edir_sfc=800.0, edn_sfc=100.0, eup_sfc=50.0, eup_toa=200.0):
    """Mimic parsed ``process=integrate`` output: dims (wavelength: 1, zout: 2)."""
    return xr.Dataset(
        {
            "edir": (("wavelength", "zout"), [[edir_sfc, edir_toa]]),
            "edn": (("wavelength", "zout"), [[edn_sfc, 0.0]]),
            "eup": (("wavelength", "zout"), [[eup_sfc, eup_toa]]),
        },
        coords={"wavelength": [200.0], "zout": [0.0, 120.0]},
    )


class TestRunRandles2013Normalization:
    """End-to-end runner logic with uvspec monkeypatched out."""

    @pytest.fixture()
    def run_with_stub(self, tmp_path, monkeypatch):
        def _run(cases=None):
            def fake_execute_many(scenes, **kwargs):
                expected = 24 if cases is None else len(cases) * 8
                assert len(scenes) == expected
                return [_synthetic_dataset() for _ in scenes]

            monkeypatch.setattr(
                "pyradtran.core.runner.Runner.execute_many", staticmethod(fake_execute_many)
            )
            return run_randles2013(tmp_path, cases=cases)

        return _run

    def test_normalized_flux_values(self, run_with_stub):
        results = run_with_stub()
        c = NORMALIZATION_CONSTANTS["bb"][30]
        entry = results["case1"]["bb"]["saw"][30]
        assert entry["edir_sfc"] == pytest.approx(800.0 / 1000.0 * c)
        assert entry["edn_sfc"] == pytest.approx(100.0 / 1000.0 * c)
        assert entry["eup_sfc"] == pytest.approx(50.0 / 1000.0 * c)
        assert entry["eup_toa"] == pytest.approx(200.0 / 1000.0 * c)
        assert entry["total_sfc_down"] == pytest.approx(900.0 / 1000.0 * c)

    def test_structure_and_derived(self, run_with_stub):
        results = run_with_stub()
        for case in CASES:
            for band in BANDS:
                for atm in ("saw", "trop"):
                    assert set(results[case][band][atm]) == set(SZAS)
        # Identical stubbed fluxes -> zero RF and absorptance = 1 - (200+0.8*900)/1000
        entry = results["case2a"]["bb"]["trop"][75]
        assert entry["rf_toa"] == pytest.approx(0.0, abs=1e-12)
        assert entry["rf_sfc"] == pytest.approx(0.0, abs=1e-12)
        c1 = results["case1"]["bb"]["saw"][30]
        assert c1["absorptance"] == pytest.approx(1.0 - (200.0 + 0.8 * 900.0) / 1000.0)
        # bb and uvvis both stubbed the same -> NIR difference is C_bb - C_uvvis
        assert c1["nir_sfc_down"] == pytest.approx(
            (NORMALIZATION_CONSTANTS["bb"][30] - NORMALIZATION_CONSTANTS["uvvis"][30]) * 0.9
        )

    def test_results_json_saved(self, run_with_stub, tmp_path):
        results = run_with_stub()
        path = tmp_path / "randles2013_results.json"
        assert results["results_path"] == str(path)
        saved = json.loads(path.read_text())
        assert (
            saved["case1"]["bb"]["saw"]["30"]["edir_sfc"]
            == results["case1"]["bb"]["saw"][30]["edir_sfc"]
        )
        assert saved["case2b"]["uvvis"]["trop"]["75"]["rf_toa"] == pytest.approx(0.0, abs=1e-12)

    def test_case_subset_without_case1_omits_rf(self, run_with_stub, tmp_path):
        results = run_with_stub(cases=["case2a"])
        assert "case1" not in results
        assert "rf_toa" not in results["case2a"]["bb"]["saw"][30]
        assert "rf_sfc" not in results["case2a"]["uvvis"]["trop"][75]

    def test_unknown_case_raises(self, tmp_path):
        with pytest.raises(ValueError, match="case9"):
            run_randles2013(tmp_path, cases=["case9"])


class TestReferenceFile:
    def test_loads_with_exact_structure(self):
        ref = load_reference()
        assert set(ref) == {"_meta", "case1", "case2a", "case2b"}
        assert set(ref["case1"]) == {
            "direct_bb_sfc_down",
            "diffuse_bb_sfc_down",
            "diffuse_bb_toa_up",
            "uvvis_sfc_down",
            "nir_sfc_down",
            "absorptance",
        }
        for case in ("case2a", "case2b"):
            assert set(ref[case]) == {
                "total_bb_sfc_down",
                "diffuse_bb_toa_up",
                "uvvis_sfc_down",
                "toa_rf",
                "sfc_rf",
            }

    def test_all_values_finite_floats_and_count(self):
        ref = load_reference()
        n = 0
        for case in ("case1", "case2a", "case2b"):
            for quantity, configs in ref[case].items():
                assert set(configs) == {"saw30", "saw75", "trop30", "trop75"}, quantity
                for value in configs.values():
                    assert isinstance(value, float), f"{case}/{quantity}: {value!r}"
                    assert np.isfinite(value)
                    n += 1
        assert n == 64  # 6 quantities x 4 configs (case1) + 2 x (5 x 4)

    def test_anchor_values_match_locked_facts(self):
        ref = load_reference()
        c1 = ref["case1"]
        assert c1["direct_bb_sfc_down"] == {
            "saw30": 942.4,
            "saw75": 216.2,
            "trop30": 844.5,
            "trop75": 179.6,
        }
        assert c1["absorptance"]["trop75"] == 0.307
        assert c1["nir_sfc_down"]["saw30"] == 519.1
        c2a = ref["case2a"]
        assert c2a["toa_rf"]["saw30"] == -8.6
        assert c2a["sfc_rf"]["trop75"] == -18.9
        c2b = ref["case2b"]
        assert c2b["toa_rf"]["trop30"] == 10.3
        assert c2b["sfc_rf"]["saw30"] == -42.1

    def test_meta_documentation(self):
        ref = load_reference()
        meta = ref["_meta"]
        assert "Randles" in meta["paper"]
        consts = meta["normalization_constants_w_m2"]
        assert consts["bb"] == {"sza30": 1189.28, "sza75": 355.43}
        assert consts["uvvis"] == {"sza30": 563.38, "sza75": 168.37}
        assert meta["thresholds"]["flux"] == {"pass_pct": 8, "warn_pct": 12}
        assert meta["thresholds"]["rf"] == {"pass_rel_pct": 15, "pass_abs_w_m2": 1.5}

    def test_reference_file_exists_for_wheel(self):
        from pyradtran.benchmarks.randles2013 import _REFERENCE_PATH

        assert _REFERENCE_PATH.is_file()
