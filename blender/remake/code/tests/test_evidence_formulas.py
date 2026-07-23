from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.evidence_formulas import (  # noqa: E402
    EXPERIMENT_ORDER,
    FORMULA_REGISTRY,
    benchmark_scope_summary,
    experiment_system_count,
    extract_fit_quality_records,
    extract_readable_intermediates,
    get_formula_spec,
    parameter_channel_manifest,
    scan_axis_count,
    summarize_fit_diagnostics,
)


EXPECTED_PARAMETERS = {
    "v1_A": {"gravity_g"},
    "v1_B": {"restitution_e"},
    "v1_C": {"kinetic_friction_mu"},
    "v1_D": {"amplitude_decay_beta"},
    "v2_A": {"gravity_g"},
    "v2_B": {"restitution_e"},
    "v2_C": {"kinetic_friction_mu_A", "kinetic_friction_mu_B"},
    "v2_D": {"gravity_g", "linear_damping_beta"},
    "v2_E": {"gravity_g", "restitution_e"},
    "v3_A": {"gravity_g", "linear_drag_beta", "restitution_e"},
    "v3_B": {"kinetic_friction_mu_k", "left_restitution_e_L", "right_restitution_e_R"},
    "v3_C": {"gravity_g", "kinetic_friction_mu", "restitution_e"},
    "v3_D": {"gravity_g", "linear_damping_beta", "magnetic_kappa"},
}


class EvidenceFormulaRegistryTests(unittest.TestCase):
    def test_registry_has_13_experiments_and_24_scan_axes(self) -> None:
        self.assertEqual(set(FORMULA_REGISTRY), set(EXPECTED_PARAMETERS))
        self.assertEqual(len(FORMULA_REGISTRY), 13)
        self.assertEqual(experiment_system_count(), 13)
        self.assertEqual(scan_axis_count(), 24)
        for experiment_id, expected in EXPECTED_PARAMETERS.items():
            self.assertEqual(set(FORMULA_REGISTRY[experiment_id]["parameters"]), expected)

    def test_24_channels_are_not_misreported_as_24_experiments(self) -> None:
        manifest = parameter_channel_manifest()
        self.assertEqual(len(EXPERIMENT_ORDER), 13)
        self.assertEqual(len(manifest), 24)
        self.assertEqual(len({row["channel_id"] for row in manifest}), 24)
        self.assertEqual({row["experiment_id"] for row in manifest}, set(EXPERIMENT_ORDER))
        counts = {
            experiment_id: sum(row["experiment_id"] == experiment_id for row in manifest)
            for experiment_id in EXPERIMENT_ORDER
        }
        self.assertEqual(
            counts,
            {
                "v1_A": 1,
                "v1_B": 1,
                "v1_C": 1,
                "v1_D": 1,
                "v2_A": 1,
                "v2_B": 1,
                "v2_C": 2,
                "v2_D": 2,
                "v2_E": 2,
                "v3_A": 3,
                "v3_B": 3,
                "v3_C": 3,
                "v3_D": 3,
            },
        )
        scope = benchmark_scope_summary()
        self.assertEqual(scope["experiment_system_count"], 13)
        self.assertEqual(scope["parameter_response_channel_count"], 24)
        self.assertIn("不是", scope["counting_rule_zh"])

    def test_specs_have_titles_observables_steps_and_real_fitter_method(self) -> None:
        for experiment_id, parameters in EXPECTED_PARAMETERS.items():
            for parameter_name in parameters:
                spec = get_formula_spec(experiment_id, parameter_name)
                self.assertEqual(spec["experiment_id"], experiment_id)
                self.assertEqual(spec["parameter_name"], parameter_name)
                self.assertTrue(spec["experiment_title_zh"])
                self.assertTrue(spec["experiment_title_en"])
                self.assertTrue(spec["parameter_title_zh"])
                self.assertTrue(spec["parameter_title_en"])
                self.assertIn("time_s", spec["observables"])
                self.assertTrue(spec["model_plain"])
                self.assertTrue(spec["model_latex"])
                self.assertTrue(spec["fitter_function"].startswith("_fit_v"))
                self.assertTrue(spec["fitter_method"])
                self.assertGreaterEqual(len(spec["estimate_steps_zh"]), 3)

    def test_key_formulas_match_numerical_fitter_definitions(self) -> None:
        self.assertEqual(get_formula_spec("v1_A", "gravity_g")["estimate_plain"], "g_hat = -2*c2")
        self.assertIn("abs(v_post/v_pre)", get_formula_spec("v1_B", "restitution_e")["estimate_plain"])
        self.assertEqual(
            get_formula_spec("v2_A", "gravity_g")["estimate_plain"],
            "g_hat = 4*a*omega_hat^2",
        )
        self.assertIn("a_floor/g_hat", get_formula_spec("v3_C", "kinetic_friction_mu")["estimate_plain"])
        self.assertIn("kappa*h(theta)", get_formula_spec("v3_D", "magnetic_kappa")["model_plain"])

    def test_updated_v1_segment_methods_are_described_exactly(self) -> None:
        v1a = get_formula_spec("v1_A", "gravity_g")
        self.assertEqual(v1a["fitter_method"], "first_motion_to_first_contact_quadratic")
        self.assertIn("首次持续向下运动", v1a["fitter_summary_zh"])
        self.assertIn("首次接触", "".join(v1a["estimate_steps_zh"]))

        v1b = get_formula_spec("v1_B", "restitution_e")
        self.assertEqual(
            v1b["fitter_method"],
            "single_impact_segmented_velocity_ratio_with_consistency_gate",
        )
        self.assertIn("保护间隔", "".join(v1b["estimate_steps_zh"]))
        self.assertIn("局部/全局估计一致性", "".join(v1b["estimate_steps_zh"]))

    def test_v2_system_titles_cannot_regress_to_projectile_or_oblique_labels(self) -> None:
        self.assertEqual(FORMULA_REGISTRY["v2_A"]["experiment_title_en"], "Cycloid-track periodic motion")
        self.assertEqual(FORMULA_REGISTRY["v2_B"]["experiment_title_en"], "Repeated two-wall impacts")
        self.assertEqual(FORMULA_REGISTRY["v2_D"]["experiment_title_en"], "Gravity-damping pendulum")
        for experiment_id in ("v2_A", "v2_B", "v2_D"):
            title = FORMULA_REGISTRY[experiment_id]["experiment_title_en"].lower()
            self.assertNotIn("projectile", title)
            self.assertNotIn("oblique", title)

    def test_casebook_renderer_uses_registry_titles_and_explains_13_vs_24(self) -> None:
        source = (CODE_ROOT / "scripts" / "render_seedance_parameter_casebook.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'FORMULA_REGISTRY[experiment_id]["experiment_title_en"]',
            source,
        )
        self.assertIn("def _scope_table_lines()", source)
        self.assertIn("def _formula_appendix_lines()", source)
        self.assertIn("13 个实验系统，不是 24 个实验", source)
        self.assertNotIn('"v2_A": "projectile', source.lower())
        self.assertNotIn('"v2_B": "oblique', source.lower())
        self.assertNotIn('"v2_D": "projectile', source.lower())

    def test_report_method_names_exist_in_numerical_fitter_source(self) -> None:
        source = (CODE_ROOT / "src" / "remake_benchmark" / "reconstruction" / "physics_parameters.py").read_text(
            encoding="utf-8"
        )
        for experiment_id, experiment in FORMULA_REGISTRY.items():
            with self.subTest(experiment_id=experiment_id):
                self.assertIn(f'def {experiment["fitter_function"]}(', source)
                self.assertIn(f'method="{experiment["fitter_method"]}"', source)

    def test_get_formula_spec_returns_a_deep_copy_and_rejects_unknown_axes(self) -> None:
        first = get_formula_spec("v2_E", "gravity_g")
        first["observables"].append("bad")
        first["estimate_steps_zh"].append("bad")
        second = get_formula_spec("v2_E", "gravity_g")
        self.assertNotIn("bad", second["observables"])
        self.assertNotIn("bad", second["estimate_steps_zh"])
        with self.assertRaises(KeyError):
            get_formula_spec("v9_Z", "gravity_g")
        with self.assertRaises(KeyError):
            get_formula_spec("v1_A", "restitution_e")


class FitDiagnosticExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diagnostics = {
            "fit_points": 31,
            "fit_nrmse": 0.08,
            "estimated_period_s": 2.7,
            "nested": {
                "fit_points": 12,
                "fit_nrmse": 0.22,
                "impact_count": 3,
                "event": {"v_pre": -2.0, "v_post": 1.4, "velocity_ratio": 0.7},
            },
            "fit_series": {
                "fit_points": 999,
                "fit_nrmse": 999.0,
                "observed": [1.0, 2.0, 3.0],
            },
        }

    def test_quality_extraction_is_recursive_but_skips_fit_series_vectors(self) -> None:
        original = copy.deepcopy(self.diagnostics)
        records = extract_fit_quality_records(self.diagnostics)
        self.assertEqual(
            records,
            [
                {"path": "diagnostics", "fit_nrmse": 0.08, "fit_points": 31},
                {"path": "diagnostics.nested", "fit_nrmse": 0.22, "fit_points": 12},
            ],
        )
        self.assertEqual(self.diagnostics, original)

    def test_intermediate_extraction_is_bounded_and_readable(self) -> None:
        rows = extract_readable_intermediates(self.diagnostics, limit=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["name"] for row in rows], ["estimated_period_s", "impact_count", "v_pre"])
        self.assertEqual(extract_readable_intermediates(self.diagnostics, limit=0), [])
        with self.assertRaises(ValueError):
            extract_readable_intermediates(self.diagnostics, limit=-1)

    def test_summary_reports_worst_nrmse_and_minimum_points(self) -> None:
        summary = summarize_fit_diagnostics(self.diagnostics, intermediate_limit=2)
        self.assertEqual(summary["quality_record_count"], 2)
        self.assertAlmostEqual(summary["worst_fit_nrmse"], 0.22)
        self.assertEqual(summary["minimum_fit_points"], 12)
        self.assertEqual(len(summary["intermediates"]), 2)


if __name__ == "__main__":
    unittest.main()
