from __future__ import annotations

import csv
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.evidence_media import (  # noqa: E402
    materialize_media,
    render_fit_process_card,
    render_trajectory_diagnostics,
    write_placeholder_image,
    write_video_montage,
)


class EvidenceMediaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _assert_png(self, path: Path, expected_size: tuple[int, int] | None = None) -> None:
        payload = path.read_bytes()
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", payload[16:24])
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        if expected_size is not None:
            self.assertEqual((width, height), expected_size)

    def test_materialize_hardlink_and_copy_fallback_are_explicit(self) -> None:
        source = self.root / "source.mp4"
        source.write_bytes(b"benchmark-media")
        hardlink = self.root / "report" / "hardlink.mp4"
        result = materialize_media(source, hardlink)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(hardlink.read_bytes(), source.read_bytes())
        self.assertIn(result["mode_used"], {"hardlink", "copy_fallback_from_hardlink"})

        copied = self.root / "report" / "fallback.mp4"
        with mock.patch("os.link", side_effect=OSError("cross-device")):
            fallback = materialize_media(source, copied, mode="hardlink")
        self.assertEqual(fallback["status"], "ok")
        self.assertEqual(fallback["mode_used"], "copy_fallback_from_hardlink")
        self.assertEqual(copied.read_bytes(), source.read_bytes())

        missing = materialize_media(self.root / "missing.mp4", self.root / "x.mp4")
        self.assertEqual(missing["status"], "unavailable")
        self.assertEqual(missing["reason_code"], "source_media_missing")

    def test_placeholder_is_valid_dependency_free_png(self) -> None:
        output = self.root / "placeholder.png"
        result = write_placeholder_image(
            output,
            "Tracking unavailable",
            "Valid fraction 0.31 is below the required 0.50.",
            width=640,
            height=360,
        )
        self.assertEqual(result["status"], "ok")
        self._assert_png(output, (640, 360))

    def test_trajectory_diagnostics_render_all_coordinate_panels(self) -> None:
        trajectory = self.root / "trajectory.csv"
        with trajectory.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frame_index",
                    "time_s",
                    "center_u_px",
                    "center_v_px",
                    "fit_x_m",
                    "fit_y_m",
                    "fit_z_m",
                    "interpolated",
                    "fit_eligible",
                ],
            )
            writer.writeheader()
            for index in range(12):
                writer.writerow(
                    {
                        "frame_index": index,
                        "time_s": index / 16,
                        "center_u_px": 100 + index * 3,
                        "center_v_px": 200 - index * 2,
                        "fit_x_m": index * 0.1,
                        "fit_y_m": index * 0.01,
                        "fit_z_m": 1.0 - index * 0.03,
                        "interpolated": index == 5,
                        "fit_eligible": index != 8,
                    }
                )
        output = self.root / "trajectory_diagnostics.png"
        result = render_trajectory_diagnostics(
            trajectory,
            output,
            title="v1_A evidence",
            route="qualified_dynamic_3d",
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(all(result["panels"].values()))
        self.assertEqual(result["world_xyz_point_count"], 12)
        self.assertEqual(result["direct_point_count"], 10)
        self.assertEqual(result["interpolated_point_count"], 1)
        self.assertEqual(result["invalid_point_count"], 1)
        self._assert_png(output, (1600, 1000))

    def test_trajectory_diagnostics_mark_missing_depth_without_inventing_it(self) -> None:
        trajectory = self.root / "trajectory_2d.csv"
        with trajectory.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["frame_index", "center_u_px", "center_v_px", "x_m", "z_m"],
            )
            writer.writeheader()
            for index in range(5):
                writer.writerow(
                    {
                        "frame_index": index,
                        "center_u_px": 40 + index,
                        "center_v_px": 70 - index,
                        "x_m": index * 0.2,
                        "z_m": 1.0 - index * 0.1,
                    }
                )
        output = self.root / "trajectory_2d.png"
        result = render_trajectory_diagnostics(trajectory, output, route="calibrated_2d")
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["panels"]["world_xz"])
        self.assertFalse(result["panels"]["orthographic_xy"])
        self.assertFalse(result["panels"]["orthographic_yz"])
        self.assertEqual(result["world_xyz_point_count"], 0)
        self._assert_png(output)

        missing_output = self.root / "missing_trajectory.png"
        missing = render_trajectory_diagnostics(
            self.root / "does_not_exist.csv",
            missing_output,
        )
        self.assertEqual(missing["status"], "unavailable")
        self.assertEqual(missing["reason_code"], "trajectory_csv_missing")
        self._assert_png(missing_output)

    def test_fit_card_uses_serialized_series_and_does_not_infer_missing_trace(self) -> None:
        result_path = self.root / "result.json"
        result_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "parameters": {
                            "gravity_g": {
                                "gt": 9.81,
                                "estimate_raw": 9.72,
                                "normalized_absolute_error": 0.0071,
                            }
                        }
                    },
                    "fit": {
                        "status": "ok",
                        "method": "robust_airborne_quadratic",
                        "target_not_used_for_fit": True,
                        "parameter_estimates": {"gravity_g": 9.72},
                        "diagnostics": {
                            "fit_points": 3,
                            "estimated_acceleration_m_s2": -9.72,
                            "fit_series": {
                                "series_name": "z_m",
                                "time_s": [0.0, 0.1, 0.2],
                                "observed": [1.0, 0.95, 0.81],
                                "predicted": [1.0, 0.951, 0.808],
                                "residual": [0.0, -0.001, 0.002],
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "fit_process.html"
        card = render_fit_process_card(
            result_path,
            output,
            {
                "title": "Free-fall gravity recovery",
                "equation": "z(tau)=c0+c1*tau+c2*tau^2; g_hat=-2*c2",
                "observables": "z_m versus time_s",
            },
            grade="L4",
        )
        text = output.read_text(encoding="utf-8")
        self.assertEqual(card["status"], "ok")
        self.assertEqual(card["series_count"], 1)
        self.assertIn("robust_airborne_quadratic", text)
        self.assertIn("9.72", text)
        self.assertIn("0.951", text)
        self.assertIn("Target used during fit:</b> no", text)
        self.assertIn("g_hat=-2*c2", text)
        self.assertIn("9.81", text)
        self.assertIn("0.0071", text)
        self.assertIn("Minimum fit points: 3", text)
        self.assertIn("estimated_acceleration_m_s2", text)

        no_trace = self.root / "fit_no_trace.md"
        partial = render_fit_process_card(
            {
                "fit": {
                    "status": "ok",
                    "method": "recorded_method",
                    "parameter_estimates": {"restitution_e": 0.6},
                }
            },
            no_trace,
            {"equation": "e=abs(v_after/v_before)"},
        )
        markdown = no_trace.read_text(encoding="utf-8")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["series_count"], 0)
        self.assertIn("No serialized fit_series", markdown)
        self.assertNotIn("coefficient", markdown.lower())

    def test_video_montage_has_structured_invalid_input_result(self) -> None:
        result = write_video_montage([], [], self.root / "empty.mp4")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason_code"], "no_video_sources")
        self._assert_png(Path(result["placeholder_image"]))

    @unittest.skipUnless(
        importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None,
        "OpenCV montage test is optional in the dependency-free local environment",
    )
    def test_video_montage_is_reopenable_when_opencv_is_available(self) -> None:
        import cv2
        import numpy as np

        sources = []
        for source_index in range(2):
            path = self.root / f"source_{source_index}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 64))
            if not writer.isOpened():
                self.skipTest("OpenCV MP4V writer is unavailable")
            for frame_index in range(12):
                frame = np.full((64, 96, 3), 30 + source_index * 80, dtype=np.uint8)
                cv2.circle(frame, (10 + frame_index * 5, 32), 5, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            sources.append(path)
        result = write_video_montage(
            sources,
            ["target=2 recovered=2.1", "target=9.8 recovered=8.7"],
            self.root / "montage.mp4",
            fps=8,
            tile_width=160,
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertGreater(result["frame_count"], 0)
        self.assertEqual(result["time_alignment"], "timestamp_sampled_no_time_warp_shortest_duration")
        self.assertTrue(Path(result["output"]).is_file())


if __name__ == "__main__":
    unittest.main()
