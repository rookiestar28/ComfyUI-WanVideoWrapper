from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_forward_module():
    spec = importlib.util.spec_from_file_location(
        "scail2_forward_under_test",
        ROOT / "SCAIL" / "scail2_forward.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def plan(scail2_input, video_shape=(16, 2, 4, 4)):
    module = import_forward_module()
    return module.build_scail2_forward_plan(
        scail2_input,
        video_shape=video_shape,
        patch_size=(1, 2, 2),
    )


class SCAIL2ForwardPlanTests(unittest.TestCase):
    def test_mask_only_control_is_not_dropped(self) -> None:
        result = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "driving_masks": [FakeTensor((28, 2, 2, 2))],
            }
        )

        self.assertFalse(result["has_pose"])
        self.assertTrue(result["has_control_mask"])
        self.assertGreater(result["control_length"], 0)
        self.assertEqual(result["main_length"] + result["control_length"], result["total_length"])

    def test_pose_only_control_is_planned_without_mask_control(self) -> None:
        result = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
            }
        )

        self.assertTrue(result["has_pose"])
        self.assertFalse(result["has_control_mask"])
        self.assertEqual((16, 2, 2, 2), result["control_shape"])
        self.assertGreater(result["control_length"], 0)

    def test_pose_and_mask_share_control_shape(self) -> None:
        result = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
                "driving_masks": [FakeTensor((28, 2, 2, 2))],
            }
        )

        self.assertTrue(result["has_pose"])
        self.assertTrue(result["has_control_mask"])
        self.assertEqual((16, 2, 2, 2), result["control_shape"])

        with self.assertRaisesRegex(ValueError, "share temporal/spatial"):
            plan(
                {
                    "pose_latents": [FakeTensor((16, 2, 2, 2))],
                    "driving_masks": [FakeTensor((28, 3, 2, 2))],
                }
            )

    def test_replace_flag_changes_rope_shifts_and_cache_key(self) -> None:
        animation = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
                "replace_flag": False,
            }
        )
        replacement = plan(
            {
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "pose_latents": [FakeTensor((16, 2, 2, 2))],
                "replace_flag": True,
            }
        )

        self.assertNotEqual(animation["cache_key"], replacement["cache_key"])
        self.assertEqual(1, animation["rope_shifts"]["t"]["video"])
        self.assertEqual(0, replacement["rope_shifts"]["t"]["video"])
        self.assertEqual(0.0, animation["rope_shifts"]["h"]["ref"])
        self.assertEqual(120.0, replacement["rope_shifts"]["h"]["ref"])

    def test_additional_references_affect_order_and_lengths(self) -> None:
        result = plan(
            {
                "additional_ref_latents": [
                    FakeTensor((16, 1, 4, 4)),
                    FakeTensor((16, 1, 4, 4)),
                ],
                "additional_ref_masks": [
                    FakeTensor((28, 1, 4, 4)),
                    FakeTensor((28, 1, 4, 4)),
                ],
                "ref_latents": [FakeTensor((16, 1, 4, 4))],
                "ref_masks": [FakeTensor((28, 1, 4, 4))],
                "driving_masks": [FakeTensor((28, 2, 2, 2))],
            }
        )

        self.assertEqual(3, result["prefix_frames"])
        self.assertEqual(2, result["additional_ref_count"])
        self.assertGreater(result["additional_ref_length"], 0)
        self.assertEqual(2, result["rope_shifts"]["t"]["ref"])
        self.assertEqual(3, result["rope_shifts"]["t"]["video"])

    def test_additional_reference_masks_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "additional_ref_masks is required"):
            plan({"additional_ref_latents": [FakeTensor((16, 1, 4, 4))]})

        with self.assertRaisesRegex(ValueError, "additional_ref_masks requires"):
            plan({"additional_ref_masks": [FakeTensor((28, 1, 4, 4))]})

    def test_model_keeps_v1_and_v2_forward_branches_distinct(self) -> None:
        model_source = (ROOT / "wanvideo" / "modules" / "model.py").read_text(encoding="utf-8")

        self.assertIn("scail_input=None,  # SCAIL pose", model_source)
        self.assertIn("scail2_input=None,  # SCAIL-2 native pose/mask conditioning", model_source)
        self.assertIn("# SCAIL pose", model_source)
        self.assertIn("if scail_input is not None:", model_source)
        self.assertIn("if scail2_input is not None:", model_source)


if __name__ == "__main__":
    unittest.main()
