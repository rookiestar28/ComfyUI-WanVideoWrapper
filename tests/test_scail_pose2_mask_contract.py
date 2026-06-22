from __future__ import annotations

import importlib.util
import unittest

from scail_pose2_mask_contract import (
    SCAIL_POSE2_CONDITION_MODE_ATTR,
    SCAIL_POSE2_MASK_ROLE_ATTR,
    SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    is_scail_pose2_replacement_noise_mask,
    resize_noise_mask_for_latents,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class ScailPose2MaskContractTests(unittest.TestCase):
    def test_tagged_replacement_mask_uses_conservative_binary_latent_conversion(self) -> None:
        import torch

        mask = torch.tensor([[[0.0, 1.0], [0.49, 0.51]]], dtype=torch.float32)
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 2, 2),
            channel_count=3,
            latent_grow_pixels=0,
        )

        self.assertTrue(is_scail_pose2_replacement_noise_mask(mask))
        self.assertEqual((1, 3, 1, 2, 2), tuple(latent_mask.shape))
        self.assertEqual("conservative_area", contract.interpolation_mode)
        self.assertTrue(contract.scail_pose2_replacement)
        self.assertEqual(0.0, float(latent_mask[0, 0, 0, 0, 0].item()))
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 0, 1].item()))
        self.assertEqual(0.0, float(latent_mask[0, 0, 0, 1, 0].item()))
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 1, 1].item()))
        self.assertAlmostEqual(0.5, contract.subject_ratio)
        self.assertAlmostEqual(0.5, contract.preserve_ratio)
        self.assertEqual(0, contract.latent_grow_pixels)
        self.assertAlmostEqual(0.5, contract.pre_grow_subject_ratio)

    def test_thin_replacement_subject_survives_latent_downsampling(self) -> None:
        import torch

        mask = torch.zeros((1, 8, 8), dtype=torch.float32)
        mask[:, :, 7] = 1.0
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 1, 1),
            channel_count=1,
            latent_grow_pixels=0,
        )

        self.assertEqual("conservative_area", contract.interpolation_mode)
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 0, 0].item()))
        self.assertAlmostEqual(1.0, contract.subject_ratio)
        self.assertAlmostEqual(0.0, contract.preserve_ratio)

    def test_untagged_mask_keeps_trilinear_policy(self) -> None:
        import torch

        mask = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 1, 2),
            channel_count=2,
        )

        self.assertFalse(is_scail_pose2_replacement_noise_mask(mask))
        self.assertEqual((1, 2, 2, 1, 2), tuple(latent_mask.shape))
        self.assertEqual("trilinear", contract.interpolation_mode)
        self.assertFalse(contract.scail_pose2_replacement)

    def test_frame_slice_policy_is_reported(self) -> None:
        import torch

        mask = torch.ones((4, 2, 2), dtype=torch.float32)
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 1, 1),
            channel_count=1,
            start_latent=1,
            end_latent=3,
        )

        self.assertEqual((1, 1, 2, 1, 1), tuple(latent_mask.shape))
        self.assertEqual("slice_1_3", contract.frame_policy)

    def test_context_slice_resizes_full_source_timeline_before_slice(self) -> None:
        import torch

        mask = torch.zeros((8, 2, 2), dtype=torch.float32)
        mask[4:] = 1.0
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 1, 1),
            channel_count=1,
            start_latent=1,
            end_latent=3,
            source_latent_frame_count=4,
        )

        self.assertEqual((1, 1, 2, 1, 1), tuple(latent_mask.shape))
        self.assertEqual("resize_full_4_then_slice_1_3", contract.frame_policy)

    def test_replacement_latent_grow_is_spatial_only(self) -> None:
        import torch

        mask = torch.zeros((2, 5, 5), dtype=torch.float32)
        mask[:, 2, 2] = 1.0
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(2, 5, 5),
            channel_count=1,
            latent_grow_pixels=1,
        )

        self.assertEqual((1, 1, 2, 5, 5), tuple(latent_mask.shape))
        self.assertEqual(1, contract.latent_grow_pixels)
        self.assertAlmostEqual(1.0 / 25.0, contract.pre_grow_subject_ratio)
        self.assertAlmostEqual(9.0 / 25.0, contract.subject_ratio)
        self.assertEqual(9.0, float(latent_mask[0, 0, 0].sum().item()))
        self.assertEqual(9.0, float(latent_mask[0, 0, 1].sum().item()))


if __name__ == "__main__":
    unittest.main()
