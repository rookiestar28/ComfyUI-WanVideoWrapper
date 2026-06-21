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
    def test_tagged_replacement_mask_uses_nearest_binary_latent_conversion(self) -> None:
        import torch

        mask = torch.tensor([[[0.0, 1.0], [0.49, 0.51]]], dtype=torch.float32)
        setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
        setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)

        latent_mask, contract = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 2, 2),
            channel_count=3,
        )

        self.assertTrue(is_scail_pose2_replacement_noise_mask(mask))
        self.assertEqual((1, 3, 1, 2, 2), tuple(latent_mask.shape))
        self.assertEqual("nearest", contract.interpolation_mode)
        self.assertTrue(contract.scail_pose2_replacement)
        self.assertEqual(0.0, float(latent_mask[0, 0, 0, 0, 0].item()))
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 0, 1].item()))
        self.assertEqual(0.0, float(latent_mask[0, 0, 0, 1, 0].item()))
        self.assertEqual(1.0, float(latent_mask[0, 0, 0, 1, 1].item()))
        self.assertAlmostEqual(0.5, contract.subject_ratio)
        self.assertAlmostEqual(0.5, contract.preserve_ratio)

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


if __name__ == "__main__":
    unittest.main()
