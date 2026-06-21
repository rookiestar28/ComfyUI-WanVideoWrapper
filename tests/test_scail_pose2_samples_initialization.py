from __future__ import annotations

import importlib.util
import unittest

from scail_pose2_mask_contract import (
    SCAIL_POSE2_CONDITION_MODE_ATTR,
    SCAIL_POSE2_MASK_ROLE_ATTR,
    SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE,
    apply_samples_to_noise,
    resize_noise_mask_for_latents,
)


def _tag_replacement_mask(mask):
    setattr(mask, SCAIL_POSE2_CONDITION_MODE_ATTR, "replacement")
    setattr(mask, SCAIL_POSE2_MASK_ROLE_ATTR, SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE)
    return mask


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
class ScailPose2SamplesInitializationTests(unittest.TestCase):
    def test_replacement_subject_keeps_random_noise_when_add_noise_is_enabled(self) -> None:
        import torch

        noise = torch.full((2, 1, 2, 2), 10.0, dtype=torch.float32)
        input_samples = torch.full_like(noise, 100.0)
        mask = _tag_replacement_mask(
            torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        )
        latent_mask, _ = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 2, 2),
            channel_count=2,
            latent_grow_pixels=0,
        )

        initialized, contract = apply_samples_to_noise(
            noise,
            input_samples,
            noise_mask=latent_mask,
            timestep=torch.tensor([500.0], dtype=torch.float32),
            add_noise_to_samples=True,
            scail_pose2_replacement=True,
        )

        self.assertTrue(contract.mask_aware)
        self.assertTrue(contract.scail_pose2_replacement)
        self.assertAlmostEqual(0.25, contract.subject_ratio)
        self.assertEqual(10.0, float(initialized[0, 0, 0, 0].item()))
        self.assertEqual(55.0, float(initialized[0, 0, 0, 1].item()))
        self.assertEqual(55.0, float(initialized[1, 0, 1, 1].item()))

    def test_replacement_subject_keeps_random_noise_when_add_noise_is_disabled(self) -> None:
        import torch

        noise = torch.full((1, 1, 2, 2), 7.0, dtype=torch.float32)
        input_samples = torch.full_like(noise, 99.0)
        mask = _tag_replacement_mask(
            torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)
        )
        latent_mask, _ = resize_noise_mask_for_latents(
            mask,
            latent_shape=(1, 2, 2),
            channel_count=1,
            latent_grow_pixels=0,
        )

        initialized, contract = apply_samples_to_noise(
            noise,
            input_samples,
            noise_mask=latent_mask,
            timestep=torch.tensor([0.0], dtype=torch.float32),
            add_noise_to_samples=False,
            scail_pose2_replacement=True,
        )

        self.assertTrue(contract.mask_aware)
        self.assertEqual(7.0, float(initialized[0, 0, 0, 1].item()))
        self.assertEqual(99.0, float(initialized[0, 0, 0, 0].item()))
        self.assertEqual(99.0, float(initialized[0, 0, 1, 1].item()))

    def test_generic_masks_keep_existing_full_sample_initialization(self) -> None:
        import torch

        noise = torch.full((1, 1, 2, 2), 10.0, dtype=torch.float32)
        input_samples = torch.full_like(noise, 100.0)
        latent_mask = torch.tensor(
            [[[[[1.0, 0.0], [0.0, 0.0]]]]],
            dtype=torch.float32,
        )

        initialized, contract = apply_samples_to_noise(
            noise,
            input_samples,
            noise_mask=latent_mask,
            timestep=torch.tensor([500.0], dtype=torch.float32),
            add_noise_to_samples=True,
            scail_pose2_replacement=False,
        )

        self.assertFalse(contract.mask_aware)
        self.assertEqual(55.0, float(initialized[0, 0, 0, 0].item()))
        self.assertEqual(55.0, float(initialized[0, 0, 1, 1].item()))


if __name__ == "__main__":
    unittest.main()
