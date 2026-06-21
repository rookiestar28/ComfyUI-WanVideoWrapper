from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WanVideoEncodeSocketNameTests(unittest.TestCase):
    def test_original_video_socket_is_named_driving_video(self) -> None:
        source = (ROOT / "nodes.py").read_text(encoding="utf-8")
        start = source.index("class WanVideoEncode:")
        end = source.index("NODE_CLASS_MAPPINGS", start)
        class_source = source[start:end]

        self.assertIn('"driving_video": ("IMAGE",)', class_source)
        self.assertIn("def encode(self, vae, driving_video,", class_source)
        self.assertIn("image = driving_video.clone()", class_source)
        self.assertNotIn('"pose_video": ("IMAGE",)', class_source)
        self.assertNotIn("def encode(self, vae, pose_video,", class_source)

    def test_scail_pose2_animation_mask_disables_samples_payload(self) -> None:
        source = (ROOT / "nodes.py").read_text(encoding="utf-8")
        start = source.index("class WanVideoEncode:")
        end = source.index("NODE_CLASS_MAPPINGS", start)
        class_source = source[start:end]

        self.assertIn("scail_pose2_mask_disables_samples(mask)", class_source)
        self.assertIn('"samples": None', class_source)
        self.assertIn('"noise_mask": None', class_source)
        self.assertIn('"scail_pose2_samples_disabled": True', class_source)

    def test_sampler_ignores_disabled_scail_pose2_samples_payload(self) -> None:
        source = (ROOT / "nodes_sampler.py").read_text(encoding="utf-8")

        self.assertIn('samples.get("scail_pose2_samples_disabled", False)', source)
        self.assertIn("samples = None", source)

    def test_sampler_uses_scail_pose2_noise_mask_contract_helper(self) -> None:
        source = (ROOT / "nodes_sampler.py").read_text(encoding="utf-8")

        self.assertIn("resize_noise_mask_for_latents", source)
        self.assertIn("apply_samples_to_noise", source)
        self.assertGreaterEqual(source.count("apply_samples_to_noise("), 2)
        self.assertIn("latent_grow_pixels=1", source)
        self.assertGreaterEqual(source.count("latent_grow_pixels=1"), 2)
        self.assertIn("noise_mask_contract.to_log_string()", source)
        self.assertIn("samples_init_contract.to_log_string()", source)
        self.assertNotIn("mode='trilinear',\n                        align_corners=False\n                    ).repeat(1, noise.shape[0], 1, 1, 1)", source)


if __name__ == "__main__":
    unittest.main()
