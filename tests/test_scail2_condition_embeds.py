from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key and key[-1] == slice(None, 3, None):
            return FakeTensor((*self.shape[:-1], 3))
        if key == 0:
            return FakeTensor(self.shape[1:])
        raise TypeError(f"unsupported fake tensor index: {key!r}")

    def permute(self, *dims):
        return FakeTensor(tuple(self.shape[index] for index in dims))

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def __mul__(self, other):
        return FakeTensor(self.shape)

    def __sub__(self, other):
        return FakeTensor(self.shape)


def install_torch_stub() -> None:
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.device = lambda name: name
    torch.as_tensor = lambda data, dtype=None: data if isinstance(data, FakeTensor) else FakeTensor(())

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")

    def interpolate(tensor, size, mode=None, align_corners=None):
        frames, channels, _height, _width = tensor.shape
        return FakeTensor((frames, channels, int(size[0]), int(size[1])))

    functional.interpolate = interpolate
    nn.functional = functional
    torch.nn = nn

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional


def install_comfy_stub() -> None:
    install_torch_stub()
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: "cpu"
    model_management.unet_offload_device = lambda: "cpu"
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management


def import_scail_nodes():
    install_comfy_stub()
    module_name = "wan_scail_nodes_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "SCAIL" / "nodes.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeVAE:
    dtype = "float32"

    def __init__(self) -> None:
        self.to_calls = []
        self.encode_calls = []

    def to(self, target):
        self.to_calls.append(str(target))
        return self

    def encode(self, images, device, tiled=False):
        image = images[0]
        self.encode_calls.append(
            {
                "shape": tuple(image.shape),
                "device": str(device),
                "tiled": tiled,
            }
        )
        _channels, frames, height, width = image.shape
        latent_h = max(height // 8, 1)
        latent_w = max(width // 8, 1)
        return [FakeTensor((16, frames, latent_h, latent_w))]


def image_batch(frames: int, height: int = 8, width: int = 8):
    return FakeTensor((frames, height, width, 3))


def runtime_mask(latent_frames: int):
    data = FakeTensor((1, latent_frames, 28, 1, 1))
    return {
        "data": data,
        "comfy_shape": tuple(data.shape),
        "scail2_shape": (28, latent_frames, 1, 1),
    }


def payload(*, include_additional: bool = True):
    additional_ref = {"image": image_batch(1)}
    additional_refs = [additional_ref] if include_additional else []
    additional_masks = [runtime_mask(1)] if include_additional else []
    condition = {
        "ref_image": image_batch(1),
        "pose_video": image_batch(5),
        "additional_references": additional_refs,
    }
    return {
        "kind": "wanvideo_scail2_condition_adapter",
        "version": 1,
        "schema": {
            "name": "scail_pose2.wanvideo_scail2_payload",
            "version": 1,
            "native_wrapper": {
                "embeds_key": "scail2_embeds",
            },
        },
        "condition": condition,
        "mode": "replacement",
        "replace_flag": True,
        "dimensions": {
            "width": 8,
            "height": 8,
            "num_frames": 5,
        },
        "source": {
            "source_kind": "unit_test",
        },
        "runtime_masks": {
            "reference": runtime_mask(1),
            "driving": runtime_mask(2),
            "additional_references": additional_masks,
        },
        "additional_references": additional_refs,
    }


class WanVideoAddSCAIL2ConditionEmbedsTests(unittest.TestCase):
    def test_node_contract_is_registered(self) -> None:
        module = import_scail_nodes()

        self.assertIn("WanVideoAddSCAIL2ConditionEmbeds", module.NODE_CLASS_MAPPINGS)
        self.assertIn("WanVideoAddSCAILPoseEmbeds", module.NODE_CLASS_MAPPINGS)
        self.assertIn("WanVideoAddSCAILReferenceEmbeds", module.NODE_CLASS_MAPPINGS)

        node_cls = module.NODE_CLASS_MAPPINGS["WanVideoAddSCAIL2ConditionEmbeds"]
        self.assertEqual(("WANVIDIMAGE_EMBEDS",), node_cls.RETURN_TYPES)
        self.assertEqual(("image_embeds",), node_cls.RETURN_NAMES)
        self.assertEqual(
            "SCAIL2_WANVIDEO_PAYLOAD",
            node_cls.INPUT_TYPES()["required"]["condition"][0],
        )

    def test_add_materializes_scail2_embeds(self) -> None:
        module = import_scail_nodes()
        vae = FakeVAE()
        embeds = {
            "target_shape": (16, 2, 1, 1),
            "num_frames": 5,
        }
        clip_embeds = {"clip_embeds": object()}

        result = module.WanVideoAddSCAIL2ConditionEmbeds().add(
            embeds,
            payload(),
            vae,
            clip_embeds=clip_embeds,
        )[0]

        self.assertIsNot(result, embeds)
        self.assertNotIn("scail_embeds", result)
        self.assertIn("scail2_embeds", result)

        scail2 = result["scail2_embeds"]
        self.assertTrue(scail2["replace_flag"])
        self.assertEqual("replacement", scail2["mode"])
        self.assertEqual([(16, 1, 1, 1)], [x.shape for x in scail2["ref_latents"]])
        self.assertEqual([(16, 5, 1, 1)], [x.shape for x in scail2["pose_latents"]])
        self.assertEqual([(28, 1, 1, 1)], [x.shape for x in scail2["ref_masks"]])
        self.assertEqual([(28, 2, 1, 1)], [x.shape for x in scail2["driving_masks"]])
        self.assertEqual([(16, 1, 1, 1)], [x.shape for x in scail2["additional_ref_latents"]])
        self.assertEqual([(28, 1, 1, 1)], [x.shape for x in scail2["additional_ref_masks"]])
        self.assertIs(scail2["clip_context"], clip_embeds["clip_embeds"])
        self.assertIsNone(scail2["segment"])
        self.assertEqual({"source_kind": "unit_test"}, scail2["source"])
        self.assertEqual((3, 1, 8, 8), vae.encode_calls[0]["shape"])
        self.assertEqual((3, 5, 4, 4), vae.encode_calls[1]["shape"])

    def test_rejects_invalid_payload(self) -> None:
        module = import_scail_nodes()
        bad_payload = payload()
        bad_payload["schema"] = {"name": "wrong", "version": 1, "native_wrapper": {"embeds_key": "scail2_embeds"}}

        with self.assertRaisesRegex(ValueError, "schema"):
            module.WanVideoAddSCAIL2ConditionEmbeds().add(
                {"target_shape": (16, 2, 1, 1)},
                bad_payload,
                FakeVAE(),
            )

    def test_rejects_existing_scail_embeds(self) -> None:
        module = import_scail_nodes()

        with self.assertRaisesRegex(ValueError, "v1 scail_embeds"):
            module.WanVideoAddSCAIL2ConditionEmbeds().add(
                {"target_shape": (16, 2, 1, 1), "scail_embeds": {}},
                payload(),
                FakeVAE(),
            )

    def test_rejects_target_shape_mismatch(self) -> None:
        module = import_scail_nodes()

        with self.assertRaisesRegex(ValueError, "dimensions"):
            module.WanVideoAddSCAIL2ConditionEmbeds().add(
                {"target_shape": (16, 2, 1, 2)},
                payload(),
                FakeVAE(),
            )


if __name__ == "__main__":
    unittest.main()
