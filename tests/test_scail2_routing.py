from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_routing_module():
    spec = importlib.util.spec_from_file_location(
        "scail2_routing_under_test",
        ROOT / "SCAIL" / "scail2_routing.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SCAIL2RoutingTests(unittest.TestCase):
    def test_prepare_returns_none_without_scail2_embeds(self) -> None:
        module = import_routing_module()

        self.assertIsNone(
            module.prepare_scail2_data(
                {"target_shape": (16, 2, 1, 1)},
                dict_to_device=lambda data, device, dtype: data,
                device="cuda",
                dtype="float16",
            )
        )

    def test_prepare_copies_and_moves_scail2_embeds(self) -> None:
        module = import_routing_module()
        original = {"pose_latents": object()}
        image_embeds = {"scail2_embeds": original}
        calls = []

        def fake_dict_to_device(data, device, dtype):
            calls.append((data, device, dtype))
            data["moved"] = True
            return data

        result = module.prepare_scail2_data(
            image_embeds,
            dict_to_device=fake_dict_to_device,
            device="cuda:0",
            dtype="bfloat16",
        )

        self.assertIsNot(result, original)
        self.assertEqual(1, len(calls))
        self.assertEqual("cuda:0", calls[0][1])
        self.assertEqual("bfloat16", calls[0][2])
        self.assertNotIn("moved", original)
        self.assertIs(result["pose_latents"], original["pose_latents"])

    def test_prepare_rejects_v1_and_v2_together(self) -> None:
        module = import_routing_module()

        with self.assertRaisesRegex(ValueError, "v1 scail_embeds"):
            module.prepare_scail2_data(
                {"scail_embeds": {}, "scail2_embeds": {}},
                dict_to_device=lambda data, device, dtype: data,
                device="cuda",
                dtype="float16",
            )

    def test_prepare_rejects_non_dict_scail2_embeds(self) -> None:
        module = import_routing_module()

        with self.assertRaisesRegex(TypeError, "must be a dict"):
            module.prepare_scail2_data(
                {"scail2_embeds": object()},
                dict_to_device=lambda data, device, dtype: data,
                device="cuda",
                dtype="float16",
            )

    def test_context_windows_are_rejected_for_scail2_native(self) -> None:
        module = import_routing_module()
        scail2_data = {"pose_latents": object()}

        self.assertIs(scail2_data, module.scail2_context_window_input(scail2_data, None))
        with self.assertRaisesRegex(ValueError, "context windows"):
            module.scail2_context_window_input(scail2_data, slice(0, 2))

    def test_model_params_receive_scail2_input_only_when_present(self) -> None:
        module = import_routing_module()
        base_params = {"scail_input": None}
        scail2_data = {"pose_latents": object()}

        result = module.add_scail2_model_param(base_params, scail2_data)

        self.assertIs(result, base_params)
        self.assertIs(result["scail2_input"], scail2_data)

        no_scail2 = {"scail_input": None}
        module.add_scail2_model_param(no_scail2, None)
        self.assertNotIn("scail2_input", no_scail2)


if __name__ == "__main__":
    unittest.main()
