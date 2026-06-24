from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (
    ROOT
    / "example_workflows"
    / "wanvideo_2_1_14B_SCAIL2_replacement_and_animate_dual_mode_example_01.json"
)


def _load_example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


class Scail2ExampleWorkflowTests(unittest.TestCase):
    def test_replacement_example_uses_raw_driving_video_condition_route(self) -> None:
        data = _load_example()
        nodes = {node["id"]: node for node in data["nodes"]}
        links = {link[0]: link for link in data["links"]}
        replacement_condition_video = nodes[479]
        self.assertEqual(
            "SCAILPose2ReplacementConditionVideo",
            replacement_condition_video["type"],
        )

        condition = nodes[447]
        driving_input = next(
            item for item in condition["inputs"] if item["name"] == "driving_video"
        )
        driving_link = links[driving_input["link"]]

        self.assertEqual([856, 479, 0, 447, 4, "IMAGE"], driving_link)
        replacement_driving_input = next(
            item
            for item in replacement_condition_video["inputs"]
            if item["name"] == "driving_video"
        )
        replacement_driving_link = links[replacement_driving_input["link"]]
        self.assertEqual([854, 478, 0, 479, 0, "IMAGE"], replacement_driving_link)
        self.assertEqual("Get_driving_video", nodes[478]["title"])

    def test_replacement_example_wires_denoise_mask_to_encode_samples_path(self) -> None:
        data = _load_example()
        nodes = {node["id"]: node for node in data["nodes"]}
        links = {link[0]: link for link in data["links"]}

        mask_node = nodes[454]
        mask_output = next(item for item in mask_node["outputs"] if item["name"] == "mask")
        mask_link = links[mask_output["links"][0]]

        self.assertEqual("SCAILPose2ReplacementDenoiseMask", mask_node["type"])
        self.assertEqual([836, 454, 0, 475, 2, "MASK"], mask_link)
        self.assertEqual("WanVideoEncode", nodes[475]["type"])


if __name__ == "__main__":
    unittest.main()
