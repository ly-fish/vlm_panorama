from __future__ import annotations

import unittest

try:
    from PIL import Image
    from panorama_editing.reproject import reproject_to_erp
    from panorama_editing.roi.roi_localization import localize_and_project_roi

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    Image = None
    reproject_to_erp = None
    localize_and_project_roi = None
    PIL_AVAILABLE = False

try:
    import torch

    from aesg.encoder import encode_aesg
    from aesg.schema import AESGGraph, build_aesg_graph, normalize_scene_graph
    from modules.hcfm import fuse_prompt_conditions

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    encode_aesg = None
    AESGGraph = None
    build_aesg_graph = None
    normalize_scene_graph = None
    fuse_prompt_conditions = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE and PIL_AVAILABLE, "torch or pillow is not available in the current interpreter")
class AESGPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene_graph = {
            "scene_theme": "chemistry lab",
            "subject_domain": "chemistry",
            "pedagogical_goal": "titration demo",
            "objects": [
                {
                    "id": "table",
                    "name": "lab table",
                    "category": "furniture",
                    "importance": "high",
                    "required": True,
                    "physical_parent": None,
                    "attributes": {"visibility": "clear", "position_hint": "center"},
                },
                {
                    "id": "poster",
                    "name": "safety poster",
                    "category": "poster",
                    "importance": "low",
                    "required": False,
                    "physical_parent": None,
                    "attributes": {"visibility": "ambient", "position_hint": "near exit"},
                },
            ],
            "spatial_relations": [
                {"subject": "table", "type": "absolute", "relation": "center_of", "target": "frame"},
            ],
        }

    def test_schema_round_trip(self) -> None:
        graph = normalize_scene_graph(self.scene_graph)
        payload = graph.to_dict()
        rebuilt = AESGGraph.from_dict(payload)
        self.assertEqual(rebuilt.anchor.subject, "chemistry")
        self.assertEqual(len(rebuilt.core_objects), 1)
        self.assertEqual(len(rebuilt.context_objects), 1)

    def test_encoder_shapes(self) -> None:
        graph = build_aesg_graph(text="test", scene_graph=self.scene_graph)
        cond = encode_aesg(graph, hidden_size=32)
        self.assertEqual(tuple(cond["anchor_tokens"].shape), (1, 2, 32))
        self.assertEqual(cond["branch_masks"]["anchor"].dtype, torch.bool)

    def test_fusion_extends_prompt(self) -> None:
        graph = build_aesg_graph(text="test", scene_graph=self.scene_graph)
        cond = encode_aesg(graph, hidden_size=16)
        text_states = torch.randn(1, 5, 16)
        fused_states, fused_mask, meta = fuse_prompt_conditions(text_states, None, cond, {})
        self.assertGreater(fused_states.shape[1], text_states.shape[1])
        self.assertEqual(fused_mask.shape[1], fused_states.shape[1])
        self.assertIn("anchor", meta["enabled_branches"])

    def test_roi_and_reproject(self) -> None:
        image = Image.new("RGB", (400, 200), color="white")
        graph = build_aesg_graph(text="test", scene_graph=self.scene_graph)
        roi = localize_and_project_roi(image, graph)
        edited = Image.new("RGB", roi["local_image"].size, color="red")
        merged = reproject_to_erp(image, edited, roi, feather_radius=0)
        self.assertEqual(merged.size, image.size)
        self.assertNotEqual(merged.getpixel((roi["bbox"][0] + 5, roi["bbox"][1] + 5)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
