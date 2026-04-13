from __future__ import annotations

import os
from typing import Any

import torch
from PIL import Image

from aesg.encoder import encode_aesg
from aesg.schema import build_aesg_graph, build_aesg_prompt
from panorama_editing.qwen_image_editing.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from panorama_editing.reproject import reproject_to_erp
from panorama_editing.roi.roi_localization import localize_and_project_roi

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


_PIPELINE: QwenImageEditPlusPipeline | None = None


def load_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    defaults = {
        "use_anchor_branch": True,
        "use_object_branch": True,
        "use_relation_branch": True,
        "use_context_branch": True,
        "use_affiliation_edge": True,
        "use_token_conditioning": False,
        "use_textual_aesg_prompt": True,
        "use_local_editing": False,
        "roi_confidence_threshold": 0.7,
        "anchor_branch_scale": 0.05,
        "object_branch_scale": 0.05,
        "relation_branch_scale": 0.03,
        "context_branch_scale": 0.03,
        "roi_width_ratio": 0.33,
        "roi_height_ratio": 0.45,
        "default_fov": 75.0,
        "debug_save_intermediates": False,
        "lambda_edit": 1.0,
        "lambda_rel": 0.25,
        "lambda_aff": 0.25,
        "lambda_ctx": 0.25,
        "lambda_seam": 0.1,
    }
    if yaml is None:
        return defaults
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "aesg_edit.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    defaults.update(loaded)
    return defaults


def get_pipeline() -> QwenImageEditPlusPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _PIPELINE = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=dtype,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _PIPELINE.to(device)
        _PIPELINE.set_progress_bar_config(disable=None)
    return _PIPELINE


def _run_qwen_edit(
    image: Image.Image,
    text: str,
    *,
    aesg_condition: dict[str, Any] | None = None,
    aesg_config: dict[str, Any] | None = None,
) -> Image.Image:
    pipeline = get_pipeline()
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    output = pipeline(
        image=[image],
        prompt=text,
        generator=torch.Generator(device=generator_device).manual_seed(0),
        true_cfg_scale=4.0,
        negative_prompt=" ",
        num_inference_steps=40,
        guidance_scale=1.0,
        num_images_per_prompt=1,
        aesg_condition=aesg_condition,
        aesg_config=aesg_config,
    )
    return output.images[0]


def edit_panorama_without_aesg(
    image_erp: str | Image.Image,
    text: str,
    return_intermediates: bool = False,
) -> Image.Image | dict[str, Any]:
    image = Image.open(image_erp).convert("RGB") if isinstance(image_erp, str) else image_erp.convert("RGB")
    edited_image = _run_qwen_edit(image=image, text=text, aesg_condition=None, aesg_config=None)

    if not return_intermediates:
        return edited_image

    return {
        "edited_image": edited_image,
        "final_image": edited_image,
    }


def edit_panorama_with_aesg(
    image_erp: str | Image.Image,
    text: str,
    dialogue: str | None = None,
    roi_hint: dict[str, Any] | None = None,
    aesg_json: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    return_intermediates: bool = False,
) -> Image.Image | dict[str, Any]:
    config = load_config(config)
    image = Image.open(image_erp).convert("RGB") if isinstance(image_erp, str) else image_erp.convert("RGB")

    scene_graph_payload = None
    if aesg_json is None:
        try:
            from scene_graph.scene_graph import SceneGraph

            scene_graph_payload = SceneGraph().generate(text if dialogue is None else f"{text}\nContext: {dialogue}")
            if isinstance(scene_graph_payload, dict) and "error" in scene_graph_payload:
                scene_graph_payload = None
        except Exception:
            scene_graph_payload = None

    aesg_graph = build_aesg_graph(text=text, scene_graph=scene_graph_payload, aesg_json=aesg_json)
    condition_tokens = encode_aesg(aesg_graph)
    aesg_prompt = build_aesg_prompt(aesg_graph)
    effective_prompt = text
    if config.get("use_textual_aesg_prompt", True) and aesg_prompt:
        effective_prompt = f"{text}\n\nStructured constraints: {aesg_prompt}"

    use_local_editing = bool(config.get("use_local_editing", False))
    roi_result = None
    if use_local_editing or roi_hint is not None:
        roi_result = localize_and_project_roi(image, aesg_graph, roi_hint=roi_hint, config=config)
        confidence = float(roi_result["projection_meta"].get("confidence", 0.0))
        use_local_editing = use_local_editing and confidence >= float(config.get("roi_confidence_threshold", 0.7))

    if use_local_editing and roi_result is not None:
        edited_local = _run_qwen_edit(
            image=roi_result["local_image"],
            text=effective_prompt,
            aesg_condition=condition_tokens,
            aesg_config=config,
        )
        final_image = reproject_to_erp(image, edited_local, roi_result)
    else:
        edited_local = None
        final_image = _run_qwen_edit(
            image=image,
            text=effective_prompt,
            aesg_condition=condition_tokens,
            aesg_config=config,
        )

    if not return_intermediates:
        return final_image

    return {
        "scene_graph": scene_graph_payload,
        "aesg_graph": aesg_graph.to_dict(),
        "aesg_prompt": aesg_prompt,
        "effective_prompt": effective_prompt,
        "condition_tokens": condition_tokens,
        "roi_result": roi_result,
        "edited_local": edited_local,
        "final_image": final_image,
    }


def edit_panorama(
    image_erp: str | Image.Image,
    text: str,
    *,
    use_aesg: bool = True,
    dialogue: str | None = None,
    roi_hint: dict[str, Any] | None = None,
    aesg_json: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    return_intermediates: bool = False,
) -> Image.Image | dict[str, Any]:
    if use_aesg:
        return edit_panorama_with_aesg(
            image_erp=image_erp,
            text=text,
            dialogue=dialogue,
            roi_hint=roi_hint,
            aesg_json=aesg_json,
            config=config,
            return_intermediates=return_intermediates,
        )
    return edit_panorama_without_aesg(
        image_erp=image_erp,
        text=text,
        return_intermediates=return_intermediates,
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # input_path = os.path.join(base_dir, "input_image", "robot_dog.png")
    input_path = r"/users/2522553y/liangyue_ws/qwen_image/input_image/painting.png"
    prompt = "这是一张360° ERP全景图，请增加一个桌子"
    use_aesg = True
    
    suffix = "_aesg" if use_aesg else ""
    output_path = os.path.join(base_dir, "output_image", f"output_image_edit_2511{suffix}.png")
    result = edit_panorama(
        input_path,
        prompt,
        use_aesg=use_aesg,
    )
    result.save(output_path)
    print("image saved at", os.path.abspath(output_path))
