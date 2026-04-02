from __future__ import annotations

from typing import Any

import torch


DEFAULT_BRANCH_FLAGS = {
    "anchor": True,
    "object": True,
    "context": True,
    "relation": True,
}


def _resolve_branch_flag(branch_name: str, config: dict[str, Any] | None) -> bool:
    if not config:
        return DEFAULT_BRANCH_FLAGS[branch_name]
    mapping = {
        "anchor": "use_anchor_branch",
        "object": "use_object_branch",
        "context": "use_context_branch",
        "relation": "use_relation_branch",
    }
    return bool(config.get(mapping[branch_name], True))


def _resolve_branch_scale(branch_name: str, config: dict[str, Any] | None) -> float:
    if not config:
        return 1.0
    return float(config.get(f"{branch_name}_branch_scale", 1.0))


def fuse_prompt_conditions(
    text_states: torch.Tensor,
    text_mask: torch.Tensor | None,
    aesg_cond: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if aesg_cond is None:
        return text_states, text_mask, {"enabled_branches": [], "num_condition_tokens": 0}
    if config and not bool(config.get("use_token_conditioning", False)):
        return text_states, text_mask, {"enabled_branches": [], "num_condition_tokens": 0, "skipped": True}

    fused_tokens = [text_states]
    fused_masks = [
        text_mask.to(dtype=torch.bool, device=text_states.device)
        if text_mask is not None
        else torch.ones(text_states.shape[:2], dtype=torch.bool, device=text_states.device)
    ]

    enabled_branches: list[str] = []
    num_condition_tokens = 0
    branch_masks = aesg_cond.get("branch_masks", {})
    branch_map = {
        "anchor": aesg_cond.get("anchor_tokens"),
        "object": aesg_cond.get("object_tokens"),
        "context": aesg_cond.get("context_tokens"),
        "relation": aesg_cond.get("relation_tokens"),
    }

    for branch_name, branch_tokens in branch_map.items():
        if branch_tokens is None or not _resolve_branch_flag(branch_name, config):
            continue
        branch_tokens = branch_tokens.to(device=text_states.device, dtype=text_states.dtype)
        branch_mask = branch_masks.get(branch_name)
        if branch_mask is None:
            branch_mask = torch.ones(branch_tokens.shape[:2], dtype=torch.bool, device=text_states.device)
        else:
            branch_mask = branch_mask.to(device=text_states.device, dtype=torch.bool)
        if not branch_mask.any():
            continue
        scale = _resolve_branch_scale(branch_name, config)
        fused_tokens.append(branch_tokens * scale)
        fused_masks.append(branch_mask)
        enabled_branches.append(branch_name)
        num_condition_tokens += int(branch_mask.sum().item())

    if len(fused_tokens) == 1:
        return text_states, text_mask, {"enabled_branches": [], "num_condition_tokens": 0}

    fused_states = torch.cat(fused_tokens, dim=1)
    fused_mask = torch.cat(fused_masks, dim=1) if fused_masks else None
    return fused_states, fused_mask, {
        "enabled_branches": enabled_branches,
        "num_condition_tokens": num_condition_tokens,
    }
