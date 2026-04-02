from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_aesg_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    relation_prediction: torch.Tensor | None = None,
    relation_target: torch.Tensor | None = None,
    affiliation_prediction: torch.Tensor | None = None,
    affiliation_target: torch.Tensor | None = None,
    context_prediction: torch.Tensor | None = None,
    context_target: torch.Tensor | None = None,
    seam_prediction: torch.Tensor | None = None,
    seam_target: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    config = config or {}
    losses = {
        "L_edit": F.l1_loss(prediction, target) * float(config.get("lambda_edit", 1.0)),
        "L_rel": _optional_l1(relation_prediction, relation_target) * float(config.get("lambda_rel", 0.25)),
        "L_aff": _optional_l1(affiliation_prediction, affiliation_target) * float(config.get("lambda_aff", 0.25)),
        "L_ctx": _optional_l1(context_prediction, context_target) * float(config.get("lambda_ctx", 0.25)),
        "L_seam": _optional_l1(seam_prediction, seam_target) * float(config.get("lambda_seam", 0.1)),
    }
    losses["total"] = sum(losses.values())
    return losses


def _optional_l1(prediction: torch.Tensor | None, target: torch.Tensor | None) -> torch.Tensor:
    if prediction is None or target is None:
        return torch.tensor(0.0)
    return F.l1_loss(prediction, target)
