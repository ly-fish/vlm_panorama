from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from aesg.schema import AESGGraph


@dataclass
class AESGCondition:
    anchor_tokens: torch.Tensor
    object_tokens: torch.Tensor
    context_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    branch_masks: dict[str, torch.Tensor]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_tokens": self.anchor_tokens,
            "object_tokens": self.object_tokens,
            "context_tokens": self.context_tokens,
            "relation_tokens": self.relation_tokens,
            "branch_masks": self.branch_masks,
            "metadata": self.metadata,
        }


class AESGEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3584,
        max_anchor_tokens: int = 2,
        max_object_tokens: int = 8,
        max_context_tokens: int = 8,
        max_relation_tokens: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.branch_limits = {
            "anchor": max_anchor_tokens,
            "object": max_object_tokens,
            "context": max_context_tokens,
            "relation": max_relation_tokens,
        }
        self.feature_dim = 8
        self.anchor_proj = nn.Linear(self.feature_dim, hidden_size)
        self.object_proj = nn.Linear(self.feature_dim, hidden_size)
        self.context_proj = nn.Linear(self.feature_dim, hidden_size)
        self.relation_proj = nn.Linear(self.feature_dim, hidden_size)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(7)
            for layer in (self.anchor_proj, self.object_proj, self.context_proj, self.relation_proj):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, graph: AESGGraph) -> AESGCondition:
        anchor_vectors = self._encode_anchor(graph)
        object_vectors = self._encode_objects(graph)
        context_vectors = self._encode_context(graph)
        relation_vectors = self._encode_relations(graph)

        anchor_tokens, anchor_mask = self._project_branch(anchor_vectors, self.anchor_proj, "anchor")
        object_tokens, object_mask = self._project_branch(object_vectors, self.object_proj, "object")
        context_tokens, context_mask = self._project_branch(context_vectors, self.context_proj, "context")
        relation_tokens, relation_mask = self._project_branch(relation_vectors, self.relation_proj, "relation")

        return AESGCondition(
            anchor_tokens=anchor_tokens,
            object_tokens=object_tokens,
            context_tokens=context_tokens,
            relation_tokens=relation_tokens,
            branch_masks={
                "anchor": anchor_mask,
                "object": object_mask,
                "context": context_mask,
                "relation": relation_mask,
            },
            metadata={
                "hidden_size": self.hidden_size,
                "branch_limits": self.branch_limits,
                "num_core_objects": len(graph.core_objects),
                "num_context_objects": len(graph.context_objects),
                "num_relations": len(graph.relations),
            },
        )

    def _encode_anchor(self, graph: AESGGraph) -> list[list[float]]:
        return [
            self._text_features(graph.anchor.scene_type),
            self._text_features(
                " ".join([graph.anchor.subject, graph.anchor.teaching_stage, graph.anchor.global_style]).strip()
            ),
        ]

    def _encode_objects(self, graph: AESGGraph) -> list[list[float]]:
        values = []
        for obj in graph.core_objects:
            values.append(
                self._text_features(
                    " ".join(
                        [
                            obj.name,
                            obj.category,
                            obj.function,
                            obj.action,
                            obj.importance,
                            obj.edit_type,
                            obj.physical_parent or "",
                        ]
                    )
                )
            )
        return values

    def _encode_context(self, graph: AESGGraph) -> list[list[float]]:
        values = []
        for obj in graph.context_objects:
            values.append(
                self._text_features(
                    " ".join([obj.name, obj.material, obj.support_role, obj.visual_neighborhood]).strip()
                )
            )
        return values

    def _encode_relations(self, graph: AESGGraph) -> list[list[float]]:
        values = []
        for rel in graph.relations:
            values.append(
                self._text_features(
                    " ".join(
                        [rel.source, rel.target, rel.relation_type, rel.direction, rel.distance, rel.affiliation_type]
                    )
                )
            )
        return values

    def _project_branch(
        self, vectors: list[list[float]], projection: nn.Linear, branch_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        limit = self.branch_limits[branch_name]
        tensor = torch.zeros((1, limit, self.feature_dim), dtype=projection.weight.dtype)
        mask = torch.zeros((1, limit), dtype=torch.bool)
        for index, vector in enumerate(vectors[:limit]):
            tensor[0, index] = torch.tensor(vector, dtype=projection.weight.dtype)
            mask[0, index] = True
        return projection(tensor), mask

    def _text_features(self, text: str) -> list[float]:
        text = text or ""
        encoded = text.encode("utf-8", errors="ignore")
        length = float(len(encoded))
        code_sum = float(sum(encoded))
        vowels = float(sum(byte in b"aeiouAEIOU" for byte in encoded))
        spaces = float(text.count(" "))
        digits = float(sum(char.isdigit() for char in text))
        unique_chars = float(len(set(text)))
        punctuation = float(sum(not char.isalnum() and not char.isspace() for char in text))
        norm = max(length, 1.0)
        return [
            length / 64.0,
            code_sum / (norm * 255.0),
            vowels / norm,
            spaces / norm,
            digits / norm,
            unique_chars / 64.0,
            punctuation / norm,
            1.0,
        ]


def encode_aesg(graph: AESGGraph, hidden_size: int = 3584) -> dict[str, Any]:
    encoder = AESGEncoder(hidden_size=hidden_size)
    return encoder(graph).to_dict()
