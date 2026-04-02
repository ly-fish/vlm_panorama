from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class Anchor:
    scene_type: str = "panorama"
    subject: str = "general"
    teaching_stage: str = "editing"
    global_style: str = "realistic"

    def validate(self) -> None:
        if not self.scene_type:
            raise ValueError("anchor.scene_type must not be empty")


@dataclass
class CoreObject:
    object_id: str
    name: str
    category: str = "object"
    function: str = ""
    action: str = ""
    importance: str = "medium"
    edit_type: str = "preserve"
    required: bool = True
    physical_parent: str | None = None

    def validate(self) -> None:
        if not self.object_id:
            raise ValueError("core_object.object_id must not be empty")
        if not self.name:
            raise ValueError(f"core_object.name must not be empty for {self.object_id}")


@dataclass
class ContextObject:
    object_id: str
    name: str
    material: str = ""
    support_role: str = "ambient"
    visual_neighborhood: str = ""

    def validate(self) -> None:
        if not self.object_id:
            raise ValueError("context_object.object_id must not be empty")
        if not self.name:
            raise ValueError(f"context_object.name must not be empty for {self.object_id}")


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    direction: str = "unspecified"
    distance: str = "unspecified"
    affiliation_type: str = "contextual"

    def validate(self) -> None:
        if not self.source or not self.target:
            raise ValueError("relation source/target must not be empty")
        if not self.relation_type:
            raise ValueError("relation.relation_type must not be empty")


@dataclass
class AESGGraph:
    anchor: Anchor
    core_objects: list[CoreObject] = field(default_factory=list)
    context_objects: list[ContextObject] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    graph_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AESGGraph":
        anchor = Anchor(**payload.get("anchor", {}))
        core_objects = [CoreObject(**item) for item in payload.get("core_objects", [])]
        context_objects = [ContextObject(**item) for item in payload.get("context_objects", [])]
        relations = [Relation(**item) for item in payload.get("relations", [])]
        graph_meta = dict(payload.get("graph_meta", {}))
        graph = cls(
            anchor=anchor,
            core_objects=core_objects,
            context_objects=context_objects,
            relations=relations,
            graph_meta=graph_meta,
        )
        graph.validate()
        return graph

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        self.anchor.validate()
        object_ids = set()
        for obj in self.core_objects:
            obj.validate()
            object_ids.add(obj.object_id)
        for obj in self.context_objects:
            obj.validate()
            object_ids.add(obj.object_id)
        for rel in self.relations:
            rel.validate()
            if rel.source not in object_ids:
                raise ValueError(f"relation.source {rel.source} not found in graph nodes")
            if rel.target not in object_ids and rel.target != "frame":
                raise ValueError(f"relation.target {rel.target} not found in graph nodes")


def normalize_scene_graph(scene_graph: dict[str, Any]) -> AESGGraph:
    objects = _as_list(scene_graph.get("objects"))
    spatial_relations = _as_list(scene_graph.get("spatial_relations"))

    anchor = Anchor(
        scene_type=_clean_text(scene_graph.get("scene_theme"), "panorama"),
        subject=_clean_text(scene_graph.get("subject_domain"), "general"),
        teaching_stage=_clean_text(scene_graph.get("pedagogical_goal"), "editing"),
        global_style="realistic",
    )

    core_objects: list[CoreObject] = []
    context_objects: list[ContextObject] = []
    for index, raw in enumerate(objects):
        object_id = _clean_text(raw.get("id"), f"obj_{index}")
        name = _clean_text(raw.get("name"), object_id)
        category = _clean_text(raw.get("category"), "object")
        importance = _clean_text(raw.get("importance"), "medium").lower()
        attributes = raw.get("attributes", {}) or {}
        function = _clean_text(attributes.get("visibility"), "")
        action = _clean_text(attributes.get("position_hint"), "")
        physical_parent = _clean_text(raw.get("physical_parent")) or None
        edit_type = "add" if raw.get("required", True) else "preserve"
        if importance == "high" or raw.get("required", False):
            core_objects.append(
                CoreObject(
                    object_id=object_id,
                    name=name,
                    category=category,
                    function=function,
                    action=action,
                    importance=importance or "high",
                    edit_type=edit_type,
                    required=bool(raw.get("required", True)),
                    physical_parent=physical_parent,
                )
            )
        else:
            context_objects.append(
                ContextObject(
                    object_id=object_id,
                    name=name,
                    material=_clean_text(attributes.get("material"), ""),
                    support_role=_clean_text(attributes.get("visibility"), "ambient"),
                    visual_neighborhood=_clean_text(attributes.get("position_hint"), ""),
                )
            )

    known_ids = {obj.object_id for obj in core_objects} | {obj.object_id for obj in context_objects}
    for obj in core_objects:
        if obj.physical_parent and obj.physical_parent not in known_ids:
            context_objects.append(
                ContextObject(
                    object_id=obj.physical_parent,
                    name=obj.physical_parent.replace("_", " "),
                    support_role="support",
                    visual_neighborhood="derived_from_physical_parent",
                )
            )
            known_ids.add(obj.physical_parent)

    relations: list[Relation] = []
    for raw in spatial_relations:
        source = _clean_text(raw.get("subject"))
        target = _clean_text(raw.get("target"), "frame")
        relation_type = _clean_text(raw.get("relation"), "related_to")
        if not source:
            continue
        relations.append(
            Relation(
                source=source,
                target=target if target in known_ids else "frame",
                relation_type=relation_type,
                direction=_clean_text(raw.get("type"), "unspecified"),
                distance=_infer_distance(relation_type),
                affiliation_type=_infer_affiliation_type(relation_type),
            )
        )

    for obj in core_objects:
        if obj.physical_parent:
            relations.append(
                Relation(
                    source=obj.object_id,
                    target=obj.physical_parent,
                    relation_type="attached_to",
                    direction="relative",
                    distance="near",
                    affiliation_type="physical_support",
                )
            )

    graph_meta = {
        "source": "scene_graph",
        "safety_constraints": _as_list(scene_graph.get("safety_constraints")),
        "success_criteria": _as_list(scene_graph.get("success_criteria")),
    }

    graph = AESGGraph(
        anchor=anchor,
        core_objects=core_objects,
        context_objects=context_objects,
        relations=relations,
        graph_meta=graph_meta,
    )
    graph.validate()
    return graph


def build_aesg_graph(text: str, scene_graph: dict[str, Any] | None = None, aesg_json: dict[str, Any] | None = None) -> AESGGraph:
    if aesg_json is not None:
        return AESGGraph.from_dict(aesg_json)
    if scene_graph is not None:
        return normalize_scene_graph(scene_graph)
    anchor = Anchor(subject="general", teaching_stage=text or "editing")
    graph = AESGGraph(anchor=anchor, graph_meta={"source": "text_only"})
    graph.validate()
    return graph


def build_aesg_prompt(graph: AESGGraph) -> str:
    parts: list[str] = []
    anchor_text = " ".join(
        value for value in [graph.anchor.scene_type, graph.anchor.subject, graph.anchor.teaching_stage] if value
    ).strip()
    if anchor_text:
        parts.append(f"Scene intent: {anchor_text}.")

    if graph.core_objects:
        objects = []
        for obj in graph.core_objects[:6]:
            snippet = obj.name
            if obj.action:
                snippet = f"{snippet} at {obj.action}"
            if obj.physical_parent:
                snippet = f"{snippet}, attached to {obj.physical_parent}"
            objects.append(snippet)
        parts.append("Primary teaching objects: " + "; ".join(objects) + ".")

    if graph.relations:
        relations = []
        for rel in graph.relations[:6]:
            relations.append(f"{rel.source} {rel.relation_type} {rel.target}")
        parts.append("Keep spatial relations: " + "; ".join(relations) + ".")

    if graph.context_objects:
        context = []
        for obj in graph.context_objects[:4]:
            snippet = obj.name
            if obj.visual_neighborhood:
                snippet = f"{snippet} near {obj.visual_neighborhood}"
            context.append(snippet)
        parts.append("Preserve supporting context: " + "; ".join(context) + ".")

    constraints = graph.graph_meta.get("safety_constraints", [])
    if constraints:
        parts.append("Safety constraints: " + "; ".join(str(item) for item in constraints[:3]) + ".")

    return " ".join(parts).strip()


def _infer_distance(relation_type: str) -> str:
    relation_type = relation_type.lower()
    if any(token in relation_type for token in ("near", "next", "beside", "on", "inside")):
        return "near"
    if any(token in relation_type for token in ("far", "opposite")):
        return "far"
    return "unspecified"


def _infer_affiliation_type(relation_type: str) -> str:
    relation_type = relation_type.lower()
    if any(token in relation_type for token in ("on", "inside", "attached", "support")):
        return "physical_support"
    return "contextual"
