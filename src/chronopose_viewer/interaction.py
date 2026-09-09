from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .model import Edge


class PointerButton(StrEnum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class PickResult:
    kind: Literal["node", "edge"]
    instance_id: str
    node_id: str | None = None
    edge: Edge | None = None
    fraction: float = 0.0

    @property
    def identity(self) -> tuple[object, ...]:
        if self.kind == "node":
            return self.kind, self.instance_id, self.node_id
        return self.kind, self.instance_id, self.edge


@dataclass(frozen=True, slots=True)
class NodeEmphasisStyle:
    role: Literal["selected", "action_source", "hovered"]
    color: str
    size_2d: float
    size_3d: float
    width: float


NODE_EMPHASIS_STYLES = (
    NodeEmphasisStyle("selected", "#00e5ff", 18.0, 17.0, 3.0),
    NodeEmphasisStyle("action_source", "#ff4fd8", 24.0, 23.0, 4.0),
    NodeEmphasisStyle("hovered", "#76ff03", 30.0, 29.0, 3.0),
)


def node_emphasis_styles(
    instance_id: str,
    node_id: str,
    *,
    selected_node: tuple[str, str] | None,
    action_source: tuple[str, str] | None,
    hovered: PickResult | None,
) -> tuple[NodeEmphasisStyle, ...]:
    """Return independently visible concentric rings for every active node state."""

    identity = (instance_id, node_id)
    active_roles: set[str] = set()
    if selected_node == identity:
        active_roles.add("selected")
    if action_source == identity:
        active_roles.add("action_source")
    if (
        hovered is not None
        and hovered.kind == "node"
        and hovered.instance_id == instance_id
        and hovered.node_id == node_id
    ):
        active_roles.add("hovered")
    return tuple(style for style in NODE_EMPHASIS_STYLES if style.role in active_roles)
