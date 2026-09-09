from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable
from uuid import uuid4

Position = tuple[float, float, float]
Edge = tuple[str, str]


def new_id() -> str:
    return uuid4().hex


def canonical_edge(first: str, second: str) -> Edge:
    if first == second:
        raise ValueError("An edge must connect two different nodes")
    return (first, second) if first < second else (second, first)


def normalise_position(position: Iterable[float]) -> Position:
    values = tuple(float(value) for value in position)
    if len(values) != 3:
        raise ValueError("Node positions must contain exactly z, y, and x")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Node positions must be finite")
    return values  # type: ignore[return-value]


def normalise_radius(radius: float) -> float:
    value = float(radius)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Node radii must be non-negative and finite")
    return value


@dataclass(slots=True)
class GraphNode:
    id: str
    position: Position
    radius: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position_zyx": list(self.position),
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        return cls(
            id=str(data["id"]),
            position=normalise_position(data["position_zyx"]),
            radius=normalise_radius(data.get("radius", 0.0)),
        )


@dataclass(slots=True)
class InstanceGraph:
    id: str
    name: str
    color: str
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: set[Edge] = field(default_factory=set)

    def add_node(
        self,
        position: Iterable[float],
        *,
        node_id: str | None = None,
        radius: float = 0.0,
    ) -> str:
        node_id = node_id or new_id()
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id!r} already exists")
        self.nodes[node_id] = GraphNode(
            node_id,
            normalise_position(position),
            normalise_radius(radius),
        )
        return node_id

    def move_node(self, node_id: str, position: Iterable[float]) -> None:
        self._require_node(node_id)
        self.nodes[node_id].position = normalise_position(position)

    def set_node_radius(self, node_id: str, radius: float) -> None:
        self._require_node(node_id)
        self.nodes[node_id].radius = normalise_radius(radius)

    def add_edge(self, first: str, second: str) -> Edge:
        self._require_node(first)
        self._require_node(second)
        edge = canonical_edge(first, second)
        if edge in self.edges:
            raise ValueError("The selected nodes are already connected")
        self.edges.add(edge)
        return edge

    def remove_edge(self, first: str, second: str) -> Edge:
        edge = canonical_edge(first, second)
        if edge not in self.edges:
            raise KeyError(f"Edge {edge!r} does not exist")
        self.edges.remove(edge)
        return edge

    def remove_node(self, node_id: str) -> GraphNode:
        self._require_node(node_id)
        node = self.nodes.pop(node_id)
        self.edges = {edge for edge in self.edges if node_id not in edge}
        return node

    def neighbours(self, node_id: str) -> set[str]:
        self._require_node(node_id)
        neighbours: set[str] = set()
        for first, second in self.edges:
            if first == node_id:
                neighbours.add(second)
            elif second == node_id:
                neighbours.add(first)
        return neighbours

    def degree(self, node_id: str) -> int:
        return len(self.neighbours(node_id))

    def connected_components(self) -> list[set[str]]:
        return self._connected_components(self.edges)

    def connected_components_without_edge(self, first: str, second: str) -> list[set[str]]:
        removed_edge = canonical_edge(first, second)
        if removed_edge not in self.edges:
            raise KeyError(f"Edge {removed_edge!r} does not exist")
        return self._connected_components(edge for edge in self.edges if edge != removed_edge)

    def _connected_components(self, edges: Iterable[Edge]) -> list[set[str]]:
        adjacency = {node_id: set() for node_id in self.nodes}
        for candidate_first, candidate_second in edges:
            adjacency[candidate_first].add(candidate_second)
            adjacency[candidate_second].add(candidate_first)

        unseen = set(self.nodes)
        components: list[set[str]] = []
        while unseen:
            start = min(unseen)
            component: set[str] = set()
            stack = [start]
            unseen.remove(start)
            while stack:
                node_id = stack.pop()
                component.add(node_id)
                neighbours = adjacency[node_id] & unseen
                unseen.difference_update(neighbours)
                stack.extend(sorted(neighbours, reverse=True))
            components.append(component)
        return components

    def dissolve_degree_two_node(self, node_id: str) -> Edge:
        neighbours = sorted(self.neighbours(node_id))
        if len(neighbours) != 2:
            raise ValueError("Only a node connected to exactly two edges can be dissolved")
        replacement = canonical_edge(neighbours[0], neighbours[1])
        self.remove_node(node_id)
        self.edges.add(replacement)
        return replacement

    def split_edge(
        self,
        first: str,
        second: str,
        position: Iterable[float] | None = None,
        *,
        node_id: str | None = None,
    ) -> str:
        edge = canonical_edge(first, second)
        if edge not in self.edges:
            raise KeyError(f"Edge {edge!r} does not exist")
        first_position = self.nodes[first].position
        second_position = self.nodes[second].position
        if position is None:
            fraction = 0.5
            position = tuple(
                (first_position[index] + second_position[index]) / 2 for index in range(3)
            )
        else:
            position = normalise_position(position)
            direction = tuple(second_position[index] - first_position[index] for index in range(3))
            denominator = sum(value * value for value in direction)
            fraction = (
                0.0
                if denominator == 0
                else sum(
                    (position[index] - first_position[index]) * direction[index]
                    for index in range(3)
                )
                / denominator
            )
            fraction = min(max(fraction, 0.0), 1.0)
        radius = self.nodes[first].radius + fraction * (
            self.nodes[second].radius - self.nodes[first].radius
        )
        self.edges.remove(edge)
        inserted = self.add_node(position, node_id=node_id, radius=radius)
        self.edges.add(canonical_edge(first, inserted))
        self.edges.add(canonical_edge(inserted, second))
        return inserted

    def _require_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id!r} does not exist")

    def validate(self) -> None:
        for node in self.nodes.values():
            normalise_position(node.position)
            normalise_radius(node.radius)
        for first, second in self.edges:
            if first == second:
                raise ValueError(f"Self-edge found on node {first!r}")
            self._require_node(first)
            self._require_node(second)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [list(edge) for edge in sorted(self.edges)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceGraph:
        graph = cls(id=str(data["id"]), name=str(data["name"]), color=str(data["color"]))
        for node_data in data.get("nodes", []):
            node = GraphNode.from_dict(node_data)
            if node.id in graph.nodes:
                raise ValueError(f"Duplicate node id {node.id!r}")
            graph.nodes[node.id] = node
        for edge_data in data.get("edges", []):
            if len(edge_data) != 2:
                raise ValueError("Edges must contain exactly two node ids")
            graph.edges.add(canonical_edge(str(edge_data[0]), str(edge_data[1])))
        graph.validate()
        return graph


class EventKind(StrEnum):
    START = "start"
    END = "end"
    ONE_TO_ONE = "one_to_one"
    FISSION = "fission"
    FUSION = "fusion"
    FISSION_FUSION = "fission_fusion"


@dataclass(frozen=True, slots=True)
class LineageEvent:
    id: str
    source_time: int | None
    target_time: int | None
    sources: tuple[str, ...]
    targets: tuple[str, ...]

    @property
    def kind(self) -> EventKind:
        if not self.sources:
            return EventKind.START
        if not self.targets:
            return EventKind.END
        if len(self.sources) == 1 and len(self.targets) == 1:
            return EventKind.ONE_TO_ONE
        if len(self.sources) == 1:
            return EventKind.FISSION
        if len(self.targets) == 1:
            return EventKind.FUSION
        return EventKind.FISSION_FUSION

    @property
    def label(self) -> str:
        labels = {
            EventKind.START: "start",
            EventKind.END: "end",
            EventKind.ONE_TO_ONE: "1 → 1",
            EventKind.FISSION: "fission",
            EventKind.FUSION: "fusion",
            EventKind.FISSION_FUSION: "fission + fusion",
        }
        return labels[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_time": self.source_time,
            "target_time": self.target_time,
            "sources": list(self.sources),
            "targets": list(self.targets),
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageEvent:
        return cls(
            id=str(data["id"]),
            source_time=None if data.get("source_time") is None else int(data["source_time"]),
            target_time=None if data.get("target_time") is None else int(data["target_time"]),
            sources=tuple(str(value) for value in data.get("sources", [])),
            targets=tuple(str(value) for value in data.get("targets", [])),
        )


@dataclass(frozen=True, slots=True)
class InstanceMergeResult:
    kept_instance_id: str
    removed_instance_id: str
    first_node_id: str
    second_node_id: str
    connecting_edge: Edge


@dataclass(frozen=True, slots=True)
class InstanceSplitResult:
    retained_instance_id: str
    new_instance_id: str
    removed_edge: Edge


DEFAULT_COLORS = (
    "#4cc9f0",
    "#f72585",
    "#b8f2a1",
    "#ffb703",
    "#9b5de5",
    "#00f5d4",
    "#fb8500",
    "#e9c46a",
    "#ef476f",
    "#90be6d",
)


@dataclass(slots=True)
class GraphProject:
    shape_tzyx: tuple[int, int, int, int]
    source_path: str
    source_axes: str = "TZYX"
    voxel_size_zyx: Position = (1.0, 1.0, 1.0)
    frames: dict[int, dict[str, InstanceGraph]] = field(default_factory=dict)
    lineage_events: list[LineageEvent] = field(default_factory=list)

    @property
    def timepoints(self) -> int:
        return self.shape_tzyx[0]

    def instances_at(self, time: int) -> dict[str, InstanceGraph]:
        self._require_time(time)
        return self.frames.setdefault(time, {})

    def add_instance(
        self,
        time: int,
        *,
        name: str | None = None,
        color: str | None = None,
        instance_id: str | None = None,
    ) -> InstanceGraph:
        instances = self.instances_at(time)
        instance_id = instance_id or new_id()
        if instance_id in instances:
            raise ValueError(f"Instance {instance_id!r} already exists at t={time}")
        ordinal = len(instances) + 1
        graph = InstanceGraph(
            id=instance_id,
            name=name or f"Instance {ordinal}",
            color=color or DEFAULT_COLORS[(ordinal - 1) % len(DEFAULT_COLORS)],
        )
        instances[instance_id] = graph
        return graph

    def copy_instance(self, source_time: int, instance_id: str, target_time: int) -> InstanceGraph:
        self._require_time(source_time)
        self._require_time(target_time)
        if source_time == target_time:
            raise ValueError("Choose a different frame for the graph copy")
        source = self.instances_at(source_time).get(instance_id)
        if source is None:
            raise KeyError(f"Instance {instance_id!r} does not exist at t={source_time}")
        copied = self.add_instance(target_time, name=source.name, color=source.color)
        node_ids: dict[str, str] = {}
        for node in source.nodes.values():
            node_ids[node.id] = copied.add_node(node.position, radius=node.radius)
        for first, second in source.edges:
            copied.add_edge(node_ids[first], node_ids[second])
        return copied

    def connect_instances(
        self,
        time: int,
        first_instance_id: str,
        first_node_id: str,
        second_instance_id: str,
        second_node_id: str,
    ) -> InstanceMergeResult:
        """Join two instance graphs with an edge, retaining the first instance's identity."""

        instances = self.instances_at(time)
        if first_instance_id == second_instance_id:
            raise ValueError("Choose nodes from two different instances")
        first_graph = instances.get(first_instance_id)
        second_graph = instances.get(second_instance_id)
        if first_graph is None:
            raise KeyError(f"Instance {first_instance_id!r} does not exist at t={time}")
        if second_graph is None:
            raise KeyError(f"Instance {second_instance_id!r} does not exist at t={time}")
        first_graph._require_node(first_node_id)
        second_graph._require_node(second_node_id)

        staged_events = self._lineage_events_after_instance_merge(
            time,
            kept_instance_id=first_instance_id,
            removed_instance_id=second_instance_id,
        )

        node_mapping: dict[str, str] = {}
        reserved_node_ids = set(first_graph.nodes)
        for node_id in second_graph.nodes:
            mapped_id = node_id
            while mapped_id in reserved_node_ids:
                mapped_id = new_id()
            node_mapping[node_id] = mapped_id
            reserved_node_ids.add(mapped_id)

        mapped_second_node = node_mapping[second_node_id]
        connecting_edge = canonical_edge(first_node_id, mapped_second_node)
        for node in second_graph.nodes.values():
            mapped_id = node_mapping[node.id]
            first_graph.nodes[mapped_id] = GraphNode(
                mapped_id,
                node.position,
                node.radius,
            )
        first_graph.edges.update(
            canonical_edge(node_mapping[first], node_mapping[second])
            for first, second in second_graph.edges
        )
        first_graph.edges.add(connecting_edge)
        del instances[second_instance_id]
        self.lineage_events = staged_events
        self.validate()
        return InstanceMergeResult(
            kept_instance_id=first_instance_id,
            removed_instance_id=second_instance_id,
            first_node_id=first_node_id,
            second_node_id=mapped_second_node,
            connecting_edge=connecting_edge,
        )

    def can_split_instance_on_edge(self, time: int, instance_id: str, edge: Edge) -> bool:
        graph = self.instances_at(time).get(instance_id)
        if graph is None or len(graph.connected_components()) != 1:
            return False
        try:
            components = graph.connected_components_without_edge(*edge)
        except (KeyError, ValueError):
            return False
        return len(components) == 2

    def split_instance_on_edge(
        self,
        time: int,
        instance_id: str,
        edge: Edge,
    ) -> InstanceSplitResult:
        """Remove a bridge and turn its two resulting components into two instances."""

        instances = self.instances_at(time)
        graph = instances.get(instance_id)
        if graph is None:
            raise KeyError(f"Instance {instance_id!r} does not exist at t={time}")
        edge = canonical_edge(*edge)
        if len(graph.connected_components()) != 1:
            raise ValueError(
                "The instance is already disconnected; an unambiguous two-instance split "
                "requires one connected graph"
            )
        components = graph.connected_components_without_edge(*edge)
        if len(components) != 2:
            raise ValueError(
                "Removing this edge would not split the instance into exactly two "
                "connected components"
            )
        retained_nodes = next(component for component in components if edge[0] in component)
        new_nodes = next(component for component in components if component is not retained_nodes)

        ordinal = len(instances) + 1
        new_instance_id = new_id()
        new_graph = InstanceGraph(
            id=new_instance_id,
            name=f"{graph.name} split",
            color=DEFAULT_COLORS[(ordinal - 1) % len(DEFAULT_COLORS)],
            nodes={node_id: graph.nodes[node_id] for node_id in new_nodes},
            edges={
                candidate
                for candidate in graph.edges
                if candidate != edge and candidate[0] in new_nodes and candidate[1] in new_nodes
            },
        )
        retained_edges = {
            candidate
            for candidate in graph.edges
            if candidate != edge
            and candidate[0] in retained_nodes
            and candidate[1] in retained_nodes
        }
        staged_events = self._lineage_events_after_instance_split(
            time,
            instance_id=instance_id,
            new_instance_id=new_instance_id,
        )

        graph.nodes = {node_id: graph.nodes[node_id] for node_id in retained_nodes}
        graph.edges = retained_edges
        instances[new_instance_id] = new_graph
        self.lineage_events = staged_events
        self.validate()
        return InstanceSplitResult(
            retained_instance_id=instance_id,
            new_instance_id=new_instance_id,
            removed_edge=edge,
        )

    def remove_instance(self, time: int, instance_id: str) -> InstanceGraph:
        instances = self.instances_at(time)
        if instance_id not in instances:
            raise KeyError(f"Instance {instance_id!r} does not exist at t={time}")
        removed = instances.pop(instance_id)
        retained: list[LineageEvent] = []
        for event in self.lineage_events:
            sources = event.sources
            targets = event.targets
            if event.source_time == time:
                sources = tuple(value for value in sources if value != instance_id)
            if event.target_time == time:
                targets = tuple(value for value in targets if value != instance_id)
            if sources or targets:
                retained.append(
                    LineageEvent(
                        id=event.id,
                        source_time=event.source_time if sources else None,
                        target_time=event.target_time if targets else None,
                        sources=sources,
                        targets=targets,
                    )
                )
        self.lineage_events = retained
        return removed

    def add_lineage_event(
        self,
        *,
        source_time: int | None,
        target_time: int | None,
        sources: Iterable[str] = (),
        targets: Iterable[str] = (),
        event_id: str | None = None,
    ) -> LineageEvent:
        source_ids = tuple(dict.fromkeys(str(value) for value in sources))
        target_ids = tuple(dict.fromkeys(str(value) for value in targets))
        event = LineageEvent(
            id=event_id or new_id(),
            source_time=source_time,
            target_time=target_time,
            sources=source_ids,
            targets=target_ids,
        )
        self._validate_event(event)
        for existing in self.lineage_events:
            if self._events_overlap(existing, event):
                raise ValueError("An instance already has a lineage event on this frame endpoint")
        self.lineage_events.append(event)
        return event

    def remove_lineage_event(self, event_id: str) -> LineageEvent:
        for index, event in enumerate(self.lineage_events):
            if event.id == event_id:
                return self.lineage_events.pop(index)
        raise KeyError(f"Lineage event {event_id!r} does not exist")

    def incoming_event(self, time: int, instance_id: str) -> LineageEvent | None:
        self._require_time(time)
        for event in self.lineage_events:
            if event.target_time == time and instance_id in event.targets:
                return event
        return None

    def outgoing_event(self, time: int, instance_id: str) -> LineageEvent | None:
        self._require_time(time)
        for event in self.lineage_events:
            if event.source_time == time and instance_id in event.sources:
                return event
        return None

    def connect_lineage_instances(
        self,
        first_time: int,
        first_instance: str,
        second_time: int,
        second_instance: str,
    ) -> LineageEvent:
        self._require_time(first_time)
        self._require_time(second_time)
        if first_time == second_time:
            raise ValueError("Lineage connections must join different timepoints")
        if first_time < second_time:
            source_time, source_id = first_time, first_instance
            target_time, target_id = second_time, second_instance
        else:
            source_time, source_id = second_time, second_instance
            target_time, target_id = first_time, first_instance
        if source_id not in self.instances_at(source_time):
            raise KeyError(f"Instance {source_id!r} does not exist at t={source_time}")
        if target_id not in self.instances_at(target_time):
            raise KeyError(f"Instance {target_id!r} does not exist at t={target_time}")

        source_event = self.outgoing_event(source_time, source_id)
        target_event = self.incoming_event(target_time, target_id)
        candidates = list(
            dict.fromkeys(event.id for event in (source_event, target_event) if event is not None)
        )
        events = [
            next(event for event in self.lineage_events if event.id == event_id)
            for event_id in candidates
        ]

        for event in events:
            if event.source_time not in (None, source_time):
                raise ValueError(f"The selected source already connects from t={event.source_time}")
            if event.target_time not in (None, target_time):
                raise ValueError(f"The selected target already connects to t={event.target_time}")

        sources = [source_id]
        targets = [target_id]
        for event in events:
            sources.extend(event.sources)
            targets.extend(event.targets)
        source_ids = tuple(dict.fromkeys(sources))
        target_ids = tuple(dict.fromkeys(targets))

        if (
            len(events) == 1
            and events[0].source_time == source_time
            and events[0].target_time == target_time
            and events[0].sources == source_ids
            and events[0].targets == target_ids
        ):
            return events[0]

        retained = [event for event in self.lineage_events if event.id not in candidates]
        merged = LineageEvent(
            id=events[0].id if events else new_id(),
            source_time=source_time,
            target_time=target_time,
            sources=source_ids,
            targets=target_ids,
        )
        self._validate_event(merged)
        for event in retained:
            if self._events_overlap(event, merged):
                raise ValueError("An instance already has a lineage event on this frame endpoint")
        self.lineage_events = retained + [merged]
        return merged

    def mark_lineage_start(self, time: int, instance_id: str) -> LineageEvent:
        self._require_time(time)
        if instance_id not in self.instances_at(time):
            raise KeyError(f"Instance {instance_id!r} does not exist at t={time}")
        existing = self.incoming_event(time, instance_id)
        if existing is not None:
            if existing.source_time is None:
                return existing
            raise ValueError("The selected instance already has an incoming lineage connection")
        return self.add_lineage_event(source_time=None, target_time=time, targets=(instance_id,))

    def mark_lineage_end(self, time: int, instance_id: str) -> LineageEvent:
        self._require_time(time)
        if instance_id not in self.instances_at(time):
            raise KeyError(f"Instance {instance_id!r} does not exist at t={time}")
        existing = self.outgoing_event(time, instance_id)
        if existing is not None:
            if existing.target_time is None:
                return existing
            raise ValueError("The selected instance already has an outgoing lineage connection")
        return self.add_lineage_event(source_time=time, target_time=None, sources=(instance_id,))

    def events_touching_time(self, time: int) -> list[LineageEvent]:
        self._require_time(time)
        return [
            event
            for event in self.lineage_events
            if event.source_time == time or event.target_time == time
        ]

    def _lineage_events_after_instance_merge(
        self,
        time: int,
        *,
        kept_instance_id: str,
        removed_instance_id: str,
    ) -> list[LineageEvent]:
        mapped: list[LineageEvent] = []
        for event in self.lineage_events:
            sources = event.sources
            targets = event.targets
            if event.source_time == time:
                sources = tuple(
                    dict.fromkeys(
                        kept_instance_id if value == removed_instance_id else value
                        for value in sources
                    )
                )
            if event.target_time == time:
                targets = tuple(
                    dict.fromkeys(
                        kept_instance_id if value == removed_instance_id else value
                        for value in targets
                    )
                )
            mapped.append(
                LineageEvent(
                    id=event.id,
                    source_time=event.source_time,
                    target_time=event.target_time,
                    sources=sources,
                    targets=targets,
                )
            )

        mapped = self._consolidate_merged_lineage_endpoint(
            mapped,
            time=time,
            instance_id=kept_instance_id,
            incoming=True,
        )
        mapped = self._consolidate_merged_lineage_endpoint(
            mapped,
            time=time,
            instance_id=kept_instance_id,
            incoming=False,
        )
        for index, event in enumerate(mapped):
            self._validate_event(event)
            for previous in mapped[:index]:
                if self._events_overlap(previous, event):
                    raise ValueError(
                        "Merging these instances would create contradictory lineage events"
                    )
        return mapped

    @staticmethod
    def _consolidate_merged_lineage_endpoint(
        events: list[LineageEvent],
        *,
        time: int,
        instance_id: str,
        incoming: bool,
    ) -> list[LineageEvent]:
        if incoming:
            candidates = [
                event
                for event in events
                if event.target_time == time and instance_id in event.targets
            ]
            matched = [event for event in candidates if event.sources]
            markers = [event for event in candidates if not event.sources]
            if len(candidates) <= 1:
                return events
            candidate_ids = {event.id for event in candidates}
            retained = [event for event in events if event.id not in candidate_ids]
            if matched:
                source_times = {event.source_time for event in matched}
                if len(source_times) != 1:
                    raise ValueError(
                        "Cannot merge instances with incoming lineage from different frames"
                    )
                for marker in markers:
                    remaining_targets = tuple(
                        value for value in marker.targets if value != instance_id
                    )
                    if remaining_targets:
                        retained.append(
                            LineageEvent(
                                marker.id,
                                None,
                                time,
                                (),
                                remaining_targets,
                            )
                        )
                retained.append(
                    LineageEvent(
                        id=matched[0].id,
                        source_time=matched[0].source_time,
                        target_time=time,
                        sources=tuple(
                            dict.fromkeys(source for event in matched for source in event.sources)
                        ),
                        targets=tuple(
                            dict.fromkeys(target for event in matched for target in event.targets)
                        ),
                    )
                )
                return retained
            retained.append(
                LineageEvent(
                    id=markers[0].id,
                    source_time=None,
                    target_time=time,
                    sources=(),
                    targets=tuple(
                        dict.fromkeys(target for event in markers for target in event.targets)
                    ),
                )
            )
            return retained

        candidates = [
            event for event in events if event.source_time == time and instance_id in event.sources
        ]
        matched = [event for event in candidates if event.targets]
        markers = [event for event in candidates if not event.targets]
        if len(candidates) <= 1:
            return events
        candidate_ids = {event.id for event in candidates}
        retained = [event for event in events if event.id not in candidate_ids]
        if matched:
            target_times = {event.target_time for event in matched}
            if len(target_times) != 1:
                raise ValueError("Cannot merge instances with outgoing lineage to different frames")
            for marker in markers:
                remaining_sources = tuple(value for value in marker.sources if value != instance_id)
                if remaining_sources:
                    retained.append(
                        LineageEvent(
                            marker.id,
                            time,
                            None,
                            remaining_sources,
                            (),
                        )
                    )
            retained.append(
                LineageEvent(
                    id=matched[0].id,
                    source_time=time,
                    target_time=matched[0].target_time,
                    sources=tuple(
                        dict.fromkeys(source for event in matched for source in event.sources)
                    ),
                    targets=tuple(
                        dict.fromkeys(target for event in matched for target in event.targets)
                    ),
                )
            )
            return retained
        retained.append(
            LineageEvent(
                id=markers[0].id,
                source_time=time,
                target_time=None,
                sources=tuple(
                    dict.fromkeys(source for event in markers for source in event.sources)
                ),
                targets=(),
            )
        )
        return retained

    def _lineage_events_after_instance_split(
        self,
        time: int,
        *,
        instance_id: str,
        new_instance_id: str,
    ) -> list[LineageEvent]:
        staged: list[LineageEvent] = []
        for event in self.lineage_events:
            sources = event.sources
            targets = event.targets
            if event.source_time == time and instance_id in sources:
                sources = tuple(
                    value
                    for source in sources
                    for value in ((source, new_instance_id) if source == instance_id else (source,))
                )
            if event.target_time == time and instance_id in targets:
                targets = tuple(
                    value
                    for target in targets
                    for value in ((target, new_instance_id) if target == instance_id else (target,))
                )
            staged.append(
                LineageEvent(
                    id=event.id,
                    source_time=event.source_time,
                    target_time=event.target_time,
                    sources=sources,
                    targets=targets,
                )
            )
        return staged

    def _validate_event(self, event: LineageEvent) -> None:
        if not event.sources and not event.targets:
            raise ValueError("A lineage event must contain a source or a target")
        if event.sources:
            if event.source_time is None:
                raise ValueError("Source instances require a source time")
            source_instances = self.instances_at(event.source_time)
            missing = set(event.sources) - set(source_instances)
            if missing:
                raise ValueError(f"Unknown source instances at t={event.source_time}: {missing}")
        elif event.source_time is not None:
            raise ValueError("A start event cannot have a source time")
        if event.targets:
            if event.target_time is None:
                raise ValueError("Target instances require a target time")
            target_instances = self.instances_at(event.target_time)
            missing = set(event.targets) - set(target_instances)
            if missing:
                raise ValueError(f"Unknown target instances at t={event.target_time}: {missing}")
        elif event.target_time is not None:
            raise ValueError("An end event cannot have a target time")
        if (
            event.source_time is not None
            and event.target_time is not None
            and event.source_time >= event.target_time
        ):
            raise ValueError("Lineage targets must occur after their sources")

    @staticmethod
    def _events_overlap(first: LineageEvent, second: LineageEvent) -> bool:
        same_source_endpoint = (
            first.source_time is not None
            and first.source_time == second.source_time
            and bool(set(first.sources) & set(second.sources))
        )
        same_target_endpoint = (
            first.target_time is not None
            and first.target_time == second.target_time
            and bool(set(first.targets) & set(second.targets))
        )
        return same_source_endpoint or same_target_endpoint

    def _require_time(self, time: int) -> None:
        if not 0 <= time < self.timepoints:
            raise IndexError(f"Time {time} is outside 0..{self.timepoints - 1}")

    def validate(self) -> None:
        if len(self.shape_tzyx) != 4 or any(value <= 0 for value in self.shape_tzyx):
            raise ValueError("shape_tzyx must contain four positive dimensions")
        if any(value <= 0 or not math.isfinite(value) for value in self.voxel_size_zyx):
            raise ValueError("voxel_size_zyx must contain three positive finite values")
        for time, instances in self.frames.items():
            self._require_time(time)
            for key, graph in instances.items():
                if key != graph.id:
                    raise ValueError("Instance dictionary keys must match instance ids")
                graph.validate()
                for node in graph.nodes.values():
                    if any(
                        coordinate < 0 or coordinate > size - 1
                        for coordinate, size in zip(node.position, self.shape_tzyx[1:], strict=True)
                    ):
                        raise ValueError(
                            f"Node {node.id!r} at t={time} lies outside the source volume"
                        )
        seen_events: set[str] = set()
        for index, event in enumerate(self.lineage_events):
            if event.id in seen_events:
                raise ValueError(f"Duplicate lineage event id {event.id!r}")
            seen_events.add(event.id)
            self._validate_event(event)
            for previous in self.lineage_events[:index]:
                if self._events_overlap(previous, event):
                    raise ValueError(
                        "An instance has contradictory lineage events on the same endpoint"
                    )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "chronopose-viewer",
            "schema_version": 1,
            "source": {
                "path": self.source_path,
                "source_axes": self.source_axes,
                "shape_tzyx": list(self.shape_tzyx),
                "voxel_size_zyx": list(self.voxel_size_zyx),
            },
            "frames": {
                str(time): [graph.to_dict() for graph in instances.values()]
                for time, instances in sorted(self.frames.items())
                if instances
            },
            "lineage_events": [event.to_dict() for event in self.lineage_events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphProject:
        if data.get("schema") != "chronopose-viewer":
            raise ValueError("This is not a Chronopose Viewer project")
        if int(data.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported project schema version")
        source = data["source"]
        project = cls(
            shape_tzyx=tuple(int(value) for value in source["shape_tzyx"]),  # type: ignore[arg-type]
            source_path=str(source["path"]),
            source_axes=str(source.get("source_axes", "TZYX")),
            voxel_size_zyx=normalise_position(source.get("voxel_size_zyx", (1, 1, 1))),
        )
        for time_text, graph_data in data.get("frames", {}).items():
            time = int(time_text)
            instances = project.instances_at(time)
            for item in graph_data:
                graph = InstanceGraph.from_dict(item)
                if graph.id in instances:
                    raise ValueError(f"Duplicate instance id {graph.id!r} at t={time}")
                instances[graph.id] = graph
        project.lineage_events = [
            LineageEvent.from_dict(item) for item in data.get("lineage_events", [])
        ]
        project.validate()
        return project
