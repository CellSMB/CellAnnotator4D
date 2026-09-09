from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .model import GraphProject, InstanceGraph


@dataclass(frozen=True, slots=True)
class SuggestedLineageMatch:
    source_time: int
    source_instance_id: str
    target_time: int
    target_instance_id: str
    cost: float


@dataclass(frozen=True, slots=True)
class _GraphSignature:
    center: np.ndarray
    length: float
    extent: np.ndarray
    components: int
    leaves: int
    junctions: int
    empty: bool


def suggest_maximal_one_to_one_matches(
    project: GraphProject,
) -> list[SuggestedLineageMatch]:
    """Return a best-cost maximal matching for every pair of adjacent frames.

    Only free lineage endpoints participate. Existing incoming/outgoing events,
    including START and END markers, are therefore left untouched.
    """

    signatures: dict[tuple[int, str], _GraphSignature] = {}

    def signature(time: int, instance_id: str) -> _GraphSignature:
        key = (time, instance_id)
        if key not in signatures:
            signatures[key] = _graph_signature(
                project.instances_at(time)[instance_id],
                project.voxel_size_zyx,
            )
        return signatures[key]

    occupied_sources = {
        (event.source_time, instance_id)
        for event in project.lineage_events
        if event.source_time is not None
        for instance_id in event.sources
    }
    occupied_targets = {
        (event.target_time, instance_id)
        for event in project.lineage_events
        if event.target_time is not None
        for instance_id in event.targets
    }
    suggestions: list[SuggestedLineageMatch] = []
    for source_time in range(project.timepoints - 1):
        target_time = source_time + 1
        source_instances = project.instances_at(source_time)
        target_instances = project.instances_at(target_time)
        source_ids = [
            instance_id
            for instance_id in source_instances
            if (source_time, instance_id) not in occupied_sources
        ]
        target_ids = [
            instance_id
            for instance_id in target_instances
            if (target_time, instance_id) not in occupied_targets
        ]
        if not source_ids or not target_ids:
            continue

        costs = np.empty((len(source_ids), len(target_ids)), dtype=np.float64)
        for source_index, source_id in enumerate(source_ids):
            source_graph = source_instances[source_id]
            for target_index, target_id in enumerate(target_ids):
                target_graph = target_instances[target_id]
                costs[source_index, target_index] = _match_cost(
                    source_graph,
                    signature(source_time, source_id),
                    target_graph,
                    signature(target_time, target_id),
                    project.voxel_size_zyx,
                )

        source_indices, target_indices = linear_sum_assignment(costs)
        suggestions.extend(
            SuggestedLineageMatch(
                source_time=source_time,
                source_instance_id=source_ids[source_index],
                target_time=target_time,
                target_instance_id=target_ids[target_index],
                cost=float(costs[source_index, target_index]),
            )
            for source_index, target_index in zip(
                source_indices,
                target_indices,
                strict=True,
            )
        )
    return suggestions


def _graph_signature(
    graph: InstanceGraph,
    voxel_size_zyx: tuple[float, float, float],
) -> _GraphSignature:
    if not graph.nodes:
        return _GraphSignature(
            center=np.zeros(3, dtype=np.float64),
            length=0.0,
            extent=np.zeros(3, dtype=np.float64),
            components=0,
            leaves=0,
            junctions=0,
            empty=True,
        )

    node_ids = list(graph.nodes)
    node_indices = {node_id: index for index, node_id in enumerate(node_ids)}
    scale = np.asarray(voxel_size_zyx, dtype=np.float64)
    positions = np.asarray(
        [graph.nodes[node_id].position for node_id in node_ids],
        dtype=np.float64,
    )
    positions *= scale

    weights = np.zeros(len(node_ids), dtype=np.float64)
    degrees = np.zeros(len(node_ids), dtype=np.int64)
    total_length = 0.0
    for first, second in graph.edges:
        first_index = node_indices[first]
        second_index = node_indices[second]
        length = float(np.linalg.norm(positions[first_index] - positions[second_index]))
        total_length += length
        weights[first_index] += length / 2
        weights[second_index] += length / 2
        degrees[first_index] += 1
        degrees[second_index] += 1

    # Edge half-lengths approximate a centerline integral without bias from
    # uneven node spacing. Give isolated components one voxel of weight.
    unit = float(np.min(scale))
    weights[weights == 0] = unit
    center = np.average(positions, axis=0, weights=weights)
    extent = np.ptp(positions, axis=0)
    return _GraphSignature(
        center=center,
        length=total_length,
        extent=extent,
        components=len(graph.connected_components()),
        leaves=int(np.count_nonzero(degrees <= 1)),
        junctions=int(np.count_nonzero(degrees > 2)),
        empty=False,
    )


def _match_cost(
    source_graph: InstanceGraph,
    source: _GraphSignature,
    target_graph: InstanceGraph,
    target: _GraphSignature,
    voxel_size_zyx: tuple[float, float, float],
) -> float:
    if source.empty != target.empty:
        return 1.0e12

    unit = min(voxel_size_zyx)
    center_distance = float(np.linalg.norm(source.center - target.center))
    length_difference = abs(source.length - target.length)
    extent_difference = float(np.linalg.norm(source.extent - target.extent))
    topology_difference = (
        0.35 * abs(source.components - target.components)
        + 0.15 * abs(source.leaves - target.leaves)
        + 0.15 * abs(source.junctions - target.junctions)
    ) * unit
    cost = (
        center_distance + 0.25 * length_difference + 0.15 * extent_difference + topology_difference
    )

    # Equal IDs are a strong semantic hint (for example, imported track IDs),
    # but geometry still determines how all remaining instances are paired.
    if source_graph.id == target_graph.id:
        cost *= 0.05
    return cost
