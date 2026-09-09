from __future__ import annotations

import numpy as np

from chronopose_viewer.mask_overlay import InstanceMaskColorizer
from chronopose_viewer.model import InstanceGraph


def test_disconnected_regions_with_same_mask_value_match_independent_graph_colors() -> None:
    masks = np.zeros((1, 5, 6, 7), dtype=np.uint8)
    masks[0, 1, 1, 1:3] = 7
    masks[0, 3, 4, 4:6] = 7
    masks[0, 0, 5, 6] = 9

    red = InstanceGraph("red", "red", "#ff1020")
    red.add_node((1, 1, 1), node_id="red-node")
    blue = InstanceGraph("blue", "blue", "#2040ff")
    blue.add_node((3, 4, 5), node_id="blue-node")
    graphs = {red.id: red, blue.id: blue}
    colorizer = InstanceMaskColorizer(masks, seed="example-mask")

    overlay = colorizer.frame(0, graphs)
    first_component = int(overlay.components[1, 1, 1])
    second_component = int(overlay.components[3, 4, 5])
    unmatched_component = int(overlay.components[0, 5, 6])

    assert first_component != second_component
    assert tuple(overlay.palette_rgba[first_component]) == (255, 16, 32, 255)
    assert tuple(overlay.palette_rgba[second_component]) == (32, 64, 255, 255)
    assert overlay.palette_rgba[unmatched_component, 3] == 255
    assert np.any(overlay.palette_rgba[unmatched_component, :3])
    assert colorizer.frame(0, graphs) is overlay

    red.name = "renamed"
    assert colorizer.frame(0, graphs) is overlay

    blue.move_node("blue-node", (0, 0, 0))
    updated = colorizer.frame(0, graphs)
    assert updated is not overlay
    assert updated.components is overlay.components
    assert tuple(updated.palette_rgba[first_component]) == (255, 16, 32, 255)


def test_component_uses_color_of_graph_with_most_centerline_voxels_inside() -> None:
    masks = np.zeros((1, 3, 3, 7), dtype=np.uint8)
    masks[0, 1, 1, 1:6] = 1

    short = InstanceGraph("short", "short", "#ff0000")
    short.add_node((1, 1, 1))
    long = InstanceGraph("long", "long", "#00ff00")
    first = long.add_node((1, 1, 2))
    second = long.add_node((1, 1, 5))
    long.add_edge(first, second)

    overlay = InstanceMaskColorizer(masks).frame(
        0,
        {short.id: short, long.id: long},
    )
    component = int(overlay.components[1, 1, 3])

    assert tuple(overlay.palette_rgba[component]) == (0, 255, 0, 255)
    colored = overlay.colorize(overlay.components[1])
    assert colored.shape == (3, 7, 4)
    assert tuple(colored[1, 3]) == (0, 255, 0, 255)
