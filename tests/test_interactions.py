from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore

from chronopose_viewer.geometry import Plane
from chronopose_viewer.interaction import PickResult, node_emphasis_styles
from chronopose_viewer.mip_view import MiddlePanTurntableCamera, Mip3DView
from chronopose_viewer.model import InstanceGraph
from chronopose_viewer.ortho_view import InteractiveViewBox, OrthoView


class FakeWheelEvent:
    def __init__(self, delta: int, modifiers: QtCore.Qt.KeyboardModifier) -> None:
        self._delta = delta
        self._modifiers = modifiers
        self.accepted = False

    def delta(self) -> int:
        return self._delta

    def modifiers(self) -> QtCore.Qt.KeyboardModifier:
        return self._modifiers

    def accept(self) -> None:
        self.accepted = True


def test_shift_wheel_requests_slice_step() -> None:
    view_box = InteractiveViewBox(Plane.XY)
    steps: list[int] = []
    view_box.slice_stepped.connect(lambda plane, step: steps.append(step))
    event = FakeWheelEvent(120, QtCore.Qt.KeyboardModifier.ShiftModifier)

    view_box.wheelEvent(event)

    assert event.accepted
    assert steps == [1]


def test_3d_camera_uses_slice_consistent_z_and_overlay_ignores_volume_depth() -> None:
    camera = MiddlePanTurntableCamera(up="-z")
    assert camera.up == "-z"

    class FakeVisual:
        state: tuple[tuple[object, ...], dict[str, object]] | None = None

        def set_gl_state(self, *args: object, **kwargs: object) -> None:
            self.state = args, kwargs

    visual = FakeVisual()
    Mip3DView._make_overlay_visible(visual)
    assert visual.state is not None
    assert visual.state[1]["depth_test"] is False


def test_selected_source_and_hover_states_have_distinct_concentric_styles() -> None:
    identity = ("instance", "node")
    styles = node_emphasis_styles(
        *identity,
        selected_node=identity,
        action_source=identity,
        hovered=PickResult("node", identity[0], node_id=identity[1]),
    )

    assert [style.role for style in styles] == ["selected", "action_source", "hovered"]
    assert len({style.color for style in styles}) == 3
    assert [style.size_2d for style in styles] == sorted(style.size_2d for style in styles)
    assert [style.size_3d for style in styles] == sorted(style.size_3d for style in styles)


def _two_graphs() -> tuple[dict[str, InstanceGraph], str, str]:
    inactive = InstanceGraph("inactive", "inactive", "#ff3355")
    first = inactive.add_node((0, 0, 0), node_id="first")
    second = inactive.add_node((0, 0, 100), node_id="second")
    inactive.add_edge(first, second)

    active = InstanceGraph("active", "active", "#33aaff")
    active_first = active.add_node((50, 100, 0), node_id="active-first")
    active_second = active.add_node((50, 100, 100), node_id="active-second")
    active.add_edge(active_first, active_second)
    return {inactive.id: inactive, active.id: active}, first, second


def test_2d_graph_items_are_batched_and_highlights_reuse_base_geometry(qtbot) -> None:
    graphs, first, second = _two_graphs()
    view = OrthoView(Plane.XY)
    qtbot.addWidget(view)
    view.resize(700, 500)
    view.configure_shape((128, 128, 128))
    view.show()
    qtbot.waitExposed(view)
    view.fit_to_data()

    view.set_graphs(
        graphs,
        active_instance="active",
        selected_node=None,
        selected_edge=None,
    )

    assert len(view._graph_items) == 4
    assert len(view._projected_nodes) == 4
    assert len(view._projected_edges) == 2
    assert view.pick(0, 0) == PickResult("node", "inactive", node_id=first)
    edge_pick = view.pick(50, 0)
    assert edge_pick is not None
    assert edge_pick.edge == tuple(sorted((first, second)))
    assert edge_pick.fraction == 0.5
    base_items = tuple(view._graph_items)

    view.set_graphs(
        graphs,
        active_instance="active",
        selected_node=None,
        selected_edge=None,
        hovered=PickResult("node", "inactive", node_id=first),
    )

    assert tuple(view._graph_items) == base_items
    assert len(view._highlight_items) == 1

    graphs["inactive"].move_node(first, (0, 20, 0))
    view.set_graphs(
        graphs,
        active_instance="active",
        selected_node=None,
        selected_edge=None,
    )

    assert tuple(view._graph_items) != base_items


def test_2d_near_slice_emphasis_uses_absolute_node_opacity(qtbot) -> None:
    graph = InstanceGraph("graph", "graph", "#ff3355")
    graph.add_node((1, 2, 2), node_id="near")
    graph.add_node((8, 4, 4), node_id="far")
    graphs = {graph.id: graph}
    view = OrthoView(Plane.XY)
    qtbot.addWidget(view)
    view.configure_shape((12, 12, 12))
    view.set_slice_index(0)

    view.set_graphs(
        graphs,
        active_instance=None,
        selected_node=None,
        selected_edge=None,
    )
    baseline_scatter = next(
        item for item in view._graph_items if isinstance(item, pg.ScatterPlotItem)
    )
    baseline_alpha = [brush.color().alpha() for brush in baseline_scatter.data["brush"]]
    baseline_items = tuple(view._graph_items)

    view.set_graphs(
        graphs,
        active_instance=None,
        selected_node=None,
        selected_edge=None,
        emphasize_near_slice_nodes=True,
    )
    emphasized_scatter = next(
        item for item in view._graph_items if isinstance(item, pg.ScatterPlotItem)
    )
    emphasized_alpha = [brush.color().alpha() for brush in emphasized_scatter.data["brush"]]

    assert tuple(view._graph_items) != baseline_items
    assert emphasized_alpha[0] > baseline_alpha[0]
    assert emphasized_alpha[1] == 0
    assert emphasized_alpha[0] == round(255 * 0.75)
    assert OrthoView._node_depth_alpha(0, active=True, emphasize_near_slice=True) == 255


@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("distance", (0.0, 0.5, 1.0, 2.0, 3.5, 4.0, 8.0))
def test_default_near_slice_opacity_range_is_absolute(
    active: bool,
    distance: float,
) -> None:
    expected = round(255 * max(0.0, 1.0 - distance / 4.0))

    assert (
        OrthoView._node_depth_alpha(
            distance,
            active,
            emphasize_near_slice=True,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("distance", "opacity"),
    (
        (0.0, 0.8),
        (1.0, 0.8),
        (2.0, 0.65),
        (3.0, 0.5),
        (5.0, 0.2),
        (9.0, 0.2),
    ),
)
def test_near_slice_opacity_interpolates_between_configured_thresholds(
    distance: float,
    opacity: float,
) -> None:
    baseline = OrthoView._depth_alpha(distance, active=False)

    alpha = OrthoView._node_depth_alpha(
        distance,
        active=False,
        emphasize_near_slice=True,
        minimum_opacity=0.2,
        maximum_opacity=0.8,
        near_distance=1.0,
        far_distance=5.0,
    )

    assert alpha == round(255 * opacity)
    assert (
        OrthoView._node_depth_alpha(
            distance,
            active=False,
            emphasize_near_slice=False,
            minimum_opacity=0.2,
            maximum_opacity=0.8,
            near_distance=1.0,
            far_distance=5.0,
        )
        == baseline
    )


def test_3d_graph_visuals_are_batched_cached_and_remain_pickable(
    qtbot,
    monkeypatch,
) -> None:
    class IdentityTransform:
        @staticmethod
        def map(positions: np.ndarray) -> np.ndarray:
            return positions

    class FakeVisual:
        def __init__(self, pos: np.ndarray | None = None, parent: object = None, **_kwargs) -> None:
            self.parent = parent
            self.positions = pos
            self.order = 0

        def set_data(self, pos: np.ndarray, **_kwargs) -> None:
            self.positions = pos

        def set_gl_state(self, *_args, **_kwargs) -> None:
            pass

        def get_transform(self, *_args) -> IdentityTransform:
            return IdentityTransform()

    class FakeVisuals:
        Line = FakeVisual
        Markers = FakeVisual

    class FakeCanvas:
        def update(self) -> None:
            pass

    monkeypatch.setenv("CHRONOPOSE_VIEWER_DISABLE_3D", "1")
    view = Mip3DView()
    qtbot.addWidget(view)
    view._enabled = True
    view._scene = SimpleNamespace(visuals=FakeVisuals())
    view._view = SimpleNamespace(scene=object())
    view._canvas = FakeCanvas()
    graphs, first, _ = _two_graphs()

    view.set_graphs(
        graphs,
        active_instance="active",
        selected_node=None,
        selected_edge=None,
    )

    assert len(view._base_graph_visuals) == 5
    assert len(view._node_visuals) == 1
    assert len(view._edge_visuals) == 2
    assert view._pick((0, 0)) == PickResult("node", "inactive", node_id=first)
    edge_pick = view._pick((50, 0))
    assert edge_pick is not None
    assert edge_pick.kind == "edge"
    assert edge_pick.instance_id == "inactive"
    assert edge_pick.fraction == 0.5
    base_visuals = tuple(view._base_graph_visuals)

    view.set_graphs(
        graphs,
        active_instance="active",
        selected_node=None,
        selected_edge=None,
        hovered=PickResult("node", "inactive", node_id=first),
    )

    assert tuple(view._base_graph_visuals) == base_visuals
    assert len(view._highlight_visuals) == 1

    graphs["inactive"].move_node(first, (0, 20, 0))
    view.set_graphs(
        graphs,
        active_instance="active",
        selected_node=None,
        selected_edge=None,
    )

    assert tuple(view._base_graph_visuals) != base_visuals


def test_3d_mask_overlay_uses_categorical_translucency_and_reuses_volume(
    qtbot,
    monkeypatch,
) -> None:
    class FakeVolume:
        def __init__(self, data: np.ndarray, **kwargs) -> None:
            self.data = data
            self.cmap = kwargs["cmap"]
            self.clim = kwargs["clim"]
            self.method = kwargs["method"]
            self.interpolation = kwargs["interpolation"]
            self.visible = True
            self.order = 0.0
            self.transform = None
            self.gl_state = None
            self.set_data_calls = 0

        def set_data(self, data: np.ndarray, *, clim: tuple[int, int]) -> None:
            self.data = data
            self.clim = clim
            self.set_data_calls += 1

        def set_gl_state(self, *args, **kwargs) -> None:
            self.gl_state = args, kwargs

    class FakeCanvas:
        def update(self) -> None:
            pass

    monkeypatch.setenv("CHRONOPOSE_VIEWER_DISABLE_3D", "1")
    view = Mip3DView()
    qtbot.addWidget(view)
    view._enabled = True
    view._scene = SimpleNamespace(visuals=SimpleNamespace(Volume=FakeVolume))
    view._view = SimpleNamespace(scene=object())
    view._canvas = FakeCanvas()
    components = np.zeros((3, 4, 5), dtype=np.uint8)
    components[1:, 1:3, 2:4] = 1
    palette = np.asarray(((0, 0, 0, 0), (255, 20, 30, 255)), dtype=np.uint8)

    view.set_mask_overlay(
        components,
        palette,
        opacity=0.6,
        visible=True,
        data_key="frame-0",
    )

    volume = view._mask_volume
    assert volume.method == "translucent"
    assert volume.interpolation == "nearest"
    assert volume.gl_state[1]["depth_test"] is False
    assert volume.visible
    assert volume.cmap.map(np.asarray((0.0, 1.0)))[:, 3] == pytest.approx((0.0, 0.6))

    recolored = palette.copy()
    recolored[1, :3] = (10, 240, 50)
    view.set_mask_overlay(
        components,
        recolored,
        opacity=0.4,
        visible=True,
        data_key="frame-0",
    )

    assert view._mask_volume is volume
    assert volume.set_data_calls == 0
    assert volume.cmap.map(np.asarray((1.0,)))[0] == pytest.approx(
        (10 / 255, 240 / 255, 50 / 255, 0.4)
    )

    view.set_mask_visible(False)
    assert not volume.visible


def test_3d_radius_fill_reuses_volume_and_node_radii_use_2d_circles(
    qtbot,
    monkeypatch,
) -> None:
    class FakeVolume:
        def __init__(self, data: np.ndarray, **kwargs) -> None:
            self.data = data
            self.cmap = kwargs["cmap"]
            self.clim = kwargs["clim"]
            self.method = kwargs["method"]
            self.interpolation = kwargs["interpolation"]
            self.visible = True
            self.order = 0.0
            self.transform = None
            self.gl_state = None
            self.set_data_calls = 0

        def set_data(self, data: np.ndarray, *, clim: tuple[int, int]) -> None:
            self.data = data
            self.clim = clim
            self.set_data_calls += 1

        def set_gl_state(self, *args, **kwargs) -> None:
            self.gl_state = args, kwargs

    class FakeMarkers:
        def __init__(self, **kwargs) -> None:
            self.parent = kwargs["parent"]
            self.scaling = kwargs["scaling"]
            self.antialias = kwargs["antialias"]
            self.alpha = kwargs["alpha"]
            self.visible = True
            self.order = 0
            self.data: dict[str, object] = {}
            self.gl_state = None
            self.set_data_calls = 0

        def set_data(self, pos: np.ndarray | None = None, **kwargs) -> None:
            self.set_data_calls += 1
            if pos is not None:
                self.data["pos"] = pos
            self.data.update(kwargs)

        def set_gl_state(self, *args, **kwargs) -> None:
            self.gl_state = args, kwargs

    class FakeCanvas:
        def update(self) -> None:
            pass

    monkeypatch.setenv("CHRONOPOSE_VIEWER_DISABLE_3D", "1")
    view = Mip3DView()
    qtbot.addWidget(view)
    view._enabled = True
    view._scene = SimpleNamespace(
        visuals=SimpleNamespace(
            Volume=FakeVolume,
            Markers=FakeMarkers,
        )
    )
    view._view = SimpleNamespace(scene=object())
    view._canvas = FakeCanvas()
    components = np.zeros((3, 4, 5), dtype=np.uint8)
    components[1:, 1:3, 2:4] = 1
    palette = np.asarray(((0, 0, 0, 0), (20, 40, 255, 255)), dtype=np.uint8)

    view.set_radius_overlay(
        components,
        palette,
        steps_zyx=(2, 3, 4),
        opacity=0.3,
        visible=True,
        data_key="radius-frame-0",
    )

    volume = view._radius_volume
    assert volume.method == "translucent"
    assert volume.order == 0.75
    assert volume.gl_state[1]["depth_test"] is False
    assert tuple(volume.transform.scale)[:3] == (4, 3, 2)
    assert volume.cmap.map(np.asarray((0.0, 1.0)))[:, 3] == pytest.approx((0.0, 0.3))

    view.set_radius_overlay(
        components,
        palette,
        steps_zyx=(2, 3, 4),
        opacity=0.5,
        visible=True,
        data_key="radius-frame-0",
    )
    assert view._radius_volume is volume
    assert volume.set_data_calls == 0
    assert volume.cmap.map(np.asarray((1.0,)))[0, 3] == pytest.approx(0.5)

    graph = InstanceGraph("graph", "graph", "#123456")
    graph.add_node((2, 3, 4), node_id="visible", radius=2.0)
    graph.add_node((8, 7, 6), node_id="zero", radius=0.0)
    view.set_radius_spheres(
        {graph.id: graph},
        opacity=0.8,
        visible=True,
    )

    black, white = view._radius_sphere_visuals
    assert black.scaling == white.scaling == "scene"
    np.testing.assert_allclose(black.data["pos"], ((4.0, 3.0, 2.0),))
    np.testing.assert_allclose(black.data["size"], (4.0,))
    assert black.data["face_color"] == (0.0, 0.0, 0.0, 0.0)
    assert black.data["edge_color"] == (0.0, 0.0, 0.0, 1.0)
    assert white.data["edge_color"] == (1.0, 1.0, 1.0, 1.0)
    assert black.alpha == white.alpha == 0.8
    assert (black.data["edge_width_rel"], white.data["edge_width_rel"]) == (0.18, 0.06)
    assert (black.order, white.order) == (8, 9)
    assert black.gl_state[1]["depth_test"] is False
    circle_visuals = tuple(view._radius_sphere_visuals)

    view.set_radius_spheres(
        {graph.id: graph},
        opacity=0.4,
        visible=True,
    )

    assert tuple(view._radius_sphere_visuals) == circle_visuals
    assert black.alpha == white.alpha == 0.4
    assert black.set_data_calls == white.set_data_calls == 1
    np.testing.assert_allclose(black.data["pos"], ((4.0, 3.0, 2.0),))
