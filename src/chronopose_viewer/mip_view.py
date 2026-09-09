from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets
from vispy import keys
from vispy.color import Colormap
from vispy.scene.cameras import TurntableCamera
from vispy.scene.cameras.perspective import PerspectiveCamera

from .interaction import NODE_EMPHASIS_STYLES, PickResult, PointerButton
from .model import Edge, InstanceGraph


@dataclass(slots=True)
class _NodeVisualData:
    visual: object
    instance_ids: list[str]
    node_ids: list[str]
    positions_xyz: np.ndarray


@dataclass(slots=True)
class _EdgeVisualData:
    visual: object
    instance_ids: list[str]
    edges: list[Edge]
    positions_xyz: np.ndarray
    pick_orders: list[int]


def _hex_to_rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    value = color.lstrip("#")
    if len(value) != 6:
        value = "4cc9f0"
    try:
        red, green, blue = (int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))
    except ValueError:
        red, green, blue = (0.298, 0.788, 0.941)
    return red, green, blue, min(max(float(alpha), 0.0), 1.0)


def _categorical_colormap(palette_rgba: np.ndarray, opacity: float) -> Colormap:
    colors = np.asarray(palette_rgba, dtype=np.float32) / 255.0
    colors = colors.copy()
    colors[:, 3] *= min(max(float(opacity), 0.0), 1.0)
    component_count = len(colors) - 1
    if component_count <= 0:
        controls = np.asarray((0.0, 1.0), dtype=np.float32)
    else:
        boundaries = (np.arange(component_count, dtype=np.float32) + 0.5) / component_count
        controls = np.concatenate(
            (
                np.asarray((0.0,), dtype=np.float32),
                boundaries,
                np.asarray((1.0,), dtype=np.float32),
            )
        )
    return Colormap(colors, controls=controls, interpolation="zero")


class MiddlePanTurntableCamera(TurntableCamera):
    """Turntable camera with MMB pan and no RMB zoom binding."""

    def viewbox_mouse_event(self, event: object) -> None:
        if event.handled or not self.interactive:
            return
        PerspectiveCamera.viewbox_mouse_event(self, event)
        if event.type == "mouse_release":
            self._event_value = None
        elif event.type == "mouse_press":
            event.handled = True
        elif event.type == "mouse_move":
            if event.press_event is None:
                return
            modifiers = event.mouse_event.modifiers
            p1 = event.mouse_event.press_event.pos
            p2 = event.mouse_event.pos
            if 1 in event.buttons and not modifiers:
                self._update_rotation(event)
            elif 3 in event.buttons and not modifiers:
                norm = np.mean(self._viewbox.size)
                if self._event_value is None or len(self._event_value) == 2:
                    self._event_value = self.center
                distance = (p1 - p2) / norm * self._scale_factor
                distance[1] *= -1
                dx, dy, dz = self._dist_to_trans(distance)
                flip_factors = self._flip_factors
                up, forward, right = self._get_dim_vectors()
                dx, dy, dz = right * dx + forward * dy + up * dz
                dx, dy, dz = (
                    flip_factors[0] * dx,
                    flip_factors[1] * dy,
                    dz * flip_factors[2],
                )
                center = self._event_value
                self.center = center[0] + dx, center[1] + dy, center[2] + dz
            elif 3 in event.buttons and keys.SHIFT in modifiers:
                return


class Mip3DView(QtWidgets.QWidget):
    pointer_clicked = QtCore.Signal(object, object)
    hover_changed = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._enabled = os.environ.get("CHRONOPOSE_VIEWER_DISABLE_3D") != "1"
        self._scene = None
        self._canvas = None
        self._view = None
        self._volume = None
        self._mask_volume = None
        self._radius_volume = None
        self._mask_palette_rgba: np.ndarray | None = None
        self._radius_palette_rgba: np.ndarray | None = None
        self._mask_opacity = 0.7
        self._radius_opacity = 0.35
        self._radius_sphere_opacity = 0.9
        self._mask_data_key: object | None = None
        self._radius_data_key: object | None = None
        self._radius_sphere_key: tuple[object, ...] | None = None
        self._radius_sphere_visuals: list[object] = []
        self._cursor = None
        self._node_visuals: list[_NodeVisualData] = []
        self._edge_visuals: list[_EdgeVisualData] = []
        self._base_graph_visuals: list[object] = []
        self._highlight_visuals: list[object] = []
        self._node_position_lookup: dict[
            tuple[str, str],
            tuple[float, float, float],
        ] = {}
        self._edge_position_lookup: dict[
            tuple[str, Edge],
            tuple[tuple[float, float, float], tuple[float, float, float]],
        ] = {}
        self._graph_key: tuple[object, ...] | None = None
        self._highlight_key: tuple[object, ...] | None = None
        self._press_position: tuple[float, float] | None = None
        self._press_button: PointerButton | None = None
        self._hover_pick: PickResult | None = None
        self._shape_zyx = (1, 1, 1)
        self._volume_steps_zyx = (1, 1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        title = QtWidgets.QLabel("3D MIP · nearest")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        if not self._enabled:
            placeholder = QtWidgets.QLabel("3D rendering disabled for this process")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("background: #11151d; color: #aab2c0;")
            placeholder.setMinimumSize(260, 220)
            layout.addWidget(placeholder, 1)
            return

        try:
            from vispy import scene

            self._scene = scene
            self._canvas = scene.SceneCanvas(
                keys="interactive",
                bgcolor="#11151d",
                parent=self,
                show=False,
                vsync=True,
            )
            layout.addWidget(self._canvas.native, 1)
            self._view = self._canvas.central_widget.add_view(border_color="#303744")
            self._view.camera = MiddlePanTurntableCamera(
                fov=45,
                elevation=28,
                azimuth=38,
                up="-z",
            )
            self._canvas.events.mouse_press.connect(self._on_mouse_press)
            self._canvas.events.mouse_release.connect(self._on_mouse_release)
            self._canvas.events.mouse_move.connect(self._on_mouse_move)
        except Exception as error:
            self._enabled = False
            placeholder = QtWidgets.QLabel(f"3D renderer unavailable\n{error}")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("background: #11151d; color: #e09f3e;")
            layout.addWidget(placeholder, 1)

    @property
    def rendering_enabled(self) -> bool:
        return self._enabled

    def set_volume(
        self,
        volume_zyx: np.ndarray,
        levels: tuple[float, float],
        *,
        reset_camera: bool = False,
    ) -> None:
        if not self._enabled:
            return
        assert self._scene is not None and self._view is not None
        data = np.asarray(volume_zyx)
        self._shape_zyx = tuple(int(value) for value in data.shape)
        render_data, steps = self._prepare_volume(data)
        self._volume_steps_zyx = steps
        if self._volume is None:
            self._volume = self._scene.visuals.Volume(
                render_data,
                parent=self._view.scene,
                method="mip",
                cmap="grays",
                clim=levels,
                interpolation="nearest",
            )
        else:
            self._volume.set_data(render_data, clim=levels)
        from vispy.visuals.transforms import STTransform

        self._volume.transform = STTransform(scale=(steps[2], steps[1], steps[0]))
        if reset_camera:
            self.reset_camera()

    @staticmethod
    def _prepare_volume(
        volume: np.ndarray, maximum_axis: int = 256
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        steps = tuple(max(1, int(np.ceil(size / maximum_axis))) for size in volume.shape)
        data = volume[:: steps[0], :: steps[1], :: steps[2]]
        padding = tuple((0, max(0, 2 - size)) for size in data.shape)
        if any(after for _, after in padding):
            data = np.pad(data, padding, mode="edge")
        return np.ascontiguousarray(data, dtype=np.float32), steps

    def set_contrast(self, levels: tuple[float, float]) -> None:
        if self._volume is not None:
            self._volume.clim = levels
            self._canvas.update()

    def set_mask_overlay(
        self,
        components_zyx: np.ndarray,
        palette_rgba: np.ndarray,
        *,
        opacity: float,
        visible: bool,
        data_key: object | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert self._scene is not None and self._view is not None
        components = np.asarray(components_zyx)
        palette = np.asarray(palette_rgba, dtype=np.uint8)
        if components.ndim != 3:
            raise ValueError(f"Expected a ZYX component volume, got shape {components.shape}")
        if palette.ndim != 2 or palette.shape[1] != 4:
            raise ValueError("Mask palette must have shape (component_count + 1, 4)")

        previous_palette = self._mask_palette_rgba
        new_opacity = min(max(float(opacity), 0.0), 1.0)
        palette_changed = previous_palette is not palette
        opacity_changed = self._mask_opacity != new_opacity
        key = id(components_zyx) if data_key is None else data_key
        data_changed = self._mask_volume is None or key != self._mask_data_key
        palette_shrank = previous_palette is not None and len(palette) < len(previous_palette)
        if (
            (data_changed or palette_shrank)
            and components.size
            and int(np.max(components)) >= len(palette)
        ):
            raise ValueError("Mask component volume contains an index outside its palette")
        self._mask_palette_rgba = palette
        self._mask_opacity = new_opacity
        render_data = None
        steps = self._volume_steps_zyx
        if data_changed:
            render_data, steps = self._prepare_volume(components)
            self._mask_data_key = key

        maximum = max(1, len(palette) - 1)
        if self._mask_volume is None:
            assert render_data is not None
            self._mask_volume = self._scene.visuals.Volume(
                render_data,
                parent=self._view.scene,
                method="translucent",
                cmap=_categorical_colormap(palette, self._mask_opacity),
                clim=(0, maximum),
                interpolation="nearest",
                relative_step_size=1.0,
            )
            self._mask_volume.order = 0.5
            self._mask_volume.set_gl_state(
                "translucent",
                depth_test=False,
                cull_face=False,
            )
        else:
            if render_data is not None:
                self._mask_volume.set_data(render_data, clim=(0, maximum))
            elif palette_changed:
                self._mask_volume.clim = (0, maximum)
            if palette_changed or opacity_changed:
                self._mask_volume.cmap = _categorical_colormap(
                    palette,
                    self._mask_opacity,
                )

        if render_data is not None:
            from vispy.visuals.transforms import STTransform

            self._mask_volume.transform = STTransform(scale=(steps[2], steps[1], steps[0]))
        self._mask_volume.visible = bool(visible)
        if self._canvas is not None:
            self._canvas.update()

    def set_mask_opacity(self, opacity: float) -> None:
        new_opacity = min(max(float(opacity), 0.0), 1.0)
        if new_opacity == self._mask_opacity:
            return
        self._mask_opacity = new_opacity
        if self._mask_volume is None or self._mask_palette_rgba is None:
            return
        self._mask_volume.cmap = _categorical_colormap(
            self._mask_palette_rgba,
            self._mask_opacity,
        )
        if self._canvas is not None:
            self._canvas.update()

    def set_mask_visible(self, visible: bool) -> None:
        if self._mask_volume is not None:
            self._mask_volume.visible = bool(visible)
            if self._canvas is not None:
                self._canvas.update()

    def clear_mask_overlay(self) -> None:
        self._mask_palette_rgba = None
        self._mask_data_key = None
        if self._mask_volume is not None:
            self._mask_volume.visible = False
        if self._canvas is not None:
            self._canvas.update()

    def set_radius_overlay(
        self,
        components_zyx: np.ndarray,
        palette_rgba: np.ndarray,
        *,
        steps_zyx: tuple[int, int, int],
        opacity: float,
        visible: bool,
        data_key: object | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert self._scene is not None and self._view is not None
        components = np.asarray(components_zyx)
        palette = np.asarray(palette_rgba, dtype=np.uint8)
        if components.ndim != 3:
            raise ValueError(f"Expected a ZYX radius volume, got shape {components.shape}")
        if palette.ndim != 2 or palette.shape[1] != 4:
            raise ValueError("Radius palette must have shape (instance_count + 1, 4)")
        if len(steps_zyx) != 3 or any(int(step) <= 0 for step in steps_zyx):
            raise ValueError("Radius volume steps must contain three positive values")

        previous_palette = self._radius_palette_rgba
        new_opacity = min(max(float(opacity), 0.0), 1.0)
        palette_changed = previous_palette is not palette
        opacity_changed = self._radius_opacity != new_opacity
        key = id(components_zyx) if data_key is None else data_key
        data_changed = self._radius_volume is None or key != self._radius_data_key
        if data_changed and components.size and int(np.max(components)) >= len(palette):
            raise ValueError("Radius volume contains an index outside its palette")
        self._radius_palette_rgba = palette
        self._radius_opacity = new_opacity
        render_data = None
        if data_changed:
            padding = tuple((0, max(0, 2 - size)) for size in components.shape)
            if any(after for _, after in padding):
                components = np.pad(components, padding, mode="edge")
            render_data = np.ascontiguousarray(components, dtype=np.float32)
            self._radius_data_key = key

        maximum = max(1, len(palette) - 1)
        if self._radius_volume is None:
            assert render_data is not None
            self._radius_volume = self._scene.visuals.Volume(
                render_data,
                parent=self._view.scene,
                method="translucent",
                cmap=_categorical_colormap(palette, self._radius_opacity),
                clim=(0, maximum),
                interpolation="nearest",
                relative_step_size=1.0,
            )
            self._radius_volume.order = 0.75
            self._radius_volume.set_gl_state(
                "translucent",
                depth_test=False,
                cull_face=False,
            )
        else:
            if render_data is not None:
                self._radius_volume.set_data(render_data, clim=(0, maximum))
            elif palette_changed:
                self._radius_volume.clim = (0, maximum)
            if palette_changed or opacity_changed:
                self._radius_volume.cmap = _categorical_colormap(
                    palette,
                    self._radius_opacity,
                )

        if render_data is not None:
            from vispy.visuals.transforms import STTransform

            self._radius_volume.transform = STTransform(
                scale=(steps_zyx[2], steps_zyx[1], steps_zyx[0])
            )
        self._radius_volume.visible = bool(visible)
        if self._canvas is not None:
            self._canvas.update()

    def set_radius_opacity(self, opacity: float) -> None:
        new_opacity = min(max(float(opacity), 0.0), 1.0)
        if new_opacity == self._radius_opacity:
            return
        self._radius_opacity = new_opacity
        if self._radius_volume is None or self._radius_palette_rgba is None:
            return
        self._radius_volume.cmap = _categorical_colormap(
            self._radius_palette_rgba,
            self._radius_opacity,
        )
        if self._canvas is not None:
            self._canvas.update()

    def set_radius_visible(self, visible: bool) -> None:
        if self._radius_volume is not None:
            self._radius_volume.visible = bool(visible)
            if self._canvas is not None:
                self._canvas.update()

    def clear_radius_overlay(self) -> None:
        self._radius_palette_rgba = None
        self._radius_data_key = None
        if self._radius_volume is not None:
            self._radius_volume.visible = False
        if self._canvas is not None:
            self._canvas.update()

    def set_radius_spheres(
        self,
        graphs: dict[str, InstanceGraph],
        *,
        opacity: float,
        visible: bool,
    ) -> None:
        if not self._enabled:
            return
        assert self._scene is not None and self._view is not None
        new_opacity = min(max(float(opacity), 0.0), 1.0)
        geometry_key = tuple(
            (
                instance_id,
                tuple(
                    (node.id, node.position, node.radius)
                    for node in graph.nodes.values()
                    if node.radius > 0
                ),
            )
            for instance_id, graph in graphs.items()
        )
        if geometry_key != self._radius_sphere_key:
            self._clear_radius_sphere_visuals()
            positions_xyz: list[tuple[float, float, float]] = []
            diameters: list[float] = []
            for graph in graphs.values():
                for node in graph.nodes.values():
                    if node.radius <= 0:
                        continue
                    positions_xyz.append(
                        (
                            float(node.position[2]),
                            float(node.position[1]),
                            float(node.position[0]),
                        )
                    )
                    diameters.append(float(node.radius) * 2.0)

            if positions_xyz:
                positions = np.asarray(positions_xyz, dtype=np.float32)
                sizes = np.asarray(diameters, dtype=np.float32)
                for color, edge_width_relative, order in (
                    ((0.0, 0.0, 0.0), 0.18, 8),
                    ((1.0, 1.0, 1.0), 0.06, 9),
                ):
                    circles = self._scene.visuals.Markers(
                        parent=self._view.scene,
                        scaling="scene",
                        antialias=1.0,
                        alpha=new_opacity,
                    )
                    circles.set_data(
                        positions,
                        size=sizes,
                        face_color=(0.0, 0.0, 0.0, 0.0),
                        edge_color=(*color, 1.0),
                        edge_width_rel=edge_width_relative,
                    )
                    circles.order = order
                    self._make_overlay_visible(circles)
                    self._radius_sphere_visuals.append(circles)
            self._radius_sphere_key = geometry_key
            self._radius_sphere_opacity = new_opacity
        elif new_opacity != self._radius_sphere_opacity:
            self.set_radius_sphere_opacity(new_opacity)

        for visual in self._radius_sphere_visuals:
            visual.visible = bool(visible)
        if self._canvas is not None:
            self._canvas.update()

    def set_radius_sphere_opacity(self, opacity: float) -> None:
        new_opacity = min(max(float(opacity), 0.0), 1.0)
        if new_opacity == self._radius_sphere_opacity:
            return
        self._radius_sphere_opacity = new_opacity
        for visual in self._radius_sphere_visuals:
            visual.alpha = self._radius_sphere_opacity
        if self._canvas is not None:
            self._canvas.update()

    def set_radius_spheres_visible(self, visible: bool) -> None:
        for visual in self._radius_sphere_visuals:
            visual.visible = bool(visible)
        if self._canvas is not None:
            self._canvas.update()

    def clear_radius_spheres(self) -> None:
        self._clear_radius_sphere_visuals()
        if self._canvas is not None:
            self._canvas.update()

    def _clear_radius_sphere_visuals(self) -> None:
        for visual in self._radius_sphere_visuals:
            visual.parent = None
        self._radius_sphere_visuals.clear()
        self._radius_sphere_key = None

    def set_graphs(
        self,
        graphs: dict[str, InstanceGraph],
        *,
        active_instance: str | None,
        selected_node: tuple[str, str] | None,
        selected_edge: tuple[str, Edge] | None,
        action_source: tuple[str, str] | None = None,
        hovered: PickResult | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert self._scene is not None and self._view is not None
        graph_key = (
            active_instance,
            tuple(
                (
                    instance_id,
                    graph.color,
                    tuple((node_id, node.position) for node_id, node in graph.nodes.items()),
                    frozenset(graph.edges),
                )
                for instance_id, graph in graphs.items()
            ),
        )
        if graph_key == self._graph_key:
            self.set_highlights(
                selected_node=selected_node,
                selected_edge=selected_edge,
                action_source=action_source,
                hovered=hovered,
            )
            return

        self._clear_base_graph_visuals()
        edge_positions: dict[bool, list[tuple[float, float, float]]] = {
            False: [],
            True: [],
        }
        edge_colors: dict[bool, list[tuple[float, float, float, float]]] = {
            False: [],
            True: [],
        }
        edge_instance_ids: dict[bool, list[str]] = {False: [], True: []}
        edge_ids: dict[bool, list[Edge]] = {False: [], True: []}
        edge_pick_orders: dict[bool, list[int]] = {False: [], True: []}
        node_positions: list[tuple[float, float, float]] = []
        node_sizes: list[float] = []
        node_face_colors: list[tuple[float, float, float, float]] = []
        node_instance_ids: list[str] = []
        node_ids: list[str] = []
        edge_pick_order = 0

        for instance_id, graph in graphs.items():
            active = instance_id == active_instance
            rgba = _hex_to_rgba(graph.color, 1.0 if active else 0.55)
            for edge in sorted(graph.edges):
                first_position = graph.nodes[edge[0]].position
                second_position = graph.nodes[edge[1]].position
                positions = (
                    (
                        first_position[2],
                        first_position[1],
                        first_position[0],
                    ),
                    (
                        second_position[2],
                        second_position[1],
                        second_position[0],
                    ),
                )
                edge_positions[active].extend(positions)
                edge_colors[active].extend((rgba, rgba))
                edge_instance_ids[active].append(instance_id)
                edge_ids[active].append(edge)
                edge_pick_orders[active].append(edge_pick_order)
                edge_pick_order += 1
                self._edge_position_lookup[(instance_id, edge)] = positions

            for node_id, node in graph.nodes.items():
                position = (
                    node.position[2],
                    node.position[1],
                    node.position[0],
                )
                node_positions.append(position)
                node_sizes.append(11 if active else 8)
                node_face_colors.append(rgba)
                node_instance_ids.append(instance_id)
                node_ids.append(node_id)
                self._node_position_lookup[(instance_id, node_id)] = position

        for active in (False, True):
            if not edge_ids[active]:
                continue
            positions = np.asarray(edge_positions[active], dtype=np.float32)
            colors = np.asarray(edge_colors[active], dtype=np.float32)
            edge_halo = self._scene.visuals.Line(
                pos=positions,
                color=(0.015, 0.02, 0.03, 0.95),
                width=7 if active else 6,
                connect="segments",
                method="gl",
                antialias=True,
                parent=self._view.scene,
            )
            edge_visual = self._scene.visuals.Line(
                pos=positions,
                color=colors,
                width=3.5 if active else 2.5,
                connect="segments",
                method="gl",
                antialias=True,
                parent=self._view.scene,
            )
            edge_halo.order = 1
            edge_visual.order = 2
            self._make_overlay_visible(edge_halo)
            self._make_overlay_visible(edge_visual)
            self._base_graph_visuals.extend((edge_halo, edge_visual))
            self._edge_visuals.append(
                _EdgeVisualData(
                    edge_visual,
                    edge_instance_ids[active],
                    edge_ids[active],
                    positions,
                    edge_pick_orders[active],
                )
            )

        if node_ids:
            positions = np.asarray(node_positions, dtype=np.float32)
            node_visual = self._scene.visuals.Markers(parent=self._view.scene)
            node_visual.set_data(
                positions,
                size=np.asarray(node_sizes, dtype=np.float32),
                face_color=np.asarray(node_face_colors, dtype=np.float32),
                edge_color=np.tile(
                    np.asarray((0.02, 0.025, 0.04, 1.0), dtype=np.float32),
                    (len(node_ids), 1),
                ),
                edge_width=2.5,
            )
            node_visual.order = 4
            self._make_overlay_visible(node_visual)
            self._base_graph_visuals.append(node_visual)
            self._node_visuals.append(
                _NodeVisualData(
                    node_visual,
                    node_instance_ids,
                    node_ids,
                    positions,
                )
            )

        self._graph_key = graph_key
        self.set_highlights(
            selected_node=selected_node,
            selected_edge=selected_edge,
            action_source=action_source,
            hovered=hovered,
        )

    def set_highlights(
        self,
        *,
        selected_node: tuple[str, str] | None,
        selected_edge: tuple[str, Edge] | None,
        action_source: tuple[str, str] | None = None,
        hovered: PickResult | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert self._scene is not None and self._view is not None
        hovered_identity = None if hovered is None else hovered.identity
        highlight_key = (
            selected_node,
            selected_edge,
            action_source,
            hovered_identity,
        )
        if highlight_key == self._highlight_key:
            return
        self._clear_highlight_visuals()

        edge_highlights = (
            (
                None
                if hovered is None or hovered.kind != "edge" or hovered.edge is None
                else (hovered.instance_id, hovered.edge),
                (0.463, 1.0, 0.012, 1.0),
                9,
                3,
            ),
            (selected_edge, (0.0, 0.898, 1.0, 1.0), 5, 4),
        )
        for identity, color, width, order in edge_highlights:
            if identity is None:
                continue
            positions = self._edge_position_lookup.get(identity)
            if positions is None:
                continue
            emphasis = self._scene.visuals.Line(
                pos=np.asarray(positions, dtype=np.float32),
                color=color,
                width=width,
                connect="segments",
                method="gl",
                antialias=True,
                parent=self._view.scene,
            )
            emphasis.order = order
            self._make_overlay_visible(emphasis)
            self._highlight_visuals.append(emphasis)

        identities = {
            "selected": selected_node,
            "action_source": action_source,
            "hovered": (
                None
                if hovered is None or hovered.kind != "node" or hovered.node_id is None
                else (hovered.instance_id, hovered.node_id)
            ),
        }
        for order, style in enumerate(NODE_EMPHASIS_STYLES, start=5):
            identity = identities[style.role]
            if identity is None:
                continue
            position = self._node_position_lookup.get(identity)
            if position is None:
                continue
            emphasis = self._scene.visuals.Markers(parent=self._view.scene)
            emphasis.set_data(
                np.asarray(position, dtype=np.float32)[np.newaxis, :],
                size=style.size_3d,
                face_color=(0.0, 0.0, 0.0, 0.0),
                edge_color=_hex_to_rgba(style.color),
                edge_width=style.width,
            )
            emphasis.order = order
            self._make_overlay_visible(emphasis)
            self._highlight_visuals.append(emphasis)
        self._highlight_key = highlight_key
        if self._canvas is not None:
            self._canvas.update()

    def set_cursor(self, position_zyx: tuple[float, float, float]) -> None:
        if not self._enabled:
            return
        assert self._scene is not None and self._view is not None
        position_xyz = np.asarray(
            [[position_zyx[2], position_zyx[1], position_zyx[0]]], dtype=np.float32
        )
        if self._cursor is None:
            self._cursor = self._scene.visuals.Markers(parent=self._view.scene)
        self._cursor.set_data(
            position_xyz,
            size=10,
            face_color=(1.0, 1.0, 0.2, 0.55),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            edge_width=1.0,
        )
        self._cursor.order = 10
        self._make_overlay_visible(self._cursor)

    def reset_camera(self) -> None:
        if not self._enabled or self._view is None:
            return
        z_size, y_size, x_size = self._shape_zyx
        self._view.camera.set_range(
            x=(-0.5, x_size - 0.5),
            y=(-0.5, y_size - 0.5),
            z=(-0.5, z_size - 0.5),
            margin=0.08,
        )

    def _clear_base_graph_visuals(self) -> None:
        self._clear_highlight_visuals()
        for visual in self._base_graph_visuals:
            visual.parent = None
        self._base_graph_visuals.clear()
        self._node_visuals.clear()
        self._edge_visuals.clear()
        self._node_position_lookup.clear()
        self._edge_position_lookup.clear()
        self._graph_key = None

    def _clear_highlight_visuals(self) -> None:
        for visual in self._highlight_visuals:
            visual.parent = None
        self._highlight_visuals.clear()
        self._highlight_key = None

    def _on_mouse_press(self, event: object) -> None:
        buttons = {1: PointerButton.LEFT, 2: PointerButton.RIGHT}
        button = buttons.get(event.button)
        if button is not None:
            self._press_position = (float(event.pos[0]), float(event.pos[1]))
            self._press_button = button

    def _on_mouse_release(self, event: object) -> None:
        buttons = {1: PointerButton.LEFT, 2: PointerButton.RIGHT}
        button = buttons.get(event.button)
        if button is None or button != self._press_button or self._press_position is None:
            return
        release = (float(event.pos[0]), float(event.pos[1]))
        moved = float(
            np.hypot(release[0] - self._press_position[0], release[1] - self._press_position[1])
        )
        self._press_position = None
        self._press_button = None
        if moved > 5:
            return
        self.pointer_clicked.emit(button, self._pick(release))

    def _on_mouse_move(self, event: object) -> None:
        if event.buttons:
            return
        self._set_hover(self._pick((float(event.pos[0]), float(event.pos[1]))))

    def _pick(self, canvas_position: tuple[float, float]) -> PickResult | None:
        node_candidates: list[tuple[float, _NodeVisualData, int]] = []
        for item in self._node_visuals:
            try:
                projected = self._project_to_canvas(item.visual, item.positions_xyz)
            except Exception:
                continue
            distances = np.hypot(
                projected[:, 0] - canvas_position[0],
                projected[:, 1] - canvas_position[1],
            )
            candidates = np.flatnonzero(distances <= 15)
            if candidates.size:
                index = int(candidates[np.argmin(distances[candidates])])
                node_candidates.append((float(distances[index]), item, index))
        if node_candidates:
            _, item, index = min(node_candidates, key=lambda candidate: candidate[0])
            return PickResult(
                "node",
                item.instance_ids[index],
                node_id=item.node_ids[index],
            )

        edge_candidates: list[tuple[float, int, _EdgeVisualData, int, float]] = []
        for item in self._edge_visuals:
            try:
                projected = self._project_to_canvas(item.visual, item.positions_xyz)
            except Exception:
                continue
            first = projected[::2, :2]
            direction = projected[1::2, :2] - first
            point = np.asarray(canvas_position)
            denominator = np.einsum("ij,ij->i", direction, direction)
            numerator = np.einsum("ij,ij->i", point - first, direction)
            fractions = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0,
            )
            np.clip(fractions, 0.0, 1.0, out=fractions)
            closest = first + fractions[:, np.newaxis] * direction
            distances = np.hypot(
                point[0] - closest[:, 0],
                point[1] - closest[:, 1],
            )
            candidates = np.flatnonzero(distances <= 8)
            if candidates.size:
                index = int(candidates[np.argmin(distances[candidates])])
                edge_candidates.append(
                    (
                        float(distances[index]),
                        item.pick_orders[index],
                        item,
                        index,
                        float(fractions[index]),
                    )
                )
        if edge_candidates:
            _, _, item, index, fraction = min(
                edge_candidates,
                key=lambda candidate: (candidate[0], candidate[1]),
            )
            return PickResult(
                "edge",
                item.instance_ids[index],
                edge=item.edges[index],
                fraction=fraction,
            )
        return None

    def _set_hover(self, pick: PickResult | None) -> None:
        old_identity = None if self._hover_pick is None else self._hover_pick.identity
        new_identity = None if pick is None else pick.identity
        if old_identity == new_identity:
            return
        self._hover_pick = pick
        self.hover_changed.emit(pick)

    def reset_hover_tracking(self) -> None:
        self._hover_pick = None

    @staticmethod
    def _make_overlay_visible(visual: object) -> None:
        visual.set_gl_state(
            "translucent",
            depth_test=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

    @staticmethod
    def _project_to_canvas(visual: object, positions_xyz: np.ndarray) -> np.ndarray:
        transform = visual.get_transform("visual", "canvas")
        projected = np.asarray(transform.map(positions_xyz), dtype=float)
        if projected.ndim != 2 or projected.shape[1] < 2:
            raise ValueError("Unexpected VisPy projection result")
        if projected.shape[1] >= 4:
            homogeneous = projected[:, 3]
            if np.any(np.isclose(homogeneous, 0)):
                raise ValueError("Cannot project a point on the camera plane")
            projected = projected.copy()
            projected[:, :3] /= homogeneous[:, np.newaxis]
        return projected
