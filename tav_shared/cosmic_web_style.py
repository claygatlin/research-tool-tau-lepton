"""
Visual style matching Tav-Superblock Conformal Exhaustion visualizer.

Palette: deep space background, cyan-blue cool nodes/web, magenta choke lines.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Conformal Exhaustion reference colors
BG_DARK = "#050510"
CYAN_GLOW = "#00d2ff"
BLUE_WEB = "#0088ff"
BLUE_DEEP = "#004488"
BLUE_MIST = "#1a6fcc"
RESONANCE_RED = "#ff0055"
TRANSITION_AMBER = "#ffaa00"
TEXT_ACCENT = "#00d2ff"
TEXT_MUTED = "#8899aa"
PANEL_BG = "rgba(10,15,30,0.85)"
PANEL_BORDER = "#334466"

CHOKE_HIGHLIGHT_THRESHOLD = 0.55
CHOKE_VISUAL_THRESHOLD = 0.38
EXHAUSTION_RADIUS_MPC = 7.0
TAU_LINE_WIDTH = 2.6


def cool_filament_rgb(choke: float) -> str:
    """Cool 3D-supported filaments: cyan → deep blue (low choking)."""
    cool = float(max(0.0, 1.0 - choke))
    r = int(0 + 30 * (1 - cool))
    g = int(100 + 110 * cool)
    b = int(180 + 75 * cool)
    return f"rgb({r},{g},{b})"


def choke_line_rgb(choke: float, threshold: float = CHOKE_HIGHLIGHT_THRESHOLD) -> str:
    """142857 resonance / 1D exhaustion: magenta-red choke lines."""
    t = (choke - threshold) / max(1.0 - threshold, 1e-6)
    t = max(0.0, min(1.0, t))
    r = int(255)
    g = int(20 + 70 * (1 - t))
    b = int(60 + 25 * (1 - t))
    return f"rgb({r},{g},{b})"


def _hidden_axis(**extra) -> dict:
    """3D axis with no grid lines — clean dark void background."""
    base = dict(
        showgrid=False,
        gridwidth=0,
        zeroline=False,
        showline=False,
        showticklabels=False,
        ticks="",
        showbackground=True,
        backgroundcolor=BG_DARK,
        title="",
    )
    base.update(extra)
    return base


def apply_exhaustion_layout(
    fig: go.Figure,
    title_html: str,
    subtitle_html: str = "",
    camera: dict | None = None,
) -> go.Figure:
    """Apply Conformal Exhaustion dark scene styling (no grid)."""
    if subtitle_html:
        title_text = f"<b>{title_html}</b><br><span style='font-size:12px;color:{TEXT_MUTED}'>{subtitle_html}</span>"
    else:
        title_text = f"<b>{title_html}</b>"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color=TEXT_ACCENT), x=0.5),
        scene=dict(
            xaxis=_hidden_axis(),
            yaxis=_hidden_axis(),
            zaxis=_hidden_axis(),
            aspectmode="cube",
            bgcolor=BG_DARK,
            camera=camera or dict(eye=dict(x=1.45, y=1.45, z=0.95)),
        ),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font=dict(color="#ddeeff", size=10),
        legend=dict(
            x=0.01,
            y=0.97,
            bgcolor=PANEL_BG,
            bordercolor=PANEL_BORDER,
            font=dict(size=9, color=CYAN_GLOW),
        ),
        margin=dict(l=0, r=0, t=70, b=12),
        hoverlabel=dict(bgcolor="rgba(10,20,40,0.92)", font_size=10, font_color=CYAN_GLOW),
    )
    return fig


def add_node_glow_traces(
    fig: go.Figure,
    nodes,
    name: str = "Cluster Nodes (Conformal Wells)",
    size: float = 5.5,
    hovertemplate: str | None = None,
) -> None:
    """Cyan node markers with a soft under-glow (Conformal Exhaustion look)."""
    hovertemplate = hovertemplate or (
        "<b>Cluster Node</b><br>"
        "Local conformal gravity well<br>"
        "(%{x:.1f}, %{y:.1f}, %{z:.1f}) h⁻¹ Mpc<extra></extra>"
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=nodes[:, 2],
            mode="markers",
            marker=dict(size=size * 2.2, color=CYAN_GLOW, opacity=0.14, symbol="circle"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=nodes[:, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=CYAN_GLOW,
                opacity=0.92,
                symbol="circle",
                line=dict(width=0.6, color="#e8f7ff"),
            ),
            name=name,
            hovertemplate=hovertemplate,
        )
    )


def add_void_mist(fig: go.Figure, void_points, name: str = "Cool Voids (3D supported)") -> None:
    fig.add_trace(
        go.Scatter3d(
            x=void_points[:, 0],
            y=void_points[:, 1],
            z=void_points[:, 2],
            mode="markers",
            marker=dict(size=2.0, color=BLUE_WEB, opacity=0.18),
            name=name,
            hoverinfo="skip",
        )
    )


def _hex_ring_vertices(center: np.ndarray, radius: float, normal: np.ndarray) -> np.ndarray:
    """Six vertices of a hex ring (+ closing point) centered on `center`."""
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    nlen = np.linalg.norm(normal)
    if nlen < 1e-8:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / nlen

    if abs(normal[2]) < 0.9:
        tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        tangent = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    angles = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=True)
    verts = []
    for angle in angles:
        offset = radius * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
        verts.append(center + offset)
    return np.asarray(verts)


def _nearest_node_index(nodes: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(nodes - point, axis=1)))


def _node_ring_normals(nodes: np.ndarray, connections: list[tuple]) -> list[np.ndarray]:
    """Orient each node's hex ring perpendicular to its local web links."""
    nodes = np.asarray(nodes, dtype=float)
    n_nodes = len(nodes)
    accum = np.zeros((n_nodes, 3), dtype=float)

    for start, end in connections:
        i = _nearest_node_index(nodes, np.asarray(start, dtype=float))
        j = _nearest_node_index(nodes, np.asarray(end, dtype=float))
        if i == j:
            continue
        direction = nodes[j] - nodes[i]
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length
        accum[i] += direction
        accum[j] -= direction

    normals = []
    for vec in accum:
        if np.linalg.norm(vec) < 1e-6:
            normals.append(np.array([0.0, 0.0, 1.0]))
        else:
            normals.append(vec / np.linalg.norm(vec))
    return normals


def add_cluster_hex_rings(
    fig: go.Figure,
    nodes,
    connections: list[tuple] | None = None,
    radius: float = EXHAUSTION_RADIUS_MPC,
    *,
    name: str = "τ-Hex Ring (7 h⁻¹ Mpc per node)",
    show_legend: bool = True,
) -> None:
    """One hexagonal τ-circle ring centered on each cluster node."""
    nodes = np.asarray(nodes, dtype=float)
    connections = connections or []
    normals = _node_ring_normals(nodes, connections)

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for node, normal in zip(nodes, normals):
        ring = _hex_ring_vertices(node, radius, normal)
        xs.extend(ring[:, 0].tolist())
        ys.extend(ring[:, 1].tolist())
        zs.extend(ring[:, 2].tolist())
        xs.append(None)
        ys.append(None)
        zs.append(None)

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=RESONANCE_RED, width=TAU_LINE_WIDTH),
            opacity=0.9,
            name=name,
            hovertemplate=(
                "<b>τ-Hex Ring</b><br>"
                "7 h⁻¹ Mpc conformal exhaustion boundary<br>"
                "Centered on cluster node<extra></extra>"
            ),
            showlegend=show_legend,
        )
    )


def add_inter_node_force_lines(
    fig: go.Figure,
    connections: list[tuple],
    *,
    name: str = "142857 Force Lines (node links)",
    show_legend: bool = True,
) -> None:
    """Red 142857 resonance force lines linking connected cluster nodes."""
    if not connections:
        return

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for start, end in connections:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        xs.extend([start[0], end[0], None])
        ys.extend([start[1], end[1], None])
        zs.extend([start[2], end[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=RESONANCE_RED, width=TAU_LINE_WIDTH),
            opacity=0.9,
            name=name,
            hovertemplate=(
                "<b>142857 Force Line</b><br>"
                "τ-circle resonance link between cluster nodes<extra></extra>"
            ),
            showlegend=show_legend,
        )
    )


def add_choke_thread(
    fig: go.Figure,
    start,
    end,
    choke: float,
    *,
    name: str = "Choke Lines (142857 Resonance)",
    hovertemplate: str | None = None,
    show_legend: bool = False,
    threshold: float = CHOKE_VISUAL_THRESHOLD,
) -> None:
    """Bright magenta-red choke filament with glow underlay for visibility in 3D."""
    if choke <= threshold:
        return

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t = (choke - threshold) / max(1.0 - threshold, 1e-6)
    t = float(max(0.0, min(1.0, t)))
    core_width = 4.0 + 8.0 * t
    glow_width = core_width * 2.4

    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=RESONANCE_RED, width=glow_width),
            opacity=0.28 + 0.22 * t,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=RESONANCE_RED, width=core_width),
            opacity=0.98,
            name=name,
            hovertemplate=hovertemplate,
            showlegend=show_legend,
        )
    )