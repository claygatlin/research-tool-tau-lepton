#!/usr/bin/env python3
"""
3D Cosmic Web Simulation with Tav-Superblock Physics
===================================================

Interactive 3D visualization of the cosmic web incorporating:

- Dense cluster nodes (gravitational wells) connected by filaments
- Base filament layer + highlighted choke lines (1D hot threads)
- Blue void haze in low-density regions
- Field galaxies scattered for texture
- Tav-Superblock choking, DM decoupling, and topological friction

Run:
    python webSim.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from tav_shared.cosmic_web_style import (
    BLUE_WEB,
    CHOKE_HIGHLIGHT_THRESHOLD,
    CHOKE_VISUAL_THRESHOLD,
    CYAN_GLOW,
    EXHAUSTION_RADIUS_MPC,
    PANEL_BG,
    PANEL_BORDER,
    RESONANCE_RED,
    TEXT_ACCENT,
    TEXT_MUTED,
    add_choke_thread,
    add_cluster_hex_rings,
    add_inter_node_force_lines,
    add_node_glow_traces,
    add_void_mist,
    apply_exhaustion_layout,
    cool_filament_rgb,
)

# ============================================================
# TAV-SUPERBLOCK PHYSICS
# ============================================================


def dm_baryon_ratio(thickness_mpc, critical_thickness=0.35, alpha=2.8):
    if thickness_mpc >= critical_thickness:
        return 5.2
    return 5.2 * (thickness_mpc / critical_thickness) ** alpha


def whim_temperature(thickness_mpc, base_T=2.5e5, friction_strength=1.8e6):
    friction = friction_strength / (thickness_mpc ** 1.8 + 0.02)
    return base_T + friction


def coupling_coefficient(thickness_mpc, critical=0.35):
    return min(1.0, (thickness_mpc / critical) ** 1.5)


def choking_state(thickness):
    """0 = fully 3D cool, 1 = strongly 1D hot/choked."""
    return 1.0 - coupling_coefficient(thickness)


def _filament_physics(thickness: float) -> dict:
    return {
        "thickness": thickness,
        "dm_ratio": dm_baryon_ratio(thickness),
        "temperature": whim_temperature(thickness),
        "choking": choking_state(thickness),
        "coupling": coupling_coefficient(thickness),
    }


# ============================================================
# GENERATE 3D COSMIC WEB
# ============================================================

def _place_major_nodes(
    n_nodes: int,
    box_size: float,
    min_sep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Poisson-style placement of dense web knots with minimum separation."""
    nodes = []
    attempts = 0
    margin = 18.0
    max_attempts = n_nodes * 400
    while len(nodes) < n_nodes and attempts < max_attempts:
        candidate = rng.uniform(margin, box_size - margin, size=3)
        if all(np.linalg.norm(candidate - existing) >= min_sep for existing in nodes):
            nodes.append(candidate)
        attempts += 1
    if len(nodes) < n_nodes:
        raise RuntimeError(f"Could only place {len(nodes)} major nodes (requested {n_nodes}).")
    return np.asarray(nodes)


def _assign_filament_thickness(rng: np.random.Generator, length: float) -> float:
    """
    Longer inter-node spans tend toward thinner (more choked) segments;
    shorter spans stay thicker (3D-supported).
    """
    if rng.random() < 0.42:
        thickness = rng.uniform(0.08, 0.24)
    else:
        thickness = rng.normal(0.55, 0.18)
    length_bias = 0.85 - 0.25 * min(length / 70.0, 1.0)
    thickness *= length_bias
    return float(np.clip(thickness, 0.08, 1.8))


def _sample_void_points(
    major_nodes: np.ndarray,
    filaments: list[dict],
    box_size: float,
    n_void: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place faint void haze preferentially in empty regions away from knots and filaments."""
    accepted = []
    margin = 8.0
    attempts = 0
    max_attempts = n_void * 80

    while len(accepted) < n_void and attempts < max_attempts:
        point = rng.uniform(margin, box_size - margin, size=3)
        node_dists = np.linalg.norm(major_nodes - point, axis=1)
        if node_dists.min() < 22.0:
            attempts += 1
            continue

        too_close = False
        for filament in filaments:
            start, end = filament["start"], filament["end"]
            seg = end - start
            seg_len_sq = np.dot(seg, seg)
            if seg_len_sq < 1e-6:
                continue
            t = np.clip(np.dot(point - start, seg) / seg_len_sq, 0.0, 1.0)
            closest = start + t * seg
            if np.linalg.norm(point - closest) < 10.0:
                too_close = True
                break
        if too_close:
            attempts += 1
            continue

        accepted.append(point)
        attempts += 1

    if len(accepted) < n_void // 3:
        fallback = rng.uniform(margin, box_size - margin, size=(n_void, 3))
        return fallback
    return np.asarray(accepted)


def generate_cosmic_web(
    n_major_nodes: int = 16,
    n_field_galaxies: int = 140,
    box_size: float = 180.0,
    linking_length: float = 62.0,
    min_node_sep: float = 38.0,
    seed: int = 42,
):
    """
    Build a cosmic-web-like graph: dense nodes first, filaments between them,
    voids in the gaps, plus scattered field galaxies.
    """
    rng = np.random.default_rng(seed)

    major_nodes = _place_major_nodes(n_major_nodes, box_size, min_node_sep, rng)

    filaments = []
    n_nodes = len(major_nodes)
    for i in range(n_nodes):
        neighbors = []
        for j in range(n_nodes):
            if i == j:
                continue
            dist = float(np.linalg.norm(major_nodes[i] - major_nodes[j]))
            if 4.0 < dist < linking_length:
                neighbors.append((j, dist))
        neighbors.sort(key=lambda item: item[1])

        for j, dist in neighbors[:4]:
            if i >= j:
                continue
            thickness = _assign_filament_thickness(rng, dist)
            physics = _filament_physics(thickness)
            filaments.append(
                {
                    "start": major_nodes[i],
                    "end": major_nodes[j],
                    "length": dist,
                    **physics,
                }
            )

    field_galaxies = rng.uniform(10, box_size - 10, size=(n_field_galaxies, 3))
    void_points = _sample_void_points(major_nodes, filaments, box_size, n_void=320, rng=rng)

    return major_nodes, field_galaxies, filaments, void_points, box_size


# ============================================================
# BUILD INTERACTIVE 3D FIGURE
# ============================================================

def _filament_hover(f: dict, label: str) -> str:
    thick = f["thickness"]
    choke = f["choking"]
    dm_r = f["dm_ratio"]
    temp = f["temperature"]
    state = (
        "CHOKE LINE (1D Hot)"
        if choke > CHOKE_HIGHLIGHT_THRESHOLD
        else "transitional"
        if choke > 0.3
        else "3D supported cool"
    )
    dm_note = (
        "DM ratio dropping (shadow decoupling)"
        if dm_r < 4.5
        else "Near full 5.2 shadow coupling"
    )
    temp_note = (
        "High WHIM T (topological friction)"
        if temp > 6.0e5
        else "Moderate WHIM T"
    )
    return (
        f"<b>{label}</b><br>"
        f"Transverse Thickness: {thick:.2f} Mpc<br>"
        f"DM/Baryon Ratio: {dm_r:.2f} — {dm_note}<br>"
        f"WHIM Temperature: {temp/1e5:.2f} × 10⁵ K — {temp_note}<br>"
        f"Coupling κ: {f['coupling']:.3f}<br>"
        f"Choking State: {choke:.2f} ({state})<br>"
        f"<i>7 h⁻¹ Mpc conformal scale active</i><extra></extra>"
    )


def create_3d_cosmic_web():
    major_nodes, field_galaxies, filaments, void_points, box_size = generate_cosmic_web()

    fig = go.Figure()
    # --- Cool void haze (3D-supported regions) ---
    add_void_mist(fig, void_points, name="Cool Voids (3D supported)")

    # --- Base filament layer: cyan-blue conformal web (dim where choked) ---
    for f in filaments:
        start, end = f["start"], f["end"]
        choke = f["choking"]
        cool = 1.0 - choke
        if choke > CHOKE_VISUAL_THRESHOLD:
            opacity = 0.06
            width = 0.8
        else:
            opacity = 0.22 + 0.38 * cool
            width = 1.4 + 1.2 * cool
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=cool_filament_rgb(choke), width=width),
                opacity=opacity,
                name="Cosmic Web (Conformal Fabric)",
                hovertemplate=_filament_hover(f, "Filament Segment"),
                showlegend=False,
            )
        )

    # --- Field galaxies (subtle blue texture) ---
    fig.add_trace(
        go.Scatter3d(
            x=field_galaxies[:, 0],
            y=field_galaxies[:, 1],
            z=field_galaxies[:, 2],
            mode="markers",
            marker=dict(
                size=2.0,
                color=BLUE_WEB,
                opacity=0.32,
                symbol="circle",
            ),
            name="Field Galaxies",
            hovertemplate="<b>Field Galaxy</b><br>(%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>",
        )
    )

    # --- Major cluster nodes: cyan conformal wells with glow ---
    add_node_glow_traces(
        fig,
        major_nodes,
        name="Dense Cluster Nodes (Conformal Wells)",
        size=9.0,
        hovertemplate=(
            "<b>Conformal Gravity Well</b><br>"
            "Local 5D spacetime binding<br>"
            "(%{x:.1f}, %{y:.1f}, %{z:.1f}) h⁻¹ Mpc<extra></extra>"
        ),
    )

    node_links = [(f["start"], f["end"]) for f in filaments]

    # --- τ-hex ring on every cluster node (centered on the well) ---
    add_cluster_hex_rings(
        fig,
        major_nodes,
        connections=node_links,
        radius=EXHAUSTION_RADIUS_MPC,
    )

    # --- 142857 force lines linking connected nodes ---
    add_inter_node_force_lines(fig, node_links)

    # --- Choke threads: 142857 resonance / 1D exhaustion (drawn on top) ---
    choke_legend_shown = False
    for f in filaments:
        if f["choking"] <= CHOKE_VISUAL_THRESHOLD:
            continue
        label = (
            "CHOKE LINE (1D Hot)"
            if f["choking"] > CHOKE_HIGHLIGHT_THRESHOLD
            else "Transitional Choke Thread"
        )
        add_choke_thread(
            fig,
            f["start"],
            f["end"],
            f["choking"],
            name="Choke Lines (142857 Resonance)",
            hovertemplate=_filament_hover(f, label),
            show_legend=not choke_legend_shown,
        )
        choke_legend_shown = True

    choke_count = sum(1 for f in filaments if f["choking"] > CHOKE_VISUAL_THRESHOLD)

    apply_exhaustion_layout(
        fig,
        "Tav-Superblock 3D Cosmic Web",
        "7 h⁻¹ Mpc Conformal Exhaustion • Local Gravity Wells • 142857 Resonance Choking",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
    )

    fig.add_annotation(
        text=(
            "<b>Conformal Exhaustion Palette</b><br>"
            f"<span style='color:{CYAN_GLOW}'>Cyan glow</span> = conformal gravity wells<br>"
            f"<span style='color:{BLUE_WEB}'>Blue web</span> = 3D-supported cool filaments<br>"
            f"<span style='color:{RESONANCE_RED}'>Magenta-red</span> = τ-hex per node + inter-node force lines<br>"
            f"<span style='color:{RESONANCE_RED}'>Bright red</span> = choke threads (≥ {CHOKE_VISUAL_THRESHOLD})<br>"
            f"<span style='color:{TEXT_MUTED}'>Blue mist</span> = void regions"
        ),
        x=0.99,
        y=0.95,
        xref="paper",
        yref="paper",
        align="right",
        font=dict(size=10, color=TEXT_ACCENT),
        bgcolor=PANEL_BG,
        bordercolor=PANEL_BORDER,
        borderwidth=1,
        showarrow=False,
    )

    fig.add_annotation(
        text=(
            f"Kill Switches: {choke_count} choke lines visible — "
            "thin filaments must show dropping DM ratio (Euclid/Rubin) and "
            "highest WHIM T in narrowest threads (XRISM)"
        ),
        x=0.5,
        y=0.02,
        xref="paper",
        yref="paper",
        align="center",
        font=dict(size=9, color="#ffaa00", style="italic"),
        showarrow=False,
    )

    return fig


if __name__ == "__main__":
    print("Generating Tav-Superblock 3D Cosmic Web simulation...")
    fig = create_3d_cosmic_web()
    fig.show()
    print(
        "\nInteractive 3D figure opened. Rotate/zoom with mouse. "
        "Hover choke lines (bright red) for 1D hot physics values."
    )