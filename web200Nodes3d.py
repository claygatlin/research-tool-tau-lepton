#!/usr/bin/env python3
"""
Tav-Superblock 3D Cosmic Web — ~200 Nodes
==========================================

A large-scale but still diagrammatic 3D visualization of the cosmic web
with approximately 200 nodes, styled similarly to the single-node
conformal exhaustion diagram.

Features:
- ~200 galactic cluster nodes arranged with realistic clustering
- Filament network showing the transition across the 7 h⁻¹ Mpc scale
- Color gradient: blue (cool 3D-supported) → red (hot 1D choked)
- Prominent bright "choke lines" where topological friction and DM decoupling are strongest
- Hover physics on filaments (DM ratio, temperature, choking state)
- Clean dark aesthetic matching the single-node explanatory style

Run:
    python cosmic_web_200_nodes_tav_superblock_3d.py
"""

import numpy as np
import plotly.graph_objects as go

from tav_shared.cosmic_web_style import (
    CHOKE_HIGHLIGHT_THRESHOLD,
    CHOKE_VISUAL_THRESHOLD,
    CYAN_GLOW,
    EXHAUSTION_RADIUS_MPC,
    PANEL_BG,
    PANEL_BORDER,
    RESONANCE_RED,
    TEXT_ACCENT,
    add_choke_thread,
    add_cluster_hex_rings,
    add_inter_node_force_lines,
    add_node_glow_traces,
    apply_exhaustion_layout,
    cool_filament_rgb,
)

# ============================================================
# PHYSICS FUNCTIONS
# ============================================================

def dm_baryon_ratio(thickness, critical=0.35, alpha=2.8):
    if thickness >= critical:
        return 5.2
    return 5.2 * (thickness / critical) ** alpha

def whim_temperature(thickness, base=2.5e5, strength=1.8e6):
    return base + strength / (thickness**1.8 + 0.02)

def choking_state(thickness, critical=0.35):
    return 1.0 - min(1.0, (thickness / critical)**1.5)

# ============================================================
# GENERATE ~200 NODES + FILAMENTS
# ============================================================

def generate_web(n_nodes=200, box_size=220.0, linking_length=15.0, seed=42):
    np.random.seed(seed)
    
    # Create several supercluster centers
    n_super = 8
    super_centers = np.random.uniform(20, box_size-20, size=(n_super, 3))
    
    nodes = []
    for center in super_centers:
        # Nodes clustered around each supercenter
        n_local = np.random.randint(18, 28)
        local = center + np.random.normal(0, 22, size=(n_local, 3))
        local = np.clip(local, 5, box_size-5)
        nodes.append(local)
    
    nodes = np.vstack(nodes)
    
    # Add some isolated field nodes
    n_field = n_nodes - len(nodes)
    if n_field > 0:
        field = np.random.uniform(8, box_size-8, size=(n_field, 3))
        nodes = np.vstack([nodes, field])
    
    nodes = nodes[:n_nodes]  # ensure exact count
    
    # Build filaments (distance-based)
    filaments = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = np.linalg.norm(nodes[i] - nodes[j])
            if 3.0 < dist < linking_length:
                # Long inter-cluster spans thin out and choke; dense cores stay thick.
                local_density = np.sum(np.linalg.norm(nodes - nodes[i], axis=1) < 25)
                length_factor = np.clip(1.15 - dist / 14.0, 0.12, 1.0)
                if np.random.random() < 0.35:
                    thickness = np.random.uniform(0.06, 0.28)
                else:
                    thickness = 0.12 + 0.05 * local_density
                thickness = np.clip(thickness * length_factor + np.random.normal(0, 0.06), 0.06, 1.4)
                
                filaments.append({
                    'start': nodes[i],
                    'end': nodes[j],
                    'dist': dist,
                    'thickness': thickness,
                    'dm_ratio': dm_baryon_ratio(thickness),
                    'temp': whim_temperature(thickness),
                    'choking': choking_state(thickness)
                })
    
    return nodes, filaments, box_size, super_centers

# ============================================================
# BUILD THE 3D FIGURE
# ============================================================

def create_200_node_web():
    nodes, filaments, box_size, _super_centers = generate_web()
    
    fig = go.Figure()
    
    # --- Nodes: cyan conformal wells with glow ---
    add_node_glow_traces(
        fig,
        nodes,
        name="Galactic Cluster Nodes (~200)",
        size=3.8,
        hovertemplate=(
            "<b>Conformal Gravity Well</b><br>"
            "(%{x:.1f}, %{y:.1f}, %{z:.1f}) h⁻¹ Mpc<extra></extra>"
        ),
    )

    # --- Base filaments: cyan-blue conformal web (dim where choked) ---
    for f in filaments:
        start, end = f['start'], f['end']
        choke = f['choking']
        cool = 1.0 - choke
        if choke > CHOKE_VISUAL_THRESHOLD:
            width = 0.8
            opacity = 0.05
        else:
            width = 1.2 + 1.8 * cool
            opacity = 0.18 + 0.32 * cool

        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=cool_filament_rgb(choke), width=width),
            opacity=opacity,
            hoverinfo='skip',
            showlegend=False,
        ))

    node_links = [(f["start"], f["end"]) for f in filaments]

    # --- τ-hex ring on every cluster node (centered on the well) ---
    add_cluster_hex_rings(
        fig,
        nodes,
        connections=node_links,
        radius=EXHAUSTION_RADIUS_MPC,
    )

    # --- 142857 force lines linking connected nodes ---
    add_inter_node_force_lines(fig, node_links)

    # --- Choke threads: 142857 resonance / 1D exhaustion (drawn on top) ---
    choke_legend_shown = False
    choke_count = 0
    for f in filaments:
        if f['choking'] <= CHOKE_VISUAL_THRESHOLD:
            continue
        label = (
            "CHOKE LINE (1D Hot)"
            if f['choking'] > CHOKE_HIGHLIGHT_THRESHOLD
            else "Transitional Choke Thread"
        )
        add_choke_thread(
            fig,
            f['start'],
            f['end'],
            f['choking'],
            name='Choke Lines (142857 Resonance)',
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Thickness: {f['thickness']:.2f} h⁻¹ Mpc<br>"
                f"DM/Baryon: {f['dm_ratio']:.2f} (dropping)<br>"
                f"WHIM Temp: {f['temp']/1e5:.2f} × 10⁵ K (high friction)<br>"
                f"Choking: {f['choking']:.2f}<br>"
                f"<i>7 h⁻¹ Mpc scale exceeded — 1D geometry</i><extra></extra>"
            ),
            show_legend=not choke_legend_shown,
        )
        choke_legend_shown = True
        choke_count += 1

    apply_exhaustion_layout(
        fig,
        "Tav-Superblock 3D Cosmic Web (~200 Nodes)",
        "7 h⁻¹ Mpc Conformal Exhaustion • Local Gravity Wells • 142857 Resonance Choking",
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
    )

    fig.add_annotation(
        text=(
            "<b>Conformal Exhaustion Palette</b><br>"
            f"<span style='color:{CYAN_GLOW}'>Cyan glow</span> = conformal gravity wells<br>"
            f"<span style='color:#0088ff'>Blue web</span> = 3D-supported cool filaments<br>"
            f"<span style='color:{RESONANCE_RED}'>Magenta-red</span> = τ-hex per node + inter-node force lines<br>"
            f"<span style='color:{RESONANCE_RED}'>Bright red</span> = choke threads (≥ {CHOKE_VISUAL_THRESHOLD})"
        ),
        x=0.99, y=0.82,
        xref="paper", yref="paper",
        align="right",
        font=dict(size=9, color=TEXT_ACCENT),
        bgcolor=PANEL_BG,
        bordercolor=PANEL_BORDER,
        borderwidth=1,
        showarrow=False,
    )
    
    fig.add_annotation(
        text=(
            f"Kill Switches: {choke_count} choke threads visible — "
            "thin filaments must show reduced DM ratio (Euclid/Rubin) and "
            "highest WHIM temperatures in narrowest threads (XRISM)"
        ),
        x=0.5, y=0.01,
        xref="paper", yref="paper",
        align="center",
        font=dict(size=8.5, color='#ffaa66', style="italic"),
        showarrow=False
    )
    
    return fig

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Generating Tav-Superblock 3D cosmic web with ~200 nodes...")
    fig = create_200_node_web()
    fig.show()
    print("Done. The bright red lines are the choked 1D filaments where the Tav-Superblock physics is most visible.")