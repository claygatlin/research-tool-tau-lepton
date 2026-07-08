#!/usr/bin/env python3
"""
Cosmic Filament 1D Choking Simulator
Tav-Superblock Framework — Geometric Kill Switches for the Cosmic Web

This interactive tool demonstrates the falsifiable mechanics of the Tav-Superblock
cosmic web model:

1. 7 h⁻¹ Mpc Conformal Exhaustion Limit
   - Localized conformal gravity (tied to S¹ 5D preonic BEC) dominates below ~1 Mpc.
   - At ~7 h⁻¹ Mpc the compression exhausts against the rigid τ-circle resonance.
   - Matter is sheared from 3D structures into 1D filamentary walls.

2. 1D Geometric Choking & Dark Matter Shadow Decoupling
   - DM is the superimposed gravitational shadow of co-located 5D domains (Ω_DM/Ω_b ≈ 5.2).
   - Requires fully open 3D conformal channels (κ → 1).
   - As filament transverse thickness narrows, channels choke and the DM shadow frays/decouples.
   - Kill Switch: Euclid / Rubin (LSST) must see DM ratio drop in thin filaments.
     If ratio stays ~5.2 in narrow filaments → model falsified.

3. Topological Friction & WHIM Temperature Anomaly
   - In 1D filaments the multi-domain "Dynamic Refresh" is jammed.
   - Photon/ kinetic states cannot efficiently distribute across E₈ × E₈ symmetry.
   - Generates Topological Friction → thinnest threads burn hottest.
   - Kill Switch: XRISM (or future X-ray missions) must find highest WHIM T in the
     narrowest, most isolated filament threads (not just near AGN).

The simulator lets you vary filament transverse thickness in real time and
immediately see the predicted DM decoupling and temperature rise.

Run:
    python cosmic_filament_1d_choking_simulator.py

(Works best with a GUI matplotlib backend: TkAgg, Qt5Agg, etc.
 In Jupyter use %matplotlib widget or ipywidgets for even smoother interaction.)

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# PHYSICS FUNCTIONS (Tav-Superblock geometric rules)
# ============================================================

def dm_baryon_ratio(thickness_mpc, critical_thickness=0.35, alpha=2.8):
    """
    Dark Matter to Baryon ratio as function of filament transverse thickness.
    
    In dense 3D regions (large thickness) → full coupling κ→1 → ratio ≈ 5.2
    As thickness drops below critical scale, conformal channels choke and
    the DM shadow decouples (ratio drops).
    
    The critical_thickness is a phenomenological scale set by the geometry
    of the Kaluza-Klein cylinder and the τ-circle stiffness.
    """
    if thickness_mpc >= critical_thickness:
        return 5.2
    else:
        # Smooth power-law drop — the "fray" of the shadow
        return 5.2 * (thickness_mpc / critical_thickness) ** alpha


def whim_temperature(thickness_mpc, base_T=2.5e5, friction_strength=1.8e6):
    """
    WHIM temperature in Kelvin.
    
    Base temperature from standard astrophysical processes.
    Additional Topological Friction term that rises sharply as the filament
    narrows (Dynamic Refresh jammed → photon/kinetic states cannot offload
    efficiently across E₈ × E₈).
    """
    # Topological friction ~ 1 / thickness^1.8 (stronger in thinnest threads)
    friction = friction_strength / (thickness_mpc ** 1.8 + 0.02)
    return base_T + friction


def coupling_coefficient(thickness_mpc, critical=0.35):
    """Effective conformal coupling between domains (κ)."""
    return min(1.0, (thickness_mpc / critical) ** 1.5)


def is_falsified_dm(thickness_mpc, observed_ratio):
    """
    Kill Switch evaluator for DM shadow.
    Returns True if current state would falsify the model given an observed ratio.
    """
    predicted = dm_baryon_ratio(thickness_mpc)
    # If observed ratio is still close to 5.2 while thickness is small → falsified
    return (thickness_mpc < 0.4) and (observed_ratio > 4.8)


def is_falsified_whim(thickness_mpc, observed_T_near_agn, observed_T_thin):
    """
    Kill Switch for Topological Friction.
    If thin threads are NOT hotter than near-AGN gas → model falsified.
    """
    predicted_thin = whim_temperature(thickness_mpc)
    # Simple heuristic: thin thread should be significantly hotter
    return observed_T_thin < (predicted_thin * 0.7)


# ============================================================
# INTERACTIVE VISUALIZATION
# ============================================================

def create_simulator():
    # Dark cosmic theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9), facecolor='#0a0a12')
    
    # Grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1.8, 1.2, 0.8],
                          hspace=0.35, wspace=0.3)
    
    # --- Top left: Filament schematic ---
    ax_fil = fig.add_subplot(gs[0, 0])
    ax_fil.set_xlim(-2.5, 2.5)
    ax_fil.set_ylim(-1.8, 1.8)
    ax_fil.set_aspect('equal')
    ax_fil.axis('off')
    ax_fil.set_title('Filament Geometry & DM Shadow\n(adjust thickness below)', 
                     color='#aaccff', fontsize=11, pad=10)
    
    # Background void
    void = patches.Rectangle((-2.5, -1.8), 5, 3.6, linewidth=0, 
                             facecolor='#050508', zorder=0)
    ax_fil.add_patch(void)
    
    # Filament body (will be updated)
    filament_body = patches.FancyBboxPatch((-2.2, -0.8), 4.4, 1.6,
                                           boxstyle="round,pad=0.05,rounding_size=0.3",
                                           facecolor='#334455', edgecolor='#88aaff',
                                           linewidth=2.5, zorder=1, alpha=0.85)
    ax_fil.add_patch(filament_body)
    
    # Baryonic core (bright inner region)
    baryon_core = patches.FancyBboxPatch((-2.0, -0.55), 4.0, 1.1,
                                         boxstyle="round,pad=0.02,rounding_size=0.2",
                                         facecolor='#ffcc66', edgecolor='none',
                                         alpha=0.9, zorder=2)
    ax_fil.add_patch(baryon_core)
    
    # DM shadow halo (fades when choked)
    dm_halo = patches.FancyBboxPatch((-2.3, -0.95), 4.6, 1.9,
                                     boxstyle="round,pad=0.08,rounding_size=0.35",
                                     facecolor='#4488ff', edgecolor='none',
                                     alpha=0.35, zorder=0)
    ax_fil.add_patch(dm_halo)
    
    # Annotations
    ax_fil.text(0, 1.55, '7 h⁻¹ Mpc Conformal Exhaustion → 1D shear', 
                ha='center', va='bottom', color='#ffaa66', fontsize=9, style='italic')
    ax_fil.text(0, -1.55, 'Transverse thickness controls conformal channel openness',
                ha='center', va='top', color='#88ddff', fontsize=8)
    
    # Colorbar legend for DM shadow
    cax = fig.add_axes([0.05, 0.62, 0.015, 0.18])
    cmap_dm = LinearSegmentedColormap.from_list('dm_shadow', ['#112244', '#4488ff', '#aaccff'])
    sm = plt.cm.ScalarMappable(cmap=cmap_dm, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('DM Shadow\nIntensity', color='white', fontsize=8, labelpad=-5)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)
    
    # --- Top middle: DM Ratio vs Thickness ---
    ax_dm = fig.add_subplot(gs[0, 1])
    thickness_range = np.linspace(0.05, 1.5, 300)
    dm_curve = [dm_baryon_ratio(t) for t in thickness_range]
    
    ax_dm.plot(thickness_range, dm_curve, color='#4488ff', linewidth=2.5, label='Predicted DM/Baryon')
    ax_dm.axhline(5.2, color='#ffaa66', linestyle='--', linewidth=1.5, alpha=0.8, label='Full coupling (5.2)')
    ax_dm.axvline(0.35, color='#ff6666', linestyle=':', linewidth=1.8, alpha=0.9, label='Choking threshold')
    
    current_dm_line, = ax_dm.plot([], [], 'o', color='#ffdd88', markersize=10, zorder=5)
    ax_dm.set_xlabel('Filament Transverse Thickness (Mpc)', color='white', fontsize=9)
    ax_dm.set_ylabel(r'$\Omega_{DM} / \Omega_b$', color='white', fontsize=10)
    ax_dm.set_title('Dark Matter Shadow Decoupling', color='#aaccff', fontsize=11, pad=6)
    ax_dm.set_xlim(0, 1.5)
    ax_dm.set_ylim(0, 6.5)
    ax_dm.legend(loc='upper right', fontsize=7, framealpha=0.3)
    ax_dm.grid(True, alpha=0.2, color='white')
    ax_dm.tick_params(colors='white', labelsize=8)
    
    # Kill switch text box
    kill_text_dm = ax_dm.text(0.98, 0.02, '', transform=ax_dm.transAxes,
                              fontsize=8, color='#ff6666', ha='right', va='bottom',
                              bbox=dict(boxstyle='round', facecolor='#330000', alpha=0.7),
                              visible=False)
    
    # --- Top right: WHIM Temperature ---
    ax_t = fig.add_subplot(gs[0, 2])
    t_curve = [whim_temperature(t) for t in thickness_range]
    
    ax_t.plot(thickness_range, np.array(t_curve)/1e5, color='#ff6666', linewidth=2.5, 
              label='Topological Friction + base')
    ax_t.axvline(0.35, color='#ff6666', linestyle=':', linewidth=1.8, alpha=0.9)
    
    current_t_line, = ax_t.plot([], [], 'o', color='#ffdd88', markersize=10, zorder=5)
    ax_t.set_xlabel('Filament Transverse Thickness (Mpc)', color='white', fontsize=9)
    ax_t.set_ylabel(r'Temperature ($10^5$ K)', color='white', fontsize=9)
    ax_t.set_title('WHIM Temperature Anomaly\n(Thinnest threads hottest)', color='#ffaa66', fontsize=11, pad=6)
    ax_t.set_xlim(0, 1.5)
    ax_t.set_ylim(1, 12)
    ax_t.legend(loc='upper right', fontsize=7, framealpha=0.3)
    ax_t.grid(True, alpha=0.2, color='white')
    ax_t.tick_params(colors='white', labelsize=8)
    
    kill_text_t = ax_t.text(0.98, 0.02, '', transform=ax_t.transAxes,
                            fontsize=8, color='#ff6666', ha='right', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='#330000', alpha=0.7),
                            visible=False)
    
    # --- Bottom: Filament cross-section detail + status panel ---
    ax_status = fig.add_subplot(gs[1:, :])
    ax_status.axis('off')
    ax_status.set_xlim(0, 10)
    ax_status.set_ylim(0, 10)
    
    status_box = patches.FancyBboxPatch((0.3, 0.3), 9.4, 9.4,
                                        boxstyle="round,pad=0.02,rounding_size=0.4",
                                        facecolor='#111122', edgecolor='#4466aa',
                                        linewidth=2, alpha=0.95)
    ax_status.add_patch(status_box)
    
    status_text = ax_status.text(0.5, 9.5, '', transform=ax_status.transAxes,
                                 fontsize=9.5, color='white', va='top',
                                 family='monospace',
                                 bbox=dict(boxstyle='round', facecolor='#0a0a18', alpha=0.0))
    
    # Slider
    slider_ax = fig.add_axes([0.15, 0.035, 0.72, 0.035])
    thickness_slider = Slider(slider_ax, 'Filament Transverse Thickness (Mpc)', 
                              0.05, 1.5, valinit=0.85, valstep=0.01,
                              color='#4488ff', track_color='#334455')
    thickness_slider.label.set_color('white')
    thickness_slider.valtext.set_color('#ffdd88')
    
    # ============================================================
    # UPDATE FUNCTION
    # ============================================================
    def update(val):
        thickness = thickness_slider.val
        
        # === Update filament visualization ===
        # Scale the filament width with thickness (normalized)
        width_scale = 0.6 + 1.4 * (thickness / 1.5)   # visual stretch
        filament_body.set_width(4.4 * width_scale)
        filament_body.set_x(-2.2 * width_scale)
        
        baryon_core.set_width(4.0 * width_scale)
        baryon_core.set_x(-2.0 * width_scale)
        
        dm_halo.set_width(4.6 * width_scale)
        dm_halo.set_x(-2.3 * width_scale)
        
        # DM shadow intensity fades when choked
        dm_alpha = 0.15 + 0.55 * coupling_coefficient(thickness)
        dm_halo.set_alpha(dm_alpha)
        
        # Color shift: hotter = more red when thin
        if thickness < 0.4:
            filament_body.set_facecolor('#553322')
            baryon_core.set_facecolor('#ffaa44')
        else:
            filament_body.set_facecolor('#334455')
            baryon_core.set_facecolor('#ffcc66')
        
        # === Update DM ratio plot ===
        current_ratio = dm_baryon_ratio(thickness)
        current_dm_line.set_data([thickness], [current_ratio])
        
        # Kill switch check (example observed ratio = 5.1)
        if is_falsified_dm(thickness, 5.1):
            kill_text_dm.set_text('⚠ KILL SWITCH TRIGGERED\nDM ratio still ~5.2 in thin filament')
            kill_text_dm.set_visible(True)
        else:
            kill_text_dm.set_visible(False)
        
        # === Update Temperature plot ===
        current_T = whim_temperature(thickness)
        current_t_line.set_data([thickness], [current_T / 1e5])
        
        if thickness < 0.35:
            kill_text_t.set_text('⚠ Topological Friction dominant\nThin thread should be hottest')
            kill_text_t.set_visible(True)
        else:
            kill_text_t.set_visible(False)
        
        # === Update status panel ===
        kappa = coupling_coefficient(thickness)
        T_now = whim_temperature(thickness)
        
        status_str = (
            f"  CURRENT STATE  |  Thickness = {thickness:.2f} Mpc   |   κ (coupling) = {kappa:.3f}\n\n"
            f"  DM / Baryon Ratio     : {current_ratio:.2f}     (full coupling = 5.2)\n"
            f"  WHIM Temperature      : {T_now/1e5:.2f} × 10⁵ K   {'  ← Topological Friction rising sharply' if thickness < 0.4 else ''}\n\n"
            f"  GEOMETRIC RULES ACTIVE:\n"
            f"  • Conformal channels {'OPEN' if kappa > 0.85 else 'PARTIALLY CHOKED' if kappa > 0.5 else 'HEAVILY CHOKED'}\n"
            f"  • DM shadow {'fully coupled to baryons' if kappa > 0.85 else 'fraying / decoupling'}\n"
            f"  • Topological friction {'low' if thickness > 0.6 else 'moderate' if thickness > 0.35 else 'EXTREME — thinnest threads burn hottest'}\n\n"
            f"  FALSIFIABLE KILL SWITCHES (Euclid / Rubin / XRISM):\n"
            f"  1. Thin filament DM ratio must drop below ~4.5–5.0. Persistent 5.2:1 → FALSIFIED\n"
            f"  2. Narrowest isolated threads must show highest WHIM T (not just near AGN) → FALSIFIED otherwise\n\n"
            f"  This simulator enforces the unyielding spatial comb of Tav-Superblock Kaluza-Klein geometry."
        )
        status_text.set_text(status_str)
        
        fig.canvas.draw_idle()
    
    # Connect slider
    thickness_slider.on_changed(update)
    
    # Initial draw
    update(thickness_slider.val)
    
    plt.suptitle('Tav-Superblock Cosmic Filament 1D Choking Simulator\n'
                 'Geometric Kill Switches: 7 h⁻¹ Mpc Limit • DM Shadow Decoupling • Topological Friction',
                 color='#88aaff', fontsize=13, y=0.985)
    
    plt.show()
    return fig, thickness_slider


if __name__ == "__main__":
    print(__doc__)
    create_simulator()
