import numpy as np
import plotly.graph_objects as go
import webbrowser
import os

# ============================================================
# CUSTOMIZABLE PARAMETERS (edit these to explore the model)
# ============================================================
np.random.seed(42)           # Change for different random distributions
n_galaxies_per_domain = 45   # Number of galaxies per domain
domain_range = 1.2           # Spatial extent of the E8 projection

# ============================================================
# DOMAIN DEFINITIONS (matching the original visualization)
# ============================================================
domains = {
    "Prime Domain (U)": {
        "color": "yellow",
        "label": "Prime Domain (U)"
    },
    "Prime Past (B)": {
        "color": "red",
        "label": "Prime Past (B)"
    },
    "Relative-Future 1 (C)": {
        "color": "darkblue",
        "label": "Relative-Future 1 (C)"
    },
    "Relative-Past 1 (S)": {
        "color": "lightblue",
        "label": "Relative-Past 1 (S)"
    },
    "Relative Future 2 (T)": {
        "color": "darkgreen",
        "label": "Relative Future 2 (T)"
    },
    "Relative Past 2 (B)": {
        "color": "lightgreen",
        "label": "Relative Past 2 (B)"
    },
}

# ============================================================
# GENERATE SYNTHETIC GALAXY DATA
# ============================================================
traces = []

for domain_name, props in domains.items():
    n = n_galaxies_per_domain
    x = np.random.uniform(-domain_range, domain_range, n)
    y = np.random.uniform(-domain_range, domain_range, n)
    z = np.random.uniform(-domain_range, domain_range, n)
    
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=props["color"],
            opacity=0.75,
            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
        ),
        name=props["label"],
        hovertemplate=(
            f"<b>{props['label']}</b><br>"
            "X (E8 Projection): %{x:.2f}<br>"
            "Y (E8 Projection): %{y:.2f}<br>"
            "Z (E8 Projection): %{z:.2f}<br>"
            "<extra></extra>"
        )
    )
    traces.append(trace)

# ============================================================
# CREATE INTERACTIVE 3D FIGURE
# ============================================================
fig = go.Figure(data=traces)

fig.update_layout(
    title=dict(
        text="E8 Ring Projection — Tav-Superblock Cosmology<br>"
             "Interactive Galaxy Distribution by Domain / Stagger-Phase",
        font=dict(size=18)
    ),
    scene=dict(
        xaxis=dict(title="X (E8 Projection)", range=[-domain_range, domain_range]),
        yaxis=dict(title="Y (E8 Projection)", range=[-domain_range, domain_range]),
        zaxis=dict(title="Z (E8 Projection)", range=[-domain_range, domain_range]),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
    ),
    legend=dict(
        title="Domains / Stagger-Phases",
        font=dict(size=12),
        itemsizing='constant'
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    height=850,
    template="plotly_white"
)

# Add explanatory annotation
fig.add_annotation(
    text=(
        "This is an <b>interactive 3D model</b> of the Tav-Superblock framework.<br>"
        "• Rotate with mouse • Zoom with scroll • Hover points for coordinates<br><br>"
        "These galaxies act as 'negative controls' where shadow-mass emergence is suppressed.<br>"
        "Edit parameters at the top of the script to change the distribution.<br><br>"
        "Key relation (placeholder model):<br>"
        "f_sh = η_overlap × (0.5 + 0.5 × sin(φ_stagger × π))"
    ),
    x=0.5,
    y=0.02,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=11),
    align="center",
    bgcolor="rgba(255,255,255,0.85)",
    bordercolor="gray",
    borderwidth=1
)

# ============================================================
# SAVE & OPEN INTERACTIVE HTML
# ============================================================
output_file = "tav_superblock_interactive_model.html"
fig.write_html(
    output_file,
    include_plotlyjs=True,
    full_html=True,
    config={'displayModeBar': True, 'scrollZoom': True}
)

print(f"\n✅ Interactive model saved to: {os.path.abspath(output_file)}")
print("   Opening in your default browser...")
webbrowser.open('file://' + os.path.abspath(output_file))

# ============================================================
# SIMPLE SHADOW-MASS FRACTION MODEL (explorable)
# ============================================================
def compute_shadow_mass_fraction(eta_overlap=0.5, phi_stagger=0.0):
    """
    Placeholder model for shadow-mass fraction f_sh.
    
    In the Tav-Superblock framework, f_sh is a geometric function
    of domain overlap (η_overlap) and E8 stagger phase (φ_stagger).
    
    This is a simple illustrative formula — the real version would
    come from the E8 lattice geometry and supersphere mapping.
    """
    f_sh = eta_overlap * (0.5 + 0.5 * np.sin(phi_stagger * np.pi))
    return max(0.0, min(1.0, f_sh))  # Clamp to [0, 1]

print("\n" + "="*60)
print("SHADOW-MASS FRACTION EXPLORER")
print("="*60)
print("Example calculations:")
print(f"  η_overlap=0.2,  φ_stagger=0     → f_sh = {compute_shadow_mass_fraction(0.2, 0):.3f}")
print(f"  η_overlap=0.7,  φ_stagger=π/2   → f_sh = {compute_shadow_mass_fraction(0.7, np.pi/2):.3f}")
print(f"  η_overlap=0.4,  φ_stagger=π     → f_sh = {compute_shadow_mass_fraction(0.4, np.pi):.3f}")
print("\nYou can call compute_shadow_mass_fraction(eta, phi) in the script or terminal.")
print("="*60)
