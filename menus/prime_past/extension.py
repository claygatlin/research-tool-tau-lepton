"""
Modular bridge: Prime Past Harmonic ↔ research_tool.py

Dispatches menu actions into TavSuperblockPrimePastHarmonic and BBN interference scans.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt

from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from menus.prime_past.harmonic import (
    ARTIFACTS_DIR,
    M0_MEV,
    TavSuperblockPrimePastHarmonic,
    ensure_artifacts_dir,
)

MODULE_TAG = "PRIME_PAST_HARMONIC"

MENU_ACTIONS = [
    "Verification Suite",
    "Generate Visualizations",
    "Export LaTeX Appendix",
    "Monte-Carlo Domain Scan",
    "BBN Interference Scan",
    "BBN Confrontation Scan",
    "Enhanced BBN Confrontation",
    "Final BBN Confrontation Tool",
    "Full Module Demo",
]

_ACTION_KEYS: Dict[str, str] = {
    "Verification Suite": "verification",
    "Generate Visualizations": "visualizations",
    "Export LaTeX Appendix": "latex",
    "Monte-Carlo Domain Scan": "monte_carlo",
    "BBN Interference Scan": "bbn_interference",
    "BBN Confrontation Scan": "bbn_confrontation",
    "Enhanced BBN Confrontation": "bbn_enhanced",
    "Final BBN Confrontation Tool": "bbn_final",
    "Full Module Demo": "full_demo",
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for Prime Past Harmonic actions."""
    if action == "Final BBN Confrontation Tool":
        return [
            {
                "key": "a_points",
                "label": "A-grid points",
                "default": "8",
                "required": False,
                "hint": "Scan + auto-save best config + comparison plot",
            },
            {
                "key": "phi_points",
                "label": "Phase grid points",
                "default": "5",
                "required": False,
                "hint": "Linspace 0→π over delta_phi_cyl",
            },
            {
                "key": "delta_k_wind",
                "label": "Winding contribution",
                "default": "0.469",
                "required": False,
            },
            {
                "key": "highlight_json",
                "label": "Highlight prior JSON (optional)",
                "default": "",
                "required": False,
                "hint": "Skip scan; load artifacts/prime_past/*.json and print top configs",
            },
        ]
    if action == "Enhanced BBN Confrontation":
        return [
            {
                "key": "a_points",
                "label": "A-grid points",
                "default": "8",
                "required": False,
                "hint": "Linspace 0→0.035 over interference amplitude A",
            },
            {
                "key": "phi_points",
                "label": "Phase grid points",
                "default": "5",
                "required": False,
                "hint": "Linspace 0→π over delta_phi_cyl",
            },
            {
                "key": "delta_k_wind",
                "label": "Winding contribution",
                "default": "0.469",
                "required": False,
                "hint": "Fixed delta_k_wind for enhanced heatmap scan",
            },
            {
                "key": "analyze_json",
                "label": "Analyze prior JSON (optional path)",
                "default": "",
                "required": False,
                "hint": "Skip scan; load artifacts/prime_past/*.json and print top configs",
            },
        ]
    if action == "BBN Confrontation Scan":
        return [
            {
                "key": "n_points",
                "label": "A-grid points (scan resolution)",
                "default": "4",
                "required": False,
                "hint": "Linspace 0→0.03 over A; total runs = n_points × 9 phase/winding combos",
            },
            {
                "key": "t_span_max",
                "label": "ODE time span max (optional)",
                "default": "300",
                "required": False,
                "hint": "RK45 integration upper bound (natural time units)",
            },
            {
                "key": "scan_plot",
                "label": "Save scan plot (yes/no)",
                "default": "yes",
                "required": False,
                "hint": "Li7 tension vs A scatter under artifacts/prime_past/",
            },
        ]
    if action == "BBN Interference Scan":
        return [
            {
                "key": "A",
                "label": "Interference amplitude A",
                "default": "0.015",
                "required": False,
                "hint": "Tau cylinder fractional Hubble perturbation (0 = standard BBN)",
            },
            {
                "key": "delta_phi_cyl",
                "label": "Cylinder phase offset (rad)",
                "default": "0.0",
                "required": False,
                "hint": "Prime-domain phase offset on the Tau cylinder",
            },
            {
                "key": "delta_k_wind",
                "label": "Winding contribution",
                "default": "0.469",
                "required": False,
                "hint": "Strange-quark winding scale; tune for charm sector tests",
            },
            {
                "key": "xi",
                "label": "Coherence length xi",
                "default": "8.0",
                "required": False,
                "hint": "n_hier sub-structure coherence (Mpc-scale mapping)",
            },
            {
                "key": "empirical_plot",
                "label": "Plot empirical confrontation (yes/no)",
                "default": "yes",
                "required": False,
                "hint": "Save bbn_abundances_test.png under empirical_tests artifacts",
            },
        ]
    return [
        {
            "key": "mc_samples",
            "label": "Monte-Carlo samples (optional)",
            "default": "3000",
            "required": False,
            "hint": "Used by Monte-Carlo and Full Demo actions",
        },
        {
            "key": "notes",
            "label": "Run notes (optional)",
            "default": "",
            "required": False,
            "hint": f"Annotation for {action}",
        },
    ]


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the Prime Past entry form."""
    base = [
        "Harmonic / verification actions use the built-in domain/octonion model.",
        "BBN confrontation scans auto-fetch empirical: targets pre-run (see action notes).",
        "Outputs: PNG plots + .tex appendix under artifacts/ (auto-created).",
        "MC samples: positive integer (e.g. 3000); used by Monte-Carlo / Full Demo.",
    ]
    if action == "BBN Interference Scan":
        base.extend(
            [
                "Evolves post-freeze-out BBN (T: 0.8 → 0.01 MeV) with Tav H(T) interference.",
                "Compares ^4He mass fraction and ^7Li/H to curated Planck/Spite observations.",
                "Report: artifacts/prime_past/bbn_interference_*.json",
            ]
        )
    if action in {
        "BBN Interference Scan",
        "BBN Confrontation Scan",
        "Enhanced BBN Confrontation",
        "Final BBN Confrontation Tool",
    }:
        base.extend(
            [
                "Pre-run auto-fetch: empirical:bbn_abundances + empirical:neutron_lifetime "
                "(pull→cache under datasets/empirical/).",
                "Provenance logged in JSON reports as empirical_provenance.",
            ]
        )
    if action == "BBN Confrontation Scan":
        base.extend(
            [
                "RK45 minimal BBN grid over A, cylinder phase, and winding.",
                "Empirical anchors: fetch_empirical_data() (PDG 2025 + Spite plateau).",
                "Scripts: Public/TauSuperblock/tav_bbn_confrontation.py",
                "Report: Public/TauSuperblock/artifacts/prime_past/tav_bbn_scan_*.json",
            ]
        )
    if action == "Enhanced BBN Confrontation":
        base.extend(
            [
                "Five-state BBN (D/^3He bottleneck) + Li7 heatmap over A vs phase.",
                "Script: Public/TauSuperblock/enhanced_tav_bbn_confrontation.py",
                "Output: artifacts/prime_past/li_tension_heatmap_*.png + enhanced_tav_bbn_scan_*.json",
            ]
        )
    if action == "Final BBN Confrontation Tool":
        base.extend(
            [
                "Full scan → best_tav_bbn_config.json + tav_vs_standard_comparison.png",
                "Script: Public/TauSuperblock/final_tav_bbn_confrontation_tool.py",
                "Output: Public/TauSuperblock/artifacts/prime_past/",
            ]
        )
    return base


def run_action(
    selection: str,
    show_plots: bool = True,
    options: dict | None = None,
) -> str | None:
    """Execute a Prime Past Harmonic module action from the research tool menu."""
    if options is None:
        options = {}
    action = _ACTION_KEYS.get(selection)
    if action is None:
        print(f"[TAV ENGINE] Unknown Prime Past action: {selection}")
        return None

    ensure_artifacts_dir()
    fw = TavSuperblockPrimePastHarmonic()

    print(f"\n[TAV ENGINE] Prime Past Harmonic module — {selection}")
    print(f"[TAV ENGINE] Topological anchor m₀ = {M0_MEV} MeV")
    print(f"[TAV ENGINE] Artifacts directory: {ARTIFACTS_DIR}")

    if action == "bbn_interference":
        from menus.prime_past.bbn_interference import (
            confront_empirical_abundances,
            run_bbn_comparison,
            save_bbn_report,
        )

        plot_emp = str(options.get("empirical_plot", "yes")).lower() in {
            "yes",
            "y",
            "true",
            "1",
        }
        comparison = run_bbn_comparison(options, verbose=True)
        confront_empirical_abundances(comparison, plot=plot_emp and show_plots, verbose=True)
        prov = options.get("_empirical_provenance")
        if prov:
            comparison["empirical_provenance"] = prov
        return save_bbn_report(comparison)

    if action == "bbn_confrontation":
        from menus.prime_past.bbn_confrontation import run_tav_scan, save_scan_report

        n_points = int(options.get("n_points") or 4)
        t_max = float(options.get("t_span_max") or 300)
        do_plot = str(options.get("scan_plot", "yes")).lower() in {"yes", "y", "true", "1"}
        results_list, best = run_tav_scan(
            n_points=n_points,
            t_span=(0.0, t_max),
            verbose=True,
            plot=do_plot and show_plots,
            show_plot=show_plots,
        )
        return save_scan_report(
            results_list,
            best,
            also_cwd=False,
            empirical_provenance=options.get("_empirical_provenance"),
        )

    if action == "bbn_enhanced":
        from menus.prime_past.bbn_enhanced import (
            load_and_analyze_scan,
            plot_lithium_tension_heatmap,
            run_enhanced_scan,
            save_enhanced_scan,
        )

        analyze_path = (options.get("analyze_json") or "").strip()
        if analyze_path:
            load_and_analyze_scan(analyze_path)
            return analyze_path

        a_points = int(options.get("a_points") or 8)
        phi_points = int(options.get("phi_points") or 5)
        delta_k_wind = float(options.get("delta_k_wind") or 0.469)
        scan_results = run_enhanced_scan(
            a_points=a_points,
            phi_points=phi_points,
            delta_k_wind=delta_k_wind,
            verbose=True,
        )
        heatmap = plot_lithium_tension_heatmap(scan_results, show=show_plots)
        return save_enhanced_scan(
            scan_results,
            heatmap_path=heatmap,
            empirical_provenance=options.get("_empirical_provenance"),
        )

    if action == "bbn_final":
        from menus.prime_past.bbn_enhanced import (
            load_and_highlight_best,
            run_final_confrontation_tool,
        )

        highlight_path = (options.get("highlight_json") or "").strip()
        if highlight_path:
            best = load_and_highlight_best(highlight_path)
            return highlight_path if best else None

        outputs = run_final_confrontation_tool(
            a_points=int(options.get("a_points") or 8),
            phi_points=int(options.get("phi_points") or 5),
            delta_k_wind=float(options.get("delta_k_wind") or 0.469),
            verbose=True,
            show_plots=show_plots,
            empirical_provenance=options.get("_empirical_provenance"),
        )
        return outputs.get("best_config_path")

    if action == "verification":
        fw.run_verification_tests(verbose=True, run_mc=False)
        _print_domain_summary(fw)
        return None

    if action == "visualizations":
        paths = _generate_visualizations(fw, show_plots=show_plots)
        for label, path in paths.items():
            print(f"  {label}: {path}")
        return None

    if action == "latex":
        tex_path = fw.write_latex_proof_appendix()
        print(f"[TAV ENGINE] LaTeX appendix written: {tex_path}")
        return str(tex_path)

    if action == "monte_carlo":
        mc_samples = int(options.get("mc_samples") or 3000)
        mc = fw.monte_carlo_domain_scan(n_samples=mc_samples)
        print(
            f"[TAV ENGINE] Monte-Carlo light-stable fraction: "
            f"{mc['light_stable_fraction'] * 100:.2f}%"
        )
        print(f"[TAV ENGINE] Mean m_eff (past sample): {mc['mean_m_eff_past']:.2f} MeV")
        print(f"[TAV ENGINE] Note: {mc['note']}")
        return None

    if action == "full_demo":
        mc_samples = int(options.get("mc_samples") or 3000)
        fw.run_verification_tests(verbose=True, run_mc=True)
        _print_domain_summary(fw)
        paths = _generate_visualizations(fw, show_plots=show_plots)
        tex_path = fw.write_latex_proof_appendix()
        mc = fw.monte_carlo_domain_scan(n_samples=mc_samples)
        print("\n[TAV ENGINE] Full demo outputs:")
        for label, path in paths.items():
            print(f"  {label}: {path}")
        print(f"  LaTeX appendix: {tex_path}")
        print(
            f"  MC light-stable fraction: {mc['light_stable_fraction'] * 100:.2f}%"
        )
        print(
            "[TOPOLOGICAL ANCHOR] 313.1 MeV mass-gap signal is independent of "
            "halo-fit and domain-sampling parameters."
        )
        return None

    return None


def _print_domain_summary(fw: TavSuperblockPrimePastHarmonic) -> None:
    print("\nUp ↔ Down geometric counterweights:")
    for name in ("PrimeFuture", "PrimePast"):
        domain = fw.domains[name]
        print(
            f"  {name:15s} | Q={fw.charge_operator(domain):+6.3f} | "
            f"m_eff={fw.effective_mass(domain):7.2f} MeV | "
            f"phase={domain.oct_phase.name}"
        )


def _generate_visualizations(
    fw: TavSuperblockPrimePastHarmonic,
    show_plots: bool = True,
) -> Dict[str, str]:
    print("[TAV ENGINE] Generating Prime Past visualizations...")
    paths = {
        "Phase clock": fw.plot_8phase_clock(),
        "Domain properties": fw.plot_domain_properties(),
        "Isospin loop": fw.plot_isospin_loop(),
    }
    if show_plots:
        for path in paths.values():
            if path and os.path.isfile(path):
                img = plt.imread(path)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
        if any(paths.values()):
            plt.show()
    return {k: v for k, v in paths.items() if v}