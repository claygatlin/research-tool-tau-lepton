# tav_bbn_confrontation.py — Complete Tav Framework BBN Scanner + Empirical Confrontation

**Framework Context**: Tav Topology / Tau Universe merged with Superblock.  
**Date**: 2026-07-01  
**Purpose**: A single entry point that combines:
- The improved minimal BBN engine with Tav cylinder interference
- Empirical data loading (neutron lifetime + BBN abundances)
- Automated parameter scans over Tav variables (`A`, `delta_phi_cyl`, `delta_k_wind`)
- Direct confrontation with observations (focus on lithium problem + D/H and Y_p stability)
- Summary tables, tension calculations, and JSON/plot output

This script directly addresses lithium-problem quantification using the Tav framework.

## Project layout (Public/TauSuperblock/)

| File | Role |
|------|------|
| `tav_bbn_confrontation.py` | CLI entry point |
| `menus/prime_past/bbn_confrontation.py` | Core scanner module |
| `menus/prime_past/bbn_interference.py` | τ-integration BBN variant |
| `minimal_bbn_tav.py` | Script helper for single-run comparisons |
| `empirical_fetch.py` | Empirical data helper |
| `artifacts/prime_past/` | Scan JSON + PNG output |

## How to use

```bash
cd ~/Public/TauSuperblock
./venv/bin/python tav_bbn_confrontation.py
```

By default it runs a small scan (4 A-grid points × 9 phase/winding combos), prints a ranked summary table, saves JSON, and writes a matplotlib scatter plot.

Modify scan ranges at the bottom of `tav_bbn_confrontation.py` (see `SCAN_RANGES` / `N_POINTS`).

## research_tool integration

**Prime Past Harmonic** → **BBN Confrontation Scan**

Entry fields: `n_points`, `t_span_max`, `scan_plot`

## Features

- **Correct lithium direction** — positive `A` reduces ^7Li (lithium problem mitigation channel)
- **Parameter scan** over `A`, `delta_phi_cyl`, `delta_k_wind`
- **Automatic tension calculation** vs PDG/literature anchors (`fetch_empirical_data`)
- **Summary table** ranked by Li7 σ
- **JSON export** — `artifacts/prime_past/tav_bbn_scan_*.json` and `tav_bbn_scan_results.json` (cwd)
- **Plot export** — `artifacts/prime_past/tav_bbn_scan_*.png`

## Empirical anchors (fallback)

- D/H = 2.547×10⁻⁵ ± 2.9×10⁻⁷  
- Y_p (^4He) = 0.245 ± 0.003  
- ^7Li/H (obs) = 1.6×10⁻¹⁰ ± 3×10⁻¹¹  
- ^7Li/H (std theory) = 5.0×10⁻¹⁰ ± 5×10⁻¹¹  
- τ_n bottle/beam = 878.5 / 888.0 s