# CMS Dimuon Pipeline — Critique Integration (2026-07-15)

This note maps the Grok critique of the DoubleMuParked validation pipeline to concrete
research-tool modules. Findings are classified as **exploratory** unless a preregistered
q_T study survives all null tests.

## Critique verdict (summary)

- Engineering milestone valid (61M-event processing, artifacts, Data-MC tooling).
- Residue-2 mod-7 excess is **trigger bias** (n_μ ≈ 2 under dimuon HLT), not sevenfold topology.
- SMHiggsToZZTo4L is **signal MC** — invalid as universal SM background for inclusive dimuon data.
- Exactly 12.00 σ was a **capped internal engine score**, not HEP significance.
- 10 GeV pT bins cannot resolve 313.1 MeV (0.313 GeV) features — use fine q_T histograms.

## Code integration

| Critique item | Implementation |
|---------------|----------------|
| Trigger-aware mod-7 null | `hep_statistics.trigger_aware_mod7_expected_fractions`, `mod7_analysis_package` |
| Engine score labeling | `hep_statistics.engine_score_from_components` (reports raw + capped, not “discovery σ”) |
| MC role warnings | `hep_statistics.classify_mc_sample_label` in Data-MC comparison |
| Fine q_T at 313.1 MeV scale | `hep_statistics.fine_qt_histogram` (10 MeV bins) |
| q_T, α, pt-balance observables | `preregistered_recoil_study` (`system_pt` = q_T) |
| Frozen signal window 0.25–0.40 GeV | `PREREG_QT_SIGNAL_WINDOW` in `hep_statistics.py` |
| Event-order shuffle null | `hep_statistics.shuffle_order_null_test` |
| Chunk/stride sensitivity | `hep_statistics.chunk_stride_sensitivity` |
| Split-sample stability | `hep_statistics.split_sample_stability` |
| Exploratory classification | All reports tagged `EXPLORATORY_PIPELINE_ANOMALY` |

## Menu actions

1. **CERN Open Data → Preregistered Recoil q_T Study (HEP Controls)** — primary HEP-grade path.
2. **CERN Open Data → Run MC Validation Suite** — full pipeline with updated labels and Option 3 DY+tt̄ stack.

## Recommended next steps (not yet automated)

- Opposite-charge selection (requires `Muon_charge` branch).
- Official CMS muon momentum-scale constraint (Z → μμ).
- Held-out Run B vs Run C after freezing cuts.
- Signal injection toys at q₀ = 0.3131 GeV.

## References

- CMS Open Data DoubleMuParked: record 12341
- CMS H→4ℓ educational example: record 12360
- Cowan et al., arXiv:1007.1727 (significance methods)