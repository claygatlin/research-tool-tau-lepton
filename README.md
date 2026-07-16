# Tau-Superblock Research Engine

Interactive curses-based research console for the **Tau Universe / Tav-Superblock** framework. Download scientific datasets, run τ-harmonic analyses across cosmology, particle physics, and empirical confrontations, and optionally review results with remote LLM providers.

**Entry point:** `research_tool.py`

## Gateway integration

This tree is merged into the Research **LLM Gateway** (`../gateway.py`):

```bash
cd /home/wyle-e/Research
./scripts/llm                 # gateway REPL
# then: /research   or  /rt   → full curses UI
#       /rt-status  /rt-modules  /rt-providers  /rt-run …

./scripts/rt                  # launch research_tool UI directly
./scripts/rt status           # bridge health check
```

API keys: `config/api_keys.env` (see example) or the same env vars the gateway uses
(`XAI_API_KEY`, `NVIDIA_API_KEY` / `NV_API_KEY`, `OLLAMA_BASE_URL`, …).
Gateway persistent memory (name, notes) is passed into the child process env.

## Remote archive transfer

Finished datasets are staged locally under `.tav_project_staging/` and can be copied to a configured remote host when enabled.

Remote SSH host updates are **disabled in code** (local staging only).

**To re-enable SSH/SFTP/rsync sync:** see [enable_ssh_sync.txt](enable_ssh_sync.txt) for file-by-file uncomment instructions, required environment variables, and verification steps.

## Features

- **Modular menus** — each domain lives in `menus/*/` with thin `*_extension.py` shims at the project root for backward compatibility
- **Dataset manager** — fetch, cache, and resume batch downloads (DESI, Planck, SPARC, FRB, LHCb, …)
- **Prime Past / BBN** — τ-Euler BBN interference scans, lithium confrontation, enhanced 5-state engine
- **Remote AI** — Grok, Gemini, OpenAI, Claude, OpenRouter, Ollama, NVIDIA NIM (`integrate.api.nvidia.com`)
- **Post-run LLM review** — set **AI verbose review = yes** on any module entry form

## Quick start

```bash
cd research_tool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp config/api_keys.env.example config/api_keys.env
# Edit config/api_keys.env with your API keys (optional; needed for Remote AI)

./venv/bin/python research_tool.py
```

Or use the helper script:

```bash
./research.sh
```

## Configuration

| File | Purpose |
|------|---------|
| `config/api_keys.env` | LLM provider keys (gitignored; copy from `api_keys.env.example`) |
| `.env` | Optional fallback for the same keys |
| `requirements-desi-production.txt` | Extra DESI MCMC dependencies (`emcee`, `ruptures`, …) |

**Never commit** `config/api_keys.env` or `.env`.

## Project layout

```
research_tool.py          # Thin entry → tav_research/runner.py
tav_research/             # Curses UI, menu tree, registry, data pull
tav_shared/               # LLM analysis, remote AI, run output, paths
menus/                    # Domain modules (prime_past, astronomical, particle, …)
config/                   # api_keys.env.example
scripts/                  # Maintenance utilities
```

## Empirical auto-fetch (`empirical:` targets)

Before each run, `ensure_datasets_before_run` materializes curated empirical tables:

| Module | Actions | Targets |
|--------|---------|---------|
| `PRIME_PAST_HARMONIC` | BBN Interference / Confrontation / Enhanced / Final | `empirical:bbn_abundances`, `empirical:neutron_lifetime` |
| `TSB_RESEARCH` | Fractal Tau Circle Likelihood | `empirical:glueball_lattice`, `empirical:neutron_lifetime` |
| `TSB_RESEARCH` | Residual Diagnostics, Calibrate SoundHorizon | `empirical:neutron_lifetime` |

Config: `tav_shared/empirical_action_targets.py`. Provenance is stored on `options['_empirical_provenance']` and written into JSON reports.

## Automated ingestion validation

Pydantic-backed validation runs before data reaches Tau-Superblock solvers and `log_likelihood`:

| Package | Role |
|---------|------|
| `tav_shared/ingestion/schemas.py` | CMS dimuon, BBN empirical, fractal-tau models |
| `tav_shared/ingestion/normalize.py` | CSV / JSON / ROOT-derived dict → canonical internal format |
| `tav_shared/ingestion/pipeline.py` | `IngestionValidationReport`, batch validators, anomaly filtering |
| `tav_shared/ingestion/config.py` | Physical bounds; override via `TAV_INGESTION_CONFIG` JSON path |

**Integrated entry points:** `extract_dimuon_kinematics_from_nanoaod`, `fetch_empirical_data` (BBN), `log_likelihood` / `log_likelihood_tep_ansatz` (fractal tau).

## Standardized dataset comparison

Apples-to-apples MC vs data and model vs observed checks during preprocessing:

| Package | Role |
|---------|------|
| `tav_shared/dataset_comparison/scaling.py` | `StandardScaler`, `MinMaxScaler` (numpy, sklearn-compatible API) |
| `tav_shared/dataset_comparison/keys.py` | Primary-key rounding, sort, inner-join alignment |
| `tav_shared/dataset_comparison/tolerance.py` | `tolerant_diff`, `compare_mass_gap_mev` with configurable thresholds |
| `tav_shared/dataset_comparison/pipeline.py` | `prepare_datasets_for_comparison`, `compare_histogram_pair`, reports |

Override tolerances and scalers via `TAV_COMPARISON_CONFIG` JSON path.

**Integrated entry points:** `tav_compare_data_mc`, `enhanced_data_vs_mc_comparison`, `normalize_dimuon_events` (key-sorted), `log_likelihood_breakdown` (model vs observed tolerance block).

## MCMC validation (TEP + Morris)

Fractal tau circle MCMC treats validation as a hard prior constraint:

| Module | Role |
|--------|------|
| `menus/tsb_research/fractal_tau_mcmc.py` | `log_prior_fractal_tau`, `log_probability_fractal_tau`, `run_fractal_tau_mcmc` |
| `menus/tsb_research/sensitivity.py` | Morris OAT screening, sensitivity-driven parameter freeze/narrow |
| `menus/tsb_research/fractal_tau_circle.py` | `satisfies_tep_142857_cycle_closure`, `tep_closure_diagnostics` |

**Menu:** TSB Research → **Fractal Tau Circle Likelihood** with `Run TEP-hard MCMC (emcee) = yes`.

Morris screening ranks `winding_density`, `fractal_level`, and `phase_slip_alpha` by impact on the 313.1 MeV floor; low-μ* parameters are frozen before emcee sampling.

## CMS dimuon validation (HEP controls)

Critique-driven upgrades (2026-07-15) for DoubleMuParked / MC comparisons:

| Module | Role |
|--------|------|
| `menus/particle/cms/hep_statistics.py` | Trigger-aware mod-7 null, labeled engine scores, MC role warnings, fine q_T binning |
| `menus/particle/cms/preregistered_recoil_study.py` | Frozen q_T window (0.25–0.40 GeV), shuffle/partition null tests, split-sample stability |
| `menus/particle/cms/validation_extension.py` | Menu: **Preregistered Recoil q_T Study (HEP Controls)** |

**CERN Open Data → Preregistered Recoil q_T Study (HEP Controls)** runs the preregistered pipeline.
**Run MC Validation Suite** now labels Higgs MC as dedicated signal (not inclusive SM background) and reports internal engine scores separately from particle-physics significance.

Reference: `Personal_Files/cms_dimuon_critique_hep_controls_2026-07-15.md`

## CLI tools (BBN)

| Script | Description |
|--------|-------------|
| `tav_bbn_confrontation.py` | RK45 BBN scanner with plots and JSON |
| `enhanced_tav_bbn_confrontation.py` | 5-state BBN, Li7 heatmap |
| `final_tav_bbn_confrontation_tool.py` | Best-config export and comparison plot |
| `tav_bbn_menu_module.py` | Tav BBN submenu engine |

## Dependencies

Core scientific stack: NumPy, SciPy, Matplotlib, Astropy, healpy, uproot/xrootd for ROOT/FITS I/O, plus [`tav-resonance`](https://pypi.org/project/tav-resonance/) for harmonic FRB/CMB analysis.

Large datasets and run artifacts are **not** in the repo — they are downloaded or written under `datasets/` and `artifacts/` at runtime (both gitignored).

## Publishing to GitHub

This directory is a self-contained git export (no venv, artifacts, or secrets). After creating a repo on GitHub:

```bash
cd /path/to/research_tool
./scripts/push_to_github.sh git@github.com:YOUR_USER/YOUR_REPO.git
```


Or set the remote yourself:

```bash
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git push -u origin main
```

## License
