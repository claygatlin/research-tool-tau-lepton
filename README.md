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

## Remote Public archive (SFTP)

Finished datasets are staged locally under `.tav_project_staging/`, then uploaded via
SFTP/SCP to:

```text
willieb@10.0.0.183:/home/willieb/Public/ProtonDrive/tav_project/
```

Corpus sync (`/sync` in the gateway, or `scripts/sync_archive.sh`) targets:

```text
willieb@10.0.0.183:/home/willieb/Public/tsb_sync/
```

Disable remote push: `export TAV_REMOTE_TRANSFER=0`.

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

Finished archives for this project live on the remote host under
`willieb@10.0.0.183:/home/willieb/Public/` (SFTP), not a local Public path.

Or set the remote yourself:

```bash
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git push -u origin main
```

## License

Research code — see individual module headers and the `tav-resonance` package for library licensing.