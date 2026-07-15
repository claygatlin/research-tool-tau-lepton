"""
Curated CERN Open Data targets — record IDs for programmatic fetch (no browser).
"""

from __future__ import annotations

from typing import Any

# Education / skim samples sized for native uproot analysis on a laptop.
CERN_MANIFEST: dict[str, dict[str, Any]] = {
    "cms_nanoaod_dimu": {
        "recid": 12341,
        "label": "CMS NanoAOD DoubleMuParked (education)",
        "filter_regexp": r"\.root$",
        "group": "cms",
        "analyze": "cms_nanoaod",
    },
    "cms_nanoaod_higgs_zz": {
        "recid": 12361,
        "label": "CMS NanoAOD Higgs→4L (education)",
        "filter_regexp": r"\.root$",
        "group": "cms",
        "analyze": "cms_nanoaod",
    },
    "cms_nanoaod_dy": {
        "recid": 12353,
        "label": "CMS NanoAOD DYJetsToLL (education)",
        "filter_regexp": r"\.root$",
        "group": "cms",
        "analyze": "cms_nanoaod",
    },
    "cms_nanoaod_ttbar": {
        "recid": 12354,
        "label": "CMS NanoAOD TTbar (education)",
        "filter_regexp": r"\.root$",
        "group": "cms",
        "analyze": "cms_nanoaod",
    },
    "cms_nanoaod_doubleelectron": {
        "recid": 12367,
        "label": "CMS NanoAOD DoubleElectron Run2012B (education)",
        "filter_regexp": r"\.root$",
        "group": "cms",
        "analyze": "cms_nanoaod",
    },
    "cms_nanoaod_zz4e": {
        "recid": 12363,
        "label": "CMS NanoAOD ZZTo4e (education)",
        "filter_regexp": r"\.root$",
        "group": "cms",
        "analyze": "cms_nanoaod",
    },
    "cms_higgs_software": {
        "recid": 5500,
        "label": "CMS Higgs example software bundle",
        "filter_regexp": r"\.(py|cc|xml|txt|pdf|png)$",
        "group": "cms",
        "analyze": None,
    },
    "alice_esd_sample": {
        "recid": 1102,
        "label": "ALICE PbPb ESD sample (1 file)",
        "filter_range": "1-1",
        "group": "alice",
        "analyze": "alice_esd",
    },
}

DEFAULT_PULL_TARGETS: tuple[str, ...] = (
    "cms_nanoaod_dimu",
    "cms_nanoaod_dy",
    "cms_nanoaod_ttbar",
    "cms_nanoaod_higgs_zz",
)

# Option 3: expanded MC stack for background-subtracted validation
OPTION3_MC_STACK_KEYS: tuple[str, ...] = (
    "cms_nanoaod_dy",
    "cms_nanoaod_ttbar",
    "cms_nanoaod_higgs_zz",
)

# Option 4: secondary gamma / EM vector-boson crosscheck datasets
OPTION4_PHOTON_CROSSCHECK_KEYS: tuple[str, ...] = (
    "cms_nanoaod_doubleelectron",
    "cms_nanoaod_zz4e",
    "cms_nanoaod_higgs_zz",
)

ACTION_DEFAULT_TARGETS: dict[str, list[str]] = {
    "Analyze CMS NanoAOD": ["cms_nanoaod_dimu"],
    "CMS 7-Fold Muon Tav Analysis": ["cms_nanoaod_dimu"],
    "CMS Full-Dataset 7-Fold Scan": ["cms_nanoaod_dimu"],
    "Analyze ALICE ROOT Sample": ["alice_esd_sample"],
    "Pull Datasets from Open Archives": list(CERN_MANIFEST.keys()),
}


def target_id(key: str) -> str:
    return f"cern:{key}"


def resolve_target_key(query: str) -> str | None:
    """Map menu query / preset name to manifest key."""
    q = (query or "").strip()
    if not q:
        return None
    if q in CERN_MANIFEST:
        return q
    if q.startswith("cern:"):
        key = q.split(":", 1)[-1]
        return key if key in CERN_MANIFEST else None
    if q.isdigit():
        for key, spec in CERN_MANIFEST.items():
            if int(spec["recid"]) == int(q):
                return key
    return None