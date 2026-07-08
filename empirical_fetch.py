#!/usr/bin/env python3
"""
Script-facing entry point for PDG/literature empirical anchors.

Usage:
    python empirical_fetch.py

    from empirical_fetch import fetch_empirical_data
    empirical = fetch_empirical_data()
"""

from menus.empirical_tests.tests import fetch_empirical_data

__all__ = ["fetch_empirical_data"]


if __name__ == "__main__":
    empirical = fetch_empirical_data()
    print(empirical["bbn_abundances"])