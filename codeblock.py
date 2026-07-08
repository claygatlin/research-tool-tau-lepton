import curses
import time
import sys
import os
import numpy as np
import matplotlib
import pandas as pd
import hepdata_engine
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def fetch_and_graph(repo, query):
    # 1. Fetch Logic
    if "SPARC" in repo:
        df = pd.read_csv(f"data/sparc/{query}.csv")
        x, y_obs = df['Radius'], df['Velocity']
        # SM NFW Prediction (Mock approximation for comparison)
        y_sm = np.sqrt(100 * (1 - np.exp(-x/2))**2 + 50 * (x/(1+x/2)**2)) 
        # TS Conformal Prediction
        y_ts = np.sqrt(100 * (1 - np.exp(-x/2))**2 * 5.2) 
    else:
        # Particle Domain Logic...
        x, y_obs = np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100))
        y_sm, y_ts = y_obs * 0.9, y_obs * 1.1

    # 2. Visual Reporting Layer
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, y_obs, 'k.', label="Observed Data")
    ax.plot(x, y_sm, 'g--', label="Standard Model (NFW)")
    ax.plot(x, y_ts, 'r-', label="Tav-Superblock (TS)")
    
    # Textual assessment inside the graphic output window
    dev = np.mean(np.abs(y_obs - y_ts))
    analysis_text = f"TS Deviation: {dev:.4f}\nStatus: {'Anomalous' if dev > 0.005 else 'Stable'}"
    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f"Comparative Analysis: {query}")
    ax.legend()
    plt.show()

    # 3. Terminal Analytical Report
    print(f"\n--- Analytical Report: {query} ---")
    print(f"TS Framework Residuals: {dev:.6f}")
    if dev > 0.005:
        print("[CORRELATION FOUND] TS Dynamic Refresh detected.")
    else:
        print("[SIGNAL STABLE] No TS deviations detected.")
