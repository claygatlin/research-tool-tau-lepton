import uproot
import awkward as ak
import numpy as np
import pandas as pd

def fetch_cms_data(file_url, tree_name="Events"):
    """
    Connects to a remote CMS Open Data .root file and extracts physics objects.
    """
    print(f"\n[TAV ENGINE] Piercing CMS domain: {file_url}")
    try:
        # Connect to the remote ROOT file
        with uproot.open(file_url) as file:
            tree = file[tree_name]
            # Extract electron/photon tracks (Example keys: 'Electron_pt', 'Electron_eta')
            data = tree.arrays(["Electron_pt", "Electron_eta", "Electron_phi"], library="ak")
            
            # Convert to Pandas for correlation analysis
            df = ak.to_pandas(data)
            print(f"[TAV ENGINE] CMS Macroscopic arrays acquired.")
            return df
    except Exception as e:
        print(f"[TAV ENGINE ERROR] CMS domain link failed: {e}")
        return None

if __name__ == "__main__":
    # Test link: A standard CMS NanoAOD sample
    test_url = "https://opendata.cern.ch/record/12345/files/nanoaod.root"
    df = fetch_cms_data(test_url)
    if df is not None:
        print(df.head())
