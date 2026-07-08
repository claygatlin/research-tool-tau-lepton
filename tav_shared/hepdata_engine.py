import requests
import pandas as pd
from urllib.parse import quote

def fetch_hepdata_record(inspire_id):
    """
    Phase 1: Fetches the metadata to map the available tables in the domain.
    """
    clean_id = str(inspire_id).lower().replace('ins', '')
    api_url = f"https://www.hepdata.net/record/ins{clean_id}?format=json"
    print(f"\n[TAV ENGINE] Initiating Phase 1: Piercing domain metadata at {api_url}")
    
    try:
        response = requests.get(api_url, timeout=15)
        if response.status_code != 200:
            print(f"[TAV ENGINE ERROR] HTTP Status: {response.status_code}")
            return None
            
        record_data = response.json()
        if 'data_tables' not in record_data:
            print("[TAV ENGINE ERROR] No data_tables index found.")
            return None
            
        tables = record_data['data_tables']
        print(f"[TAV ENGINE] Phase 1 Complete. Found {len(tables)} table nodes.")
        return (clean_id, tables)
        
    except Exception as e:
        print(f"[TAV ENGINE ERROR] Phase 1 failed: {e}")
        return None

def process_and_filter_table(record_info, table_index=0):
    """
    Phase 2: Fetches the exact numerical arrays for the specific table and maps them.
    """
    if not record_info:
        return None
        
    clean_id, tables_metadata = record_info
    
    try:
        # 1. Identify the exact table name from Phase 1
        target_table = tables_metadata[table_index]
        table_name = target_table.get('name', f'Table {table_index + 1}')
        
        # URL-encode the table name (e.g., "Table 1" becomes "Table%201")
        safe_table_name = quote(table_name)
        
        # 2. Ping the explicit discrete table download endpoint (Targeting Version 1)
        data_url = f"https://www.hepdata.net/download/table/ins{clean_id}/{safe_table_name}/1/json"
        print(f"[TAV ENGINE] Initiating Phase 2: Extracting macroscopic arrays from {data_url}")
        
        response = requests.get(data_url, timeout=15)
        
        if response.status_code != 200:
            print(f"[TAV ENGINE ERROR] CERN rejected Phase 2 link. HTTP Status: {response.status_code}")
            return None
            
        # 3. Safely attempt to parse the payload, catching HTML firewalls
        try:
            table_payload = response.json()
        except Exception:
            print(f"[TAV ENGINE ERROR] Payload is not JSON. CERN shielded the array with HTML.")
            print(f"[TAV ENGINE DEBUG] Server response snippet: {response.text[:100]}")
            return None
        
# 4. Directly map the unshielded arrays to Pandas
        print(f"[TAV ENGINE] Arrays acquired. Constructing live matrix for {table_name}...")
        
        if 'values' not in table_payload:
            print(f"[TAV ENGINE ERROR] 'values' matrix missing. Keys found: {table_payload.keys()}")
            return None
            
        try:
            # Load the raw nested rows
            raw_df = pd.json_normalize(table_payload['values'])
            
            # Define translation functions to extract pure floats from the nested lists
            def parse_x(x_list):
                x_data = x_list[0]
                # If it's an exact point, return it. If it's a bin, return the midpoint.
                if 'value' in x_data:
                    return float(x_data['value'])
                return (float(x_data.get('high', 0)) + float(x_data.get('low', 0))) / 2.0
                
            def parse_y(y_list):
                y_data = y_list[0]
                return float(y_data.get('value', 0))
                
            # Construct the finalized, mathematically ready DataFrame
            df = pd.DataFrame()
            df['Kinematic_X'] = raw_df['x'].apply(parse_x)
            df['Observable_Y'] = raw_df['y'].apply(parse_y)
            
            return df

        except Exception as e:
            print(f"[TAV ENGINE ERROR] Matrix flattening failed: {e}")
            return None
            
        # 5. Map the arrays to Pandas for Tav-Superblock injection
        print(f"[TAV ENGINE] Arrays acquired. Constructing live matrix for {table_name}...")
        
        indep_vars = extracted_table.get('independent_variables', [])
        x_raw = indep_vars[0]['values']
        x_parsed = [float(v.get('value', v.get('high', 0))) for v in x_raw]
        x_label = indep_vars[0].get('header', {}).get('name', 'Kinematic_X')
        
        dep_vars = extracted_table.get('dependent_variables', [])
        y_raw = dep_vars[0]['values']
        y_parsed = [float(v.get('value', 0)) for v in y_raw]
        y_label = dep_vars[0].get('header', {}).get('name', 'Observable_Y')
        
        df = pd.DataFrame({
            x_label: x_parsed,
            y_label: y_parsed
        })
        
        return df

    except Exception as e:
        print(f"[TAV ENGINE ERROR] Phase 2 mapping failed: {e}")
        return None
        
# --- Local Testing Block ---
if __name__ == "__main__":
    test_id = "1803608" 
    record_info = fetch_hepdata_record(test_id)
    
    if record_info:
        tav_dataframe = process_and_filter_table(record_info, table_index=0)
        
        if tav_dataframe is not None:
            print("\n[TAV ENGINE] Live Data Ready for Cross-Domain Correlation:")
            print(tav_dataframe.head())
        else:
            print("\n[TAV ENGINE ERROR] DataFrame construction failed. Aborting matrix injection.")
