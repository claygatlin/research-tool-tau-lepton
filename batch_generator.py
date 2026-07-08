import sys
import os

def generate_batch_from_index(input_file, output_file="batch_data.txt"):
    """
    Reads index file and extracts only the Dataset ID for the HEPData engine.
    Example: Extracts '1803608' from lines or paths.
    """
    if not os.path.exists(input_file):
        print(f"[TAV ENGINE ERROR] Source index file '{input_file}' not found.")
        return

    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            count = 0
            for line in f_in:
                clean_line = line.strip()
                # Skip empty lines or headers
                if not clean_line or clean_line.startswith('['):
                    continue
                
                # Logic: If your ALICE data requires numerical IDs, we need the ID 
                # instead of the root:// path. If the root:// path IS the ID, 
                # we need to shorten it. Here we assume you want the numeric ID 
                # identified in the file name or path.
                
                # If your file index contains the ID as a specific segment, 
                # we extract it. For now, we clean the line:
                ds_id = clean_line.split('/')[-2] # Extracts the ID folder segment
                
                f_out.write(f"{ds_id}\n")
                count += 1
        
        print(f"[TAV ENGINE] Batch file '{output_file}' generated with {count} targets.")
    except Exception as e:
        print(f"[TAV ENGINE ERROR] Failed to process index file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_batch_from_index(sys.argv[1])
    else:
        print("[TAV ENGINE] Usage: ./venv/bin/python3 batch_generator.py <INDEX_FILE>")
