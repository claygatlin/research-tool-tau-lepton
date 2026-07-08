import os

def finalize_batch(batch_filename="batch_data.txt", ignore_filename="ignore.txt"):
    """
    Appends processed batch data to ignore.txt and clears the batch file.
    """
    if os.path.exists(batch_filename) and os.path.getsize(batch_filename) > 0:
        with open(batch_filename, 'r') as batch_file:
            entries = batch_file.readlines()
        
        # Append entries to ignore.txt
        with open(ignore_filename, 'a') as ignore_file:
            ignore_file.writelines(entries)
            
        # Reset batch_data.txt
        open(batch_filename, 'w').close()
        print(f"\n[TAV ENGINE] Batch '{batch_filename}' archived to '{ignore_filename}'.")
    else:
        print(f"\n[TAV ENGINE] No active batch data found in '{batch_filename}' to archive.")

if __name__ == "__main__":
    finalize_batch()
