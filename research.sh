#!/bin/bash

# 1. Generate the batch list
echo "[TAV ENGINE] Generating batch list..."
./venv/bin/python3 batch_generator.py

# 2. Run the research processor
echo "[TAV ENGINE] Initiating Research Processor..."
./venv/bin/python3 research_tool.py

# 3. Perform cleanup
echo "[TAV ENGINE] Performing post-run housekeeping..."
./venv/bin/python3 housekeeping.py
