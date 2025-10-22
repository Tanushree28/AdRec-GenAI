#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 11:39:59 2025

@author: feng
"""

import os
import json
import pickle
from pathlib import Path

folder_path = Path("../TencentGR_1k/TencentGR_1k/creative_emb/emb_82_1024/")

# Store as {anonymous_cid: embedding}
emb_dict = {}

# Loop through each part file
for filename in sorted(os.listdir(folder_path)):
    
    if filename.startswith("part-"):  # match your naming pattern
        print(folder_path / filename)
        with open(folder_path / filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # skip empty lines
                    obj = json.loads(line)
                    emb_dict[obj["anonymous_cid"]] = obj.get("emb", None)  # None if no emb
                    

# Save as pickle
output_path = folder_path / f"emb_82_31024.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(emb_dict, f)

print(f"Saved {len(emb_dict)} embeddings to {output_path}")
