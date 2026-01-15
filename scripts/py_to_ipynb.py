import json
import os
import re
import textwrap

def py_to_ipynb(py_path, ipynb_path):
    print(f"Converting {py_path} to {ipynb_path}...")
    
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the specific markers used in our format
    # # %% [markdown] and # %% [code]
    
    # Simple state machine parser
    lines = content.splitlines()
    cells = []
    current_cell_type = None
    current_source = []

    def save_current_cell():
        if current_cell_type and (current_source or current_cell_type == 'code'):
            # Dedent the block first
            full_source = "".join(line + "\n" for line in current_source)
            dedented_source = textwrap.dedent(full_source)
            # Split back into lines for processing
            source_lines = dedented_source.splitlines(True) # Keep newlines

            # For markdown, remove the leading "# " if present
            cleaned_source = []
            if current_cell_type == 'markdown':
                for line in source_lines:
                    # Remove newline for strip check
                    line_content = line.rstrip()
                    if line_content.startswith("# "):
                        cleaned_source.append(line[2:]) # Keep newline from original if needed? No, splitlines(True) keeps it.
                        # Wait, line slice [2:] keeps newline if it was there
                    elif line_content == "#":
                         cleaned_source.append("\n")
                    else:
                        # Should unlikely happen given our conversion format, but just in case
                        cleaned_source.append(line)
            else:
                 cleaned_source = source_lines

            # Remove trailing newline from the last line if exists
            # (Just for neatness, though Jupyter cells often don't end with newline in json)
            if cleaned_source and cleaned_source[-1].endswith("\n"):
                cleaned_source[-1] = cleaned_source[-1][:-1]

            cell = {
                "cell_type": current_cell_type,
                "metadata": {},
                "source": cleaned_source
            }
            if current_cell_type == 'code':
                cell["execution_count"] = None
                cell["outputs"] = []
            
            cells.append(cell)

    for line in lines:
        if line.strip() == "# %% [markdown]":
            save_current_cell()
            current_cell_type = 'markdown'
            current_source = []
        elif line.strip() == "# %% [code]":
            save_current_cell()
            current_cell_type = 'code'
            current_source = []
        elif current_cell_type:
            current_source.append(line)
        else:
            # Content before the first marker? ignore or handle?
            # For this specific case, we can probably ignore imports/comments before first block if any
            pass
            
    # Save last cell
    save_current_cell()

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
           "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
           }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print("Conversion complete.")

if __name__ == "__main__":
    # Hardcoded paths as per requirement/context, but could be arguments
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PY_FILE = os.path.join(PROJECT_ROOT, "src", "notebooks", "report", "report.py")
    IPYNB_FILE = os.path.join(PROJECT_ROOT, "src", "notebooks", "report", "report.ipynb")
    
    if os.path.exists(PY_FILE):
        py_to_ipynb(PY_FILE, IPYNB_FILE)
    else:
        print(f"Error: Source file {PY_FILE} not found.")
