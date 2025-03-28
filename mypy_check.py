#!/usr/bin/env python3
"""Run MyPy checks with the correct setup."""

import os
import sys
import subprocess

if __name__ == "__main__":
    # Ensure we check from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Run mypy excluding docs
    cmd = ["mypy", "--exclude", "docs/", "prompt_tuner"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)