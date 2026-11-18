"""
backend/main.py
===============
BioPredict — FastAPI Backend
"""
import os
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import backend._compiled_main as _compiled_main
from backend._compiled_main import *
from backend._compiled_main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 65)
    print("  Starting BioPredict Web Platform...")
    print(f"  Web Interface : http://127.0.0.1:{port}/app/")
    print(f"  Swagger Docs   : http://127.0.0.1:{port}/docs")
    print("  Target Enzyme  : Cyclooxygenase-2 (COX-2)")
    print("  Model Engine   : XGBoost Classifier (Morgan 2048-bit)")
    print("=" * 65)
    uvicorn.run("backend.main:app", host=host, port=port)
