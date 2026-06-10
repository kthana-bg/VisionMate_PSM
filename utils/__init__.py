import sys, os
# Ensure project root is always on sys.path for Streamlit Cloud
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
