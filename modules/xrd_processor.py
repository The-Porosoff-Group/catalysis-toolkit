"""
modules/xrd_processor.py
XRD module entry point for app.py.
Delegates to the xrd/ package.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from xrd import MODULE_INFO, run, parse_xrd_file, COMMON_WAVELENGTHS
from xrd.cod_api import search_by_elements, search_by_name, fetch_cif, get_stick_pattern
