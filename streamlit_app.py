# Redirect to the main app
import sys
import os

# Add eMptyCommerce folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eMptyCommerce'))

# Import and run the app
from app import *
