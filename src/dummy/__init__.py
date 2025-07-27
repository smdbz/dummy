"""
Dummy data generator.

Creates a DataFrame with fake data.
"""
# Standard library imports
from importlib import resources

try:
    import pandas
except ModuleNotFoundError:
    # Third party imports
    import pandas as pd

try:
    import numpy
except ModuleNotFoundError:
    # Third party imports
    import numpy as np


from .dummy import GenerateData

__version__ = "0.1.0"
__all__ = ["GenerateData"]
