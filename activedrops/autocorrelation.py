import pandas as pd
import numpy as np
from scipy.signal import correlate2d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
Title:
    autocorrelation.py
Last update:
    2023-10-03
Author(s):
    David Larios
Purpose:
    This file compiles all of the relevant functions for processing raw
    PIV data from Matlab PIVlab for the ActiveDROPS project.
"""

def load_and_convert_data(file_path):
    df = pd.read_csv(file_path, sep=',', skiprows=2)
    
    # Convert position and velocity to microns and microns/second
    df['x [um]'] = df['x [m]'] * 1E6
    df['y [um]'] = df['y [m]'] * 1E6
    df['u [um/s]'] = df['u [m/s]'] * 1E6
    df['v [um/s]'] = df['v [m/s]'] * 1E6
    
    # Drop the original columns and retain only the converted ones
    df = df[['x [um]', 'y [um]', 'u [um/s]', 'v [um/s]']]
    
    return df



