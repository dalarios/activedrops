# Import plotting utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # make plots prettier
import os
import matplotlib.gridspec as gridspec
import glob
from scipy.ndimage import gaussian_filter
import colorcet as cc

"""
Title:
    viz.py
Last update:
    2023-10-03
Author(s):
    David Larios
Purpose:
    This file compiles all of the relevant functions for processing raw
    PIV data from Matlab PIVlab for the ActiveDROPS project.
"""


def process_piv_data(link, µl, vmax=True, time_int=3, L=100, sigma=5, 
                     skip_frames=0, remove_end_frames=0, save_path=None):
    """
    Processes PIV data from a series of text files and calculates various physical quantities.
    
    Parameters:
    -link (str): The path to the text files containing the data.
    -µl (float): The volume in microliters used in calculations.
    -vmax (bool): Whether to use the average of the top 10 velocities or the mean velocity.
    -time_int (int): The time interval between measurements, used to calculate time in minutes.
    -L (int): The correlation length in microns, used in calculations.
    -sigma (float): The standard deviation of the Gaussian kernel, used in the Gaussian filter.
    -skip_frames (int): The number of initial frames to skip.
    -remove_end_frames (int): The number of final frames to remove.
    
    Returns:
    pd.DataFrame: A DataFrame containing time, velocities, vorticities, divergences, distance, 
                  power, work, drag force, and cumulative drag force.
    """
    
    # Get sorted list of text files, skip the initial frames and remove the end frames as specified
    files = sorted(glob.glob(link))[skip_frames:-remove_end_frames or None]
    
  # Initialize lists to store extracted magnitudes and number of vectors
    velocities, vorticities, divergences, num_vectors = [], [], [], []
       
    for file in files:
        df = pd.read_csv(file, skiprows=2).fillna(0)
        
        # Determine method to calculate velocity
        if vmax:
            n = int(0.1 * len(df))  # Top 10% of the vectors
            v = df['magnitude [m/s]'].nlargest(n).mean()
        else:
            v = df['magnitude [m/s]'].mean()
        
        velocities.append(v)
        vorticities.append(df["vorticity [1/s]"].mean())
        divergences.append(df["divergence [1/s]"].mean())
        
        # Store the number of vectors for this frame
        num_vectors.append(len(df))
    
    # Create a DataFrame with time and processed velocities
    result_df = pd.DataFrame(velocities, columns=["v (m/s)"]).reset_index()
    result_df['time_min'] = result_df['index'] * time_int / 60
    result_df['v (m/s)'] = gaussian_filter(result_df['v (m/s)'] - result_df['v (m/s)'].min(), sigma=sigma)
    
    # Calculate additional physical quantities
    result_df['vorticity (1/s)'], result_df['divergence (1/s)'] = vorticities, divergences
    result_df['num_vectors'] = num_vectors  # Add the number of vectors per frame
    result_df['distance (m)'] = (result_df['v (m/s)'] * result_df['time_min'].diff()).cumsum()
    
    vol, µ, corr = µl * 1E-9, 1E-3, L * 1E-6  # Convert units for volume, viscosity, and correlation length
    result_df["Power (W)"] = vol * µ * (result_df["v (m/s)"] / corr) ** 2
    result_df['Work (J)'] = (result_df["Power (W)"] * result_df['time_min'].diff()).cumsum()

    
    # Re-order columns
    new_order = ['time_min', 'v (m/s)', 'index', 'vorticity (1/s)', 'divergence (1/s)',
                 'num_vectors', 'distance (m)', 'Power (W)', 'Work (J)']
    result_df = result_df[new_order]
    
    if save_path is not None:
        result_df.to_csv(save_path, index=False)
        
    return result_df