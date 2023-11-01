import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt


import ipywidgets as widgets
from IPython.display import display
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, ColorBar
from bokeh.layouts import column
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from scipy.interpolate import CubicSpline
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


def autocorrelation_movie(file_paths, r_values=50, output_dir=None):
    """
    Processes a list of file paths and extracts various autocorrelation metrics.
    
    Parameters:
    - file_paths (list of str): List of paths to data files.
    - r_values (int, optional): Number of r values for the autocorrelation. Default is 50.
    - output_dir (str, optional): Directory path to save the resulting DataFrame. If not provided, data won't be saved.
    
    Returns:
    - DataFrame: Contains columns:
        - 'file_name': Name of the processed file
        - 'time [min]': Computed time in minutes
        - 'Correlation Length': Extracted correlation length
        - 'inverse': Inverse Fourier transform of the magnitude
        - 'results': Autocorrelation results
        - 'fitted_values': Values fitted to the exponential decay model.
    """
    
    def autocorrelation(file_path, r_values):
        """
        Computes autocorrelation metrics for data from a given file path.
        
        Parameters:
        - file_path (str): Path to the data file.
        - r_values (int): Number of r values for the autocorrelation.

        Returns:
        - tuple: Contains:
            - lambda_tau: Computed correlation length
            - inverse: Inverse Fourier transform of the magnitude
            - results: Autocorrelation results
            - fitted_values: Values fitted to the exponential decay model
        """
        # Load and preprocess data
        df = pd.read_csv(file_path, sep=',', skiprows=2)
        df['x [um]'] = df['x [m]'] * 1E6
        df['y [um]'] = df['y [m]'] * 1E6
        df['u [um/s]'] = df['u [m/s]'] * 1E6
        df['v [um/s]'] = df['v [m/s]'] * 1E6
        df = df.fillna(0)
        df = df[['x [um]', 'y [um]', 'u [um/s]', 'v [um/s]']]
        
        # Convert to matrix form for Fourier transformation
        u = df.pivot(index='y [um]', columns='x [um]', values='u [um/s]').values
        v = df.pivot(index='y [um]', columns='x [um]', values='v [um/s]').values
        
        # Compute the magnitude and its Fourier transform
        magnitude = np.sqrt(u**2 + v**2)
        full_product = np.fft.fft2(magnitude) * np.conj(np.fft.fft2(magnitude))
        inverse = np.real(np.fft.ifft2(full_product))
        
        # Compute autocorrelation
        results = np.zeros(r_values)
        for r in list(range(0, r_values)):
            autocorrelation_value = (inverse[r, r] + inverse[-r, -r]) / (magnitude.shape[0] * magnitude.shape[1])
            results[r] = autocorrelation_value
        results = results / results[0]
        
        # Fit to exponential decay model
        def exponential_decay(tau, A, B, C):
            return A * np.exp(-tau/B) + C
        params, _ = curve_fit(exponential_decay, np.arange(len(results)), results, maxfev=5000)
        A, B, C = params
        fitted_values = exponential_decay(np.arange(r_values), A, B, C)
        
        lambda_tau = -B * np.log((0.3 - C) / A) 
        
        return lambda_tau, inverse, results, fitted_values
    
    data = []
    for idx, file_path in enumerate(file_paths):
        lambda_tau, inverse, results, fitted_values = autocorrelation(file_path, r_values)
        time_minutes = (idx * 3) / 60  # Compute time in minutes
        data.append([os.path.basename(file_path), time_minutes, lambda_tau, inverse, results, fitted_values])
    
    # Convert the data to a dataframe
    df = pd.DataFrame(data, columns=['file_name', 'time [min]', 'Correlation Length', 'inverse', 'results', 'fitted_values'])
    
    # Save DataFrame to CSV if output_dir is provided
    if output_dir:
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Define the full path for the CSV file
        csv_path = os.path.join(output_dir, 'autocorrelation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    
    return df


def autocorrelation_csv(path_df):
    """Load the dataset and convert the string representation of numpy arrays to actual numpy arrays."""
    
    df = pd.read_csv(path_df)
    
    # Convert 'results' and 'fitted_values' columns to numpy arrays using lambda functions
    df['results'] = df['results'].apply(lambda s: np.fromstring(s.strip('[]').replace('\n', ''), sep=' '))
    df['fitted_values'] = df['fitted_values'].apply(lambda s: np.fromstring(s.strip('[]').replace('\n', ''), sep=' '))
    
    return df



    # Generate dynamic colors based on number of frames using a colormap
def plot_autocorrelation_values_multiple_frames(df):
    """
    Plots the autocorrelation values and the fitted exponential decay for multiple frames.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'Correlation Length', 'results', and 'fitted_values' columns.
    
    Returns:
    - None: Displays the plot.
    """
    
    plt.figure(figsize=(12, 7))
    

    def generate_dynamic_colors(n):
        """
        Generate a list of distinct colors based on the number of frames using a colormap.

        Parameters:
        - n (int): Number of required colors.

        Returns:
        - list: List of RGBA colors.
        """
        colormap = plt.cm.viridis  # using the 'viridis' colormap, but this can be changed to any other colormap
        return [colormap(i) for i in np.linspace(0, 1, n)]

    # Get dynamic colors based on the number of frames
    colors = generate_dynamic_colors(len(df))
    
    for idx, row in df.iterrows():
        lambda_tau = row['Correlation Length']
        results = row['results']
        fitted_values = row['fitted_values']
        
        # Plot autocorrelation values and fitted exponential decay
        plt.plot(results, marker='o', linestyle='-', markersize=5, color=colors[idx], label=f'Frame {idx} Data')
        plt.plot(fitted_values, linestyle='--', color=colors[idx], label=f'Frame {idx} Fit')
        plt.axvline(x=lambda_tau, color=colors[idx], linestyle='-.', linewidth=0.8, label=f'Correlation Length (Frame {idx}) = {lambda_tau:.2f}')
    
    # Adding labels, title, and legend
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function and Fitted Exponential Decay for Multiple Frames')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()



def plot_histogram(dataframe, bins=None):
    """
    Manually plots a histogram of correlation lengths in viridis colors 
    and fits the data to a normal distribution with a black line.
    
    Parameters:
    - dataframe (DataFrame): Dataframe with the necessary data.
    - bins (int, optional): Number of bins for the histogram. If None, will use the number of unique values.
    
    Returns:
    - mu (float): Mean of the fitted normal distribution.
    - std (float): Standard deviation of the fitted normal distribution.
    """
    # Extract the correlation lengths and clean any non-finite values
    cl_values = dataframe['Correlation Length'].dropna().values
    cl_values = cl_values[np.isfinite(cl_values)]
    
    # Set default bins to the number of unique values if not provided
    if bins is None:
        bins = len(np.unique(cl_values))
    
    # Compute the histogram manually
    counts, bin_edges = np.histogram(cl_values, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(12, 7))
    
    # Get colormap colors
    colormap = plt.cm.viridis
    colors = colormap(np.linspace(0, 1, len(counts)))
    
    # Plot histogram bars manually with viridis colors
    for center, count, color in zip(bin_centers, counts, colors):
        plt.bar(center, count, width=bin_width, align='center', color=color)
    
    # Fit a normal distribution to the data
    mu, std = norm.fit(cl_values)
    
    # Plot the fitted distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k-', linewidth=2)
    plt.title(f"Fit results: Mean = {mu:.2f}, Std. Dev. = {std:.2f}")
    plt.xlabel('Correlation Lengths')
    plt.ylabel('Density')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    return mu, std


def histogram_animated(dataframe):
    # Extract the correlation lengths and clean any non-finite values
    cl_values = dataframe['Correlation Length'].dropna().values
    cl_values = cl_values[np.isfinite(cl_values)]
    max_bins = len(cl_values)

    @widgets.interact(bins=widgets.IntSlider(min=1, max=max_bins, step=1, value=max_bins, description='Bins:'))
    def interactive_plot(bins):
        plt.figure(figsize=(12, 7))

        # Compute the histogram manually
        counts, bin_edges = np.histogram(cl_values, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Get colormap colors
        colormap = plt.cm.viridis
        colors = colormap(np.linspace(0, 1, len(counts)))

        # Plot histogram bars manually with viridis colors
        for center, count, color in zip(bin_centers, counts, colors):
            plt.bar(center, count, width=bin_width, align='center', color=color)

        # Fit a normal distribution to the data
        mu, std = norm.fit(cl_values)

        # Plot the fitted distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k-', linewidth=2)
        
        # Vertical dashed line for the mean
        plt.axvline(mu, color='red', linestyle='dashed', linewidth=1.5)
        
        # Title and labels
        plt.title(f"Fit results: Mean = {mu:.2f}, Std. Dev. = {std:.2f}")
        plt.xlabel('Correlation Lengths')
        plt.ylabel('Density')
        
        # Enhance x-axis ticks for more clarity
        plt.xticks(np.linspace(xmin, xmax, min(20, bins)))
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()



def plot_autocorrelation_length_vs_time(df):
    """
    Plots the correlation length against time with interpolation, filtering out non-finite values.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'time [min]' and 'Correlation Length' columns.
    
    Returns:
    - None: Displays the plot.
    """
    
    plt.figure(figsize=(14, 8))
    
    # Filter out non-finite values
    df_filtered = df[df['Correlation Length'].notna() & np.isfinite(df['Correlation Length'])]
    
    # Use scatter for heatmap coloring
    plt.scatter(df_filtered['time [min]'], df_filtered['Correlation Length'], 
                c=df_filtered['Correlation Length'], cmap='viridis', s=100, edgecolor='black')
    
    # Interpolate using cubic spline
    cs = CubicSpline(df_filtered['time [min]'], df_filtered['Correlation Length'])
    times = np.linspace(df_filtered['time [min]'].min(), df_filtered['time [min]'].max(), 500)
    plt.plot(times, cs(times), 'r-')
    
    # Add labels, title, and colorbar
    plt.colorbar(label='Correlation Length')
    plt.xlabel('Time (min)')
    plt.ylabel('Correlation Length')
    plt.title('Correlation Length vs. Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_correlation_length_vs_time_interactive(df_correlation_length):
    """
    Plot correlation length against time using provided DataFrame with Bokeh, color the y-values with a heatmap in viridis,
    and connect the points with an interpolated line. Tooltips display the correlation length on hover.
    
    Parameters:
    - df_correlation_length (DataFrame): DataFrame with columns for time (in minutes) and correlation length.
    
    Returns:
    - None: Displays the interactive plot.
    """
    df_correlation_length["Correlation Length"] = df_correlation_length["Correlation Length"].fillna(df_correlation_length["Correlation Length"].mean())

    # Interpolate using cubic spline
    cs = CubicSpline(df_correlation_length['time [min]'], df_correlation_length['Correlation Length'])
    times = np.linspace(df_correlation_length['time [min]'].min(), df_correlation_length['time [min]'].max(), 500)
    
    # Set up the Bokeh plot
    p = figure(width=800, height=400, title='Correlation Length vs. Time with Interpolation',
               x_axis_label='time [min]', y_axis_label='Correlation Length', tools='')
    
    # Add the data points with hover info
    mapper = linear_cmap(field_name='Correlation Length', palette=Viridis256, 
                         low=min(df_correlation_length['Correlation Length']), 
                         high=max(df_correlation_length['Correlation Length']))
    
    source = ColumnDataSource(df_correlation_length)
    p.scatter(x='time [min]', y='Correlation Length', source=source, size=10, color=mapper, alpha=1)  # Added alpha for transparency
    
    # Add the interpolated line (after scatter to ensure it's on top)
    p.line(times, cs(times), line_color='orange', line_width=2)
    
    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [("Correlation Length", "@{Correlation Length}"), ("File Name", "@{file_name}")]
    p.add_tools(hover)
    
    # Color bar
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    p.add_layout(color_bar, 'right')
    
    output_notebook()
    show(p)