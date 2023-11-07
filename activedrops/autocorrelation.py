import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt


import ipywidgets as widgets
from ipywidgets import interact, IntSlider
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


def autocorrelation_movie(file_paths, sint=3, r_values=50, output_dir=None):
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
        
        # u = u - np.mean(u)
        # v = v - np.mean(v)
        
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
        
        lambda_tau = -B * np.log(((1/np.e) - C) / A) 
        
        return lambda_tau, inverse, results, fitted_values
    
    data = []
    for idx, file_path in enumerate(file_paths):
        lambda_tau, inverse, results, fitted_values = autocorrelation(file_path, r_values)
        time_minutes = (idx * sint) / 60  # Compute time in minutes
        data.append([os.path.basename(file_path), time_minutes, lambda_tau, inverse, results, fitted_values])
    
    # Convert the data to a dataframe
    df = pd.DataFrame(data, columns=['file_name', 'time [min]', 'Correlation Length', 'inverse', 'results', 'fitted_values'])
    df['Correlation Length'] = df['Correlation Length'] * 22  # Convert to microns
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



def plot_autocorrelation_values_multiple_frames(df):
    """
    Plots the autocorrelation values and the fitted exponential decay for multiple frames,
    with the lag values scaled by a factor of 22.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'Correlation Length', 'results', and 'fitted_values' columns.
    
    Returns:
    - None: Displays the plot.
    """
    # Set the size of the plot
    plt.figure(figsize=(12, 7))

    def generate_dynamic_colors(n):
        """
        Generate a list of distinct colors based on the number of frames using a colormap.
        
        Parameters:
        - n (int): Number of required colors.
        
        Returns:
        - list: List of RGBA colors.
        """
        colormap = plt.cm.viridis  # Using the 'viridis' colormap
        return [colormap(i) for i in np.linspace(0, 1, n)]

    # Get dynamic colors based on the number of frames
    colors = generate_dynamic_colors(len(df))
    
    # Iterate over each row to plot the data and fits
    for idx, row in df.iterrows():
        lambda_tau = row['Correlation Length']
        results = row['results']
        fitted_values = row['fitted_values']
        lags = np.arange(len(results)) * 22  # Scale the lag values by 22

        # Plot autocorrelation values and fitted exponential decay
        plt.plot(lags, results, marker='o', linestyle='-', markersize=5, color=colors[idx], label=f'Frame {idx} Data')
        plt.plot(lags, fitted_values, linestyle='--', color=colors[idx], label=f'Frame {idx} Fit')
        plt.axvline(x=lambda_tau, color=colors[idx], linestyle='-.', linewidth=0.8, label=f'Correlation Length (Frame {idx}) = {lambda_tau:.2f}')
    
    # Adding labels, title, and legend to the plot
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function and Fitted Exponential Decay for Multiple Frames')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust the layout to fit all components
    plt.show()  # Display the plot




def plot_histogram(dataframe, bins=None):
    """
    Plots a histogram of correlation lengths using viridis colors and displays the mean and standard deviation.
    
    Parameters:
    - dataframe (DataFrame): Dataframe with the necessary data.
    - bins (int, optional): Number of bins for the histogram. If None, will use the number of unique values.
    
    Returns:
    - None: The function plots the histogram and shows it.
    """
    # Extract the correlation lengths and clean any non-finite values
    cl_values = dataframe['Correlation Length'].dropna().values
    cl_values = cl_values[np.isfinite(cl_values)]
    
    # Calculate mean and standard deviation
    mean_value = np.mean(cl_values)
    std_dev = np.std(cl_values)
    
    # Set default bins to the number of unique values if not provided
    if bins is None:
        bins = len(np.unique(cl_values))
    
    # Compute the histogram manually
    counts, bin_edges = np.histogram(cl_values, bins=bins, density=True)
    
    plt.figure(figsize=(12, 7))
    
    # Get colormap colors
    colormap = plt.cm.viridis
    normalize = plt.Normalize(vmin=min(counts), vmax=max(counts))
    colors = colormap(normalize(counts))
    
    # Plot histogram bars manually with viridis colors
    for edge_left, count, color in zip(bin_edges[:-1], counts, colors):
        plt.bar(edge_left, count, width=bin_edges[1] - bin_edges[0], align='edge', color=color)
    
    # Create a legend
    plt.legend(['Mean: {:.2f}\nStd: {:.2f}'.format(mean_value, std_dev)])
    
    plt.title('Histogram of Correlation Lengths')
    plt.xlabel('Correlation Lengths')
    plt.ylabel('Density')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return None


    


def plot_histogram_animated(dataframe):
    """
    Plots an interactive histogram of correlation lengths using viridis colors, with a slider to adjust the number of bins,
    and a legend showing the mean and standard deviation.
    
    Parameters:
    - dataframe (DataFrame): Dataframe with the necessary data.
    
    Returns:
    - None: The function creates an interactive histogram plot.
    """
    # Extract the correlation lengths and clean any non-finite values
    cl_values = dataframe['Correlation Length'].dropna().values
    cl_values = cl_values[np.isfinite(cl_values)]
    max_bins = len(np.unique(cl_values))  # Maximum number of bins for the slider
    
    # Calculate mean and standard deviation
    mean_value = np.mean(cl_values)
    std_dev = np.std(cl_values)

    @interact(bins=IntSlider(min=1, max=max_bins, step=1, value=200, description='Bins:'))
    def update_histogram(bins):
        plt.figure(figsize=(12, 7))
        
        # Compute the histogram manually
        counts, bin_edges = np.histogram(cl_values, bins=bins, density=True)
        
        # Get colormap colors
        colormap = plt.cm.viridis
        normalize = plt.Normalize(vmin=min(counts), vmax=max(counts))
        colors = colormap(normalize(counts))
        
        # Clear the current axes
        plt.gca().clear()
        
        # Plot histogram bars manually with viridis colors
        for edge_left, count, color in zip(bin_edges[:-1], counts, colors):
            plt.bar(edge_left, count, width=bin_edges[1] - bin_edges[0], align='edge', color=color)
        
        # Create a legend for mean and standard deviation
        legend_label = f'Mean: {mean_value:.2f}\nStd Dev: {std_dev:.2f}'
        plt.legend([legend_label])
        
        plt.title('Interactive Histogram of Correlation Lengths')
        plt.xlabel('Correlation Lengths')
        plt.ylabel('Density')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    return None






def plot_autocorrelation_length_vs_time(df, time_hours=False):
    """
    Plots the autocorrelation length as a function of time with interpolation, providing a visual representation 
    of how the correlation length evolves. It filters out non-finite values before plotting and can represent 
    time in minutes or hours.

    This function uses a scatter plot to represent individual data points with heatmap coloring based on the 
    correlation length values and applies a cubic spline interpolation to create a smooth curve that goes through 
    the data points.

    Parameters:
    - df (DataFrame): A pandas DataFrame containing the data to plot. It must include the following columns:
        - 'time [min]': The time values in minutes.
        - 'Correlation Length': The correlation length values corresponding to each time point.
    - time_hours (bool): If True, the time axis will be plotted in hours instead of minutes.

    Returns:
    - None: The function does not return any values. Instead, it displays a plot with the interpolated correlation 
            length over time. The plot includes a colorbar indicating the magnitude of the correlation length, axis 
            labels, a title, and a grid for easier interpretation of the data.

    Note:
    - The function assumes that the input DataFrame is pre-processed to contain the necessary columns and that the 
      'Correlation Length' column contains numerical data.
    - The cubic spline interpolation requires that there are no NaNs or infinite values in the 'Correlation Length' 
      column, hence the pre-filtering.
    """
    
    plt.figure(figsize=(14, 8))
    
    # Filter out non-finite values
    df_filtered = df[df['Correlation Length'].notna() & np.isfinite(df['Correlation Length'])]
    
    # Determine the time for plotting and labeling
    time_label = 'Time (hours)' if time_hours else 'Time (min)'
    times = df_filtered['time [min]'] / 60 if time_hours else df_filtered['time [min]']
    
    # Use scatter for heatmap coloring
    plt.scatter(times, df_filtered['Correlation Length'], 
                c=df_filtered['Correlation Length'], cmap='viridis', s=100, edgecolor='black')
    
    # Interpolate using cubic spline
    cs = CubicSpline(df_filtered['time [min]'], df_filtered['Correlation Length'])
    time_values_for_interpolation = np.linspace(df_filtered['time [min]'].min(), df_filtered['time [min]'].max(), 500)
    interpolated_times = time_values_for_interpolation / 60 if time_hours else time_values_for_interpolation
    plt.plot(interpolated_times, cs(time_values_for_interpolation), 'r-')
    
    # Add labels, title, and colorbar
    plt.colorbar(label='Correlation Length')
    plt.xlabel(time_label)
    plt.ylabel('Correlation Length')
    plt.title('Correlation Length vs. Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_correlation_length_vs_time_interactive(df_correlation_length, time_hours=False):
    """
    Plot correlation length against time using provided DataFrame with Bokeh. Points are colored with a heatmap in viridis
    based on the y-values, and they are connected with an interpolated line. Tooltips display the correlation length and
    optionally the file name on hover.

    This interactive plot allows for dynamic exploration of the data points and the trend line, with the ability to zoom,
    pan, and hover for more detailed information.

    Parameters:
    - df_correlation_length (DataFrame): DataFrame with columns for time (in minutes), correlation length, and file name.
    - time_hours (bool): If True, the time axis will be plotted in hours instead of minutes.

    Returns:
    - None: Displays the interactive plot within the Jupyter notebook.
    """
    # Fill NaN with mean of the column
    df_correlation_length["Correlation Length"] = df_correlation_length["Correlation Length"].fillna(df_correlation_length["Correlation Length"].mean())

    # Convert 'time [min]' to hours if necessary
    if time_hours:
        df_correlation_length['time'] = df_correlation_length['time [min]'] / 60
        x_axis_label = 'Time (hours)'
    else:
        df_correlation_length['time'] = df_correlation_length['time [min]']
        x_axis_label = 'Time (min)'

    # Interpolate using cubic spline
    cs = CubicSpline(df_correlation_length['time'], df_correlation_length['Correlation Length'])
    times_interpolated = np.linspace(df_correlation_length['time'].min(), df_correlation_length['time'].max(), 500)

    # Set up the Bokeh plot
    p = figure(width=800, height=400, title='Correlation Length vs. Time with Interpolation',
               x_axis_label=x_axis_label, y_axis_label='Correlation Length', tools='')

    # Color mapping
    mapper = linear_cmap(field_name='Correlation Length', palette=Viridis256,
                         low=min(df_correlation_length['Correlation Length']),
                         high=max(df_correlation_length['Correlation Length']))

    source = ColumnDataSource(df_correlation_length)
    p.scatter(x='time', y='Correlation Length', source=source, size=10, color=mapper, alpha=1)

    # Add the interpolated line (after scatter to ensure it's on top)
    p.line(times_interpolated, cs(times_interpolated), line_color='orange', line_width=2)

    # Add hover tool with file name
    hover = HoverTool()
    hover.tooltips = [
        ("Correlation Length", "@{Correlation Length}"),
        ("Time", "@{time}"),
        ("File Name", "@{file_name}")  # Add the file name to the hover tooltip
    ]

    p.add_tools(hover)

    # Color bar
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    p.add_layout(color_bar, 'right')

    output_notebook()
    show(p)
