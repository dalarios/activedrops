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
    This file compiles all of the relevant functions for plotting style
    related to the ActiveDROPS project.
"""

# Default RP plotting style
def set_plotting_style():
    """
    Formats plotting enviroment to that used in Physical Biology of the Cell,
    2nd edition. To format all plots within a script, simply execute
    `mwc_induction_utils.set_plotting_style() in the preamble.
    """
    rc = {'lines.linewidth': 1.25,
          'axes.labelsize': 8,
          'axes.titlesize': 9,
          'axes.facecolor': '#E3DCD0',
          'xtick.labelsize': 7,
          'ytick.labelsize': 7,
        #   'font.family': 'Lucida Sans Unicode',
          'grid.linestyle': '-',
          'grid.linewidth': 0.5,
          'grid.color': '#ffffff',
          'legend.fontsize': 9}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('xtick.major', pad=-1)
    plt.rc('ytick.major', pad=-1)
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('figure', figsize=[3.5, 2.5])
    plt.rc('svg', fonttype='none')
    plt.rc('legend', title_fontsize='8', frameon=True, 
           facecolor='#E3DCD0', framealpha=1)
    sns.set_style('darkgrid', rc=rc)
    sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)


def velocity(dfs, labels, filename=None, max_frame=None, x_limits=None, y_limits=None, logy=False, figsize=(10, 6)):
    """
    dfs : list of pd.DataFrame
        A list of DataFrames each containing at least the columns 'time_min' and 'v (m/s)'.
    labels : list of str
        A list of string labels for the plots. Should be the same length as dfs.
    filename : str, optional
        The name of the file where the plot will be saved. If None, the plot will be shown using plt.show().
    max_frame : int, optional
        The maximum frame (row number) to plot.
    x_limits : tuple of (float, float), optional
        The limits for the x-axis.
    y_limits : tuple of (float, float), optional
        The limits for the y-axis.
    """
    if len(dfs) != len(labels):
        raise ValueError("Number of dataframes and labels must be the same")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for df, label in zip(dfs, labels):
        df = df.iloc[:max_frame,:]
        
        # Choose a color
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        ax.plot(df['time_min'], df['v (m/s)'] * 1e6, label=label, color=color)
        ax.fill_between(df['time_min'], df['v (m/s)'] * 1e6, color=color, alpha=0.2)
    
    ax.set_xlabel('time (min)')
    ax.set_ylabel('Velocity (µm/s)')
    ax.legend()
    
    if logy == True:
        ax.set_yscale('log')
        
    # Set axis limits if provided
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)
    
    if filename:
        plt.savefig(filename, format='jpg', dpi=150)
    else:
        plt.show()
    
    plt.close(fig)

    


def distance(dfs, labels, filename=None, max_frame=None, x_limits=None, y_limits=None, figsize=(10, 6), logy=False):
    if len(dfs) != len(labels):
        raise ValueError("Number of dataframes and labels must be the same")

    fig, ax = plt.subplots(figsize=figsize)

    for df, label in zip(dfs, labels):
        df = df.iloc[:max_frame,:]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        ax.plot(df['time_min'], df['distance (m)'] * 1e6, '-', color=color)  # Plotting the line
        ax.plot(df['time_min'].iloc[-1], df['distance (m)'].iloc[-1] * 1e6, 'o', label=label, markersize=16, color=color)  # Plotting the glyph

    ax.set_xlabel('time (min)')
    ax.set_ylabel('Distance (µm)')
    ax.legend()

    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)

    if logy == True:
        ax.set_yscale('log')
        
    if filename:
        plt.savefig(filename, format='jpg', dpi=150)
    else:
        plt.show()

    plt.close(fig)

    
    
def power(dfs, labels, filename=None, max_frame=None, x_limits=None, y_limits=None, logy=True, figsize=(10, 6)):
    if len(dfs) != len(labels):
        raise ValueError("Number of dataframes and labels must be the same")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for df, label in zip(dfs, labels):
        df = df.iloc[:max_frame,:]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        ax.plot(df['time_min'], df['Power (W)'], label=label, color=color)
        ax.fill_between(df['time_min'], df['Power (W)'], alpha=0.2, color=color)
    
    ax.set_xlabel('time (min)')
    ax.set_ylabel('Power (W)')
    
    if logy == True:
        ax.set_yscale('log')
    
    ax.legend()
    
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)
    
    if filename:
        plt.savefig(filename, format='jpg', dpi=150)
    else:
        plt.show()
    
    plt.close(fig)

    
    
    
def vorticity(dfs, labels, filename=None, max_frame=None, x_limits=None, y_limits=None, logy=False, figsize=(10, 6)):
    if len(dfs) != len(labels):
        raise ValueError("Number of dataframes and labels must be the same")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for df, label in zip(dfs, labels):
        df = df.iloc[:max_frame,:]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        ax.plot(df['time_min'], df['vorticity (1/s)'], label=label, color=color)
        ax.fill_between(df['time_min'], df['vorticity (1/s)'], alpha=0.2, color=color)
    
    ax.set_xlabel('time (min)')
    ax.set_ylabel('Vorticity (1/s)')
    ax.legend()
    
    if logy == True:
        ax.set_yscale('log')
    
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)
    
    if filename:
        plt.savefig(filename, format='jpg', dpi=150)
    else:
        plt.show()
    
    plt.close(fig)

    
    
def divergence(dfs, labels, filename=None, max_frame=None, x_limits=None, y_limits=None, logy=False, figsize=(10, 6)):
    if len(dfs) != len(labels):
        raise ValueError("Number of dataframes and labels must be the same")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for df, label in zip(dfs, labels):
        df = df.iloc[:max_frame,:]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        ax.plot(df['time_min'], df['divergence (1/s)'], label=label, color=color)
        ax.fill_between(df['time_min'], df['divergence (1/s)'], alpha=0.2, color=color)
    
    ax.set_xlabel('time (min)')
    ax.set_ylabel('Divergence (1/s)')
    ax.legend()
    
    if logy == True:
        ax.set_yscale('log')
  
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)
    
    if filename:
        plt.savefig(filename, format='jpg', dpi=150)
    else:
        plt.show()
    
    plt.close(fig)



import os

def generate_image_sequence(dfs, plot_type, dir_name, labels, x_limits=None, y_limits=None, **kwargs):
    """
    Generates an image sequence of plots from a list of DataFrames, saving each image to a specified directory.
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        The list of DataFrames containing the data to be plotted.
    plot_type : str
        The type of plot to generate, e.g., 'velocity', 'distance', 'power'.
    dir_name : str
        The name of the directory where the plots will be saved.
    labels : list of str
        The labels for the plots.
    x_limits : tuple of (float, float), optional
        The limits for the x-axis.
    y_limits : tuple of (float, float), optional
        The limits for the y-axis.
    kwargs : dict, optional
        Additional keyword arguments to pass to the inner plotting function.
    """
    
    if len(dfs) != len(labels):
        raise ValueError("The number of dataframes and labels must be the same.")
    
    # Check if directory exists, if not, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Define a mapping of plot types to functions
    plot_functions = {
        'velocity': velocity,
        'distance': distance,
        'power': power,
        'vorticity': vorticity,
        'divergence': divergence
    }
    
    # Get the appropriate plot function
    plot_func = plot_functions.get(plot_type)
    if plot_func is None:
        raise ValueError(f"Invalid plot type: {plot_type}. Expected one of: {', '.join(plot_functions.keys())}.")
    
    # Create a subdirectory for the specific plot type
    plot_dir = os.path.join(dir_name, plot_type)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Find the length of the largest DataFrame
    max_length = max(len(df) for df in dfs)
    
    # Loop through each frame and save a plot for all dataframes
    for i in range(1, max_length + 1):
        # Create a list of dataframes each truncated to the current frame
        truncated_dfs = [df.iloc[:i,:] for df in dfs]
        plot_func(
            truncated_dfs,
            labels,
            x_limits=x_limits, y_limits=y_limits,
            filename=f"{plot_dir}/{plot_type}_plot_{i}.jpg",
            **kwargs  # Forward the additional keyword arguments to the inner plotting function
        )





import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm

def plot_feature_heatmaps(feature_keywords, input_dir, output_dir, display_single_frame=False):
    """
    Plots combined heatmaps for the given features across all frames and saves them.
    
    Args:
    - feature_keywords (list of str): Keywords representing the desired features (e.g., ["velocity", "vorticity"]).
    - input_dir (str): Directory where the .txt files are located.
    - output_dir (str): Directory where the output plots should be saved.
    - display_single_frame (bool): If True, displays the heatmap of the first frame in the notebook itself.
    """
    
    # Mapping from user-friendly keywords to actual column names in the .txt files
    feature_mapping = {
        'velocity': 'magnitude [μm/s]',
        'vorticity': 'vorticity [1/s]',
        'divergence': 'divergence [1/s]',
        'shear': 'simple shear [1/s]',
        'strain': 'simple strain [1/s]'
    }
    
    # Mapping from feature to colormap
    colormap_mapping = {
        'velocity': 'viridis',
        'vorticity': 'bwr',
        'divergence': 'Set2',
        'shear': 'PiYG',
        'strain': 'coolwarm'
    }
    
    # Create combined feature name for folder and filenames
    combined_name = "_".join(feature_keywords) + ("_heatmap" if len(feature_keywords) > 1 else "")
    
    # Pattern for input .txt files
    input_pattern = os.path.join(input_dir, 'PIVlab_*.txt')
    
    # Create a dictionary to store global min and max values for each feature
    global_feature_range = {keyword: [float('inf'), float('-inf')] for keyword in feature_keywords}

    for filepath in glob.glob(input_pattern):
        df = pd.read_csv(filepath, sep=',', skiprows=2)
        
        # Add a new column for velocity in micrometers per second (μm/s)
        if 'magnitude [m/s]' in df.columns and 'magnitude [μm/s]' not in df.columns:
            df['magnitude [μm/s]'] = df['magnitude [m/s]'] * 1e6
        
        for keyword in feature_keywords:
            feature_column = feature_mapping[keyword]
            local_min = df[feature_column].min()
            local_max = df[feature_column].max()
            global_feature_range[keyword][0] = min(global_feature_range[keyword][0], local_min)
            global_feature_range[keyword][1] = max(global_feature_range[keyword][1], local_max)
    
    # Construct a new sub-directory path
    sub_dir = os.path.join(output_dir, combined_name)

    # Ensure the sub-directory exists
    if not os.path.exists(sub_dir) and not display_single_frame:
        os.makedirs(sub_dir)

    # Use these values to fix the range of the colormap when plotting each heatmap
    filepaths = sorted(glob.glob(input_pattern))
    
    for filepath in filepaths:
        frame_num = os.path.basename(filepath).split('_')[-1].split('.')[0]
        df = pd.read_csv(filepath, sep=',', skiprows=2)
        df['x [mm]'] = df['x [m]'] * 1000
        df['y [mm]'] = df['y [m]'] * 1000
        
        # Ensure the velocity column in micrometers per second is present
        if 'magnitude [m/s]' in df.columns and 'magnitude [μm/s]' not in df.columns:
            df['magnitude [μm/s]'] = df['magnitude [m/s]'] * 1e6
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for keyword in feature_keywords:
            feature_column = feature_mapping[keyword]
            x = df['x [mm]'].values
            y = df['y [mm]'].values
            feature_values = df[feature_column].values

            xi = np.linspace(x.min(), x.max(), len(np.unique(x)))
            yi = np.linspace(y.min(), y.max(), len(np.unique(y)))
            zi = griddata((x, y), feature_values, (xi[None, :], yi[:, None]), method='linear')

            cmap = colormap_mapping[keyword]
            alpha = 0.5 if keyword != 'velocity' else 1.0  # Adjust alpha for overlapping

            c = ax.pcolormesh(xi, yi, zi, shading='auto', cmap=cmap, alpha=alpha, 
                              vmin=global_feature_range[keyword][0], 
                              vmax=global_feature_range[keyword][1])
            fig.colorbar(c, ax=ax, label=feature_column)

        ax.set_title(f"{combined_name.replace('_', ' ').title()} - Frame {frame_num}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        # If display_single_frame is True, show the plot for the first frame and return
        if display_single_frame:
            plt.show()
            return "Displayed the heatmap for the first frame in the notebook!"
        
        filename = f'{combined_name}_{frame_num}.png'
        fig.savefig(os.path.join(sub_dir, filename))
        plt.close(fig)

    return f"Global min and max of features determined, and all combined plots saved in {sub_dir}!"

