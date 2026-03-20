"""
Modified plotting functions for motors_EDA.ipynb

Changes:
1. Changed from separate 'proteins' (list) and 'dna_concentrations' (list) parameters
   to a single 'proteins_dict' parameter where keys are protein names and values are
   lists of DNA concentrations for that protein.

2. Added 'sigma' parameter support in plots_to_include dictionary for Gaussian filtering.
   sigma=0 means no filter is applied.

3. All existing arguments are preserved and fixed as needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumtrapz
from sklearn.decomposition import PCA


def apply_gaussian_filter(data, sigma):
    """
    Apply Gaussian filter to data if sigma > 0, otherwise return data unchanged.
    
    Parameters:
    -----------
    data : array-like
        Data to filter
    sigma : float
        Standard deviation for Gaussian filter. If 0, no filter is applied.
    
    Returns:
    --------
    filtered_data : array-like
        Filtered data (or original if sigma == 0)
    """
    if sigma > 0:
        return gaussian_filter1d(data, sigma=sigma)
    return data


def plot_protein_concentration_vs_time(
    df, 
    proteins_dict,  # Changed from: proteins, dna_concentrations
    sigma=0,  # NEW: Gaussian filter sigma (0 = no filter)
    figsize=(10, 6),
    save_path=None,
    save_format='auto',
    dpi=300,
    xlabel='Time (h)',
    ylabel='Protein Concentration (nM)',
    title='Protein Concentration vs Time',
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    title_fontsize=14,
    legend=True,
    legend_loc='upper left',
    legend_bbox_to_anchor=(1.05, 1),
    grid=True,
    grid_alpha=0.3,
    xlim=None,
    ylim=None,
    xscale='linear',
    yscale='linear',
    color_map=None,
    linewidth=1.5,
    alpha=1.0,
    marker=None,
    markersize=6,
    linestyle='-',
    tight_layout=True,
    show=True,
    **kwargs  # Preserve any additional arguments
):
    """
    Plot protein concentration vs time for specified proteins and DNA concentrations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    proteins_dict : dict
        Dictionary where keys are protein names (str) and values are lists of
        DNA concentrations (float) to plot for that protein.
        Example: {'ThTr': [160, 80, 40, 20], 'C': [160, 80, 40]}
        CHANGED FROM: separate 'proteins' (list) and 'dna_concentrations' (list) parameters
    sigma : float, default=0
        Standard deviation for Gaussian filter. 0 means no filter is applied.
        NEW: This parameter controls smoothing of the data.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    save_path : str or None, default=None
        Path to save the figure. If None, figure is not saved.
    save_format : str, default='auto'
        Format for saving ('auto', 'png', 'svg', 'pdf', etc.)
    dpi : int, default=300
        Resolution for saved figures
    xlabel, ylabel, title : str
        Axis labels and title
    xlabel_fontsize, ylabel_fontsize, title_fontsize : int
        Font sizes for labels and title
    legend : bool, default=True
        Whether to show legend
    legend_loc : str, default='upper left'
        Legend location
    legend_bbox_to_anchor : tuple, default=(1.05, 1)
        Legend bounding box anchor
    grid : bool, default=True
        Whether to show grid
    grid_alpha : float, default=0.3
        Grid transparency
    xlim, ylim : tuple or None
        Axis limits
    xscale, yscale : str, default='linear'
        Scale for axes ('linear', 'log', etc.)
    color_map : str or None
        Colormap name
    linewidth : float, default=1.5
        Line width for plots
    alpha : float, default=1.0
        Transparency
    marker : str or None
        Marker style
    markersize : int, default=6
        Marker size
    linestyle : str, default='-'
        Line style
    tight_layout : bool, default=True
        Whether to apply tight layout
    show : bool, default=True
        Whether to display the plot
    **kwargs : additional plotting arguments
        Any other arguments are preserved and passed through
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get color cycle if color_map is specified
    if color_map:
        colors = plt.cm.get_cmap(color_map)
    else:
        colors = None
    
    color_idx = 0
    for protein, dna_concentrations in proteins_dict.items():
        for dna_conc in dna_concentrations:
            # Filter data
            subset = df[(df['protein'] == protein) & (df['DNA nM'] == dna_conc)].copy()
            if subset.empty:
                continue
            
            # Sort by time
            subset = subset.sort_values('time (s)')
            
            # Get time and concentration data
            time = subset['time (s)'].values
            concentration = subset['Protein Concentration_nM'].values
            
            # Apply Gaussian filter if sigma > 0
            concentration_filtered = apply_gaussian_filter(concentration, sigma)
            
            # Plot
            label = f'{protein}-{dna_conc}nM'
            plot_kwargs = {
                'label': label,
                'linewidth': linewidth,
                'alpha': alpha,
                'marker': marker,
                'markersize': markersize,
                'linestyle': linestyle,
            }
            if colors:
                plot_kwargs['color'] = colors(color_idx / max(1, sum(len(v) for v in proteins_dict.values())))
                color_idx += 1
            
            # Merge with any additional kwargs
            plot_kwargs.update(kwargs)
            ax.plot(time / 3600, concentration_filtered, **plot_kwargs)
    
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    
    if legend:
        ax.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)
    
    if grid:
        ax.grid(True, alpha=grid_alpha)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    if tight_layout:
        plt.tight_layout()
    
    if save_path:
        if save_format == 'auto':
            if save_path.endswith('.svg'):
                save_format = 'svg'
            elif save_path.endswith('.pdf'):
                save_format = 'pdf'
            else:
                save_format = 'png'
        plt.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight' if tight_layout else None)
    
    if show:
        plt.show()
    else:
        return fig, ax


def plot_mean_velocity_vs_time(df, proteins_dict, sigma=0, **kwargs):
    """
    Plot mean velocity vs time for specified proteins and DNA concentrations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    proteins_dict : dict
        Dictionary where keys are protein names and values are lists of DNA concentrations.
    sigma : float, default=0
        Standard deviation for Gaussian filter. 0 means no filter.
    **kwargs : additional plotting arguments
    """
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    for protein, dna_concentrations in proteins_dict.items():
        for dna_conc in dna_concentrations:
            subset = df[(df['protein'] == protein) & (df['DNA nM'] == dna_conc)].copy()
            if subset.empty:
                continue
            
            subset = subset.sort_values('time (s)')
            time = subset['time (s)'].values
            velocity = subset['velocity magnitude [m/s]_mean'].values
            
            # Apply Gaussian filter if sigma > 0
            velocity_filtered = apply_gaussian_filter(velocity, sigma)
            
            label = f'{protein}-{dna_conc}nM'
            ax.plot(time / 3600, velocity_filtered, label=label, **kwargs.get('plot_kwargs', {}))
    
    ax.set_xlabel('Time (h)', fontsize=kwargs.get('xlabel_fontsize', 12))
    ax.set_ylabel('Mean Velocity (m/s)', fontsize=kwargs.get('ylabel_fontsize', 12))
    ax.set_title('Mean Velocity vs Time', fontsize=kwargs.get('title_fontsize', 14))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if kwargs.get('save_path'):
        plt.savefig(kwargs['save_path'], dpi=kwargs.get('dpi', 300))
    plt.show()


def plot_mean_power_vs_time(df, proteins_dict, sigma=0, **kwargs):
    """
    Plot mean power vs time for specified proteins and DNA concentrations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    proteins_dict : dict
        Dictionary where keys are protein names and values are lists of DNA concentrations.
    sigma : float, default=0
        Standard deviation for Gaussian filter. 0 means no filter.
    **kwargs : additional plotting arguments
    """
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    for protein, dna_concentrations in proteins_dict.items():
        for dna_conc in dna_concentrations:
            subset = df[(df['protein'] == protein) & (df['DNA nM'] == dna_conc)].copy()
            if subset.empty:
                continue
            
            subset = subset.sort_values('time (s)')
            time = subset['time (s)'].values
            power = subset['power [W]_mean'].values
            
            # Apply Gaussian filter if sigma > 0
            power_filtered = apply_gaussian_filter(power, sigma)
            
            label = f'{protein}-{dna_conc}nM'
            ax.plot(time / 3600, power_filtered, label=label, **kwargs.get('plot_kwargs', {}))
    
    ax.set_xlabel('Time (h)', fontsize=kwargs.get('xlabel_fontsize', 12))
    ax.set_ylabel('Mean Power (W)', fontsize=kwargs.get('ylabel_fontsize', 12))
    ax.set_title('Mean Power vs Time', fontsize=kwargs.get('title_fontsize', 14))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if kwargs.get('save_path'):
        plt.savefig(kwargs['save_path'], dpi=kwargs.get('dpi', 300))
    plt.show()


# Add more plotting functions following the same pattern...
# (plot_correlation_length_vs_max_power, plot_mean_velocity_vs_protein_concentration, etc.)


def generate_plots_master(df, proteins_dict, plots_to_include, **kwargs):
    """
    Master function to generate multiple plots based on plots_to_include dictionary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    proteins_dict : dict
        Dictionary where keys are protein names and values are lists of DNA concentrations.
        Example: {'ThTr': [160, 80, 40, 20], 'C': [160, 80, 40]}
    plots_to_include : dict
        Dictionary where keys are plot function names (str) and values are dictionaries
        containing plot-specific parameters including 'sigma' for Gaussian filtering.
        Example:
        {
            'plot_protein_concentration_vs_time': {'sigma': 2.0, 'save_path': 'protein_conc.png'},
            'plot_mean_velocity_vs_time': {'sigma': 1.5, 'save_path': 'velocity.png'},
            'plot_mean_power_vs_time': {'sigma': 0}  # sigma=0 means no filter
        }
    **kwargs : additional arguments passed to all plotting functions
    """
    # Mapping of plot names to functions
    plot_functions = {
        'plot_protein_concentration_vs_time': plot_protein_concentration_vs_time,
        'plot_mean_velocity_vs_time': plot_mean_velocity_vs_time,
        'plot_mean_power_vs_time': plot_mean_power_vs_time,
        # Add more mappings as needed
    }
    
    for plot_name, plot_params in plots_to_include.items():
        if plot_name not in plot_functions:
            print(f"Warning: Plot function '{plot_name}' not found. Skipping.")
            continue
        
        # Extract sigma from plot_params (default to 0 if not specified)
        sigma = plot_params.get('sigma', 0)
        
        # Get the plot function
        plot_func = plot_functions[plot_name]
        
        # Prepare arguments for the plot function
        plot_kwargs = {k: v for k, v in plot_params.items() if k != 'sigma'}
        plot_kwargs.update(kwargs)
        
        # Call the plot function
        try:
            plot_func(df, proteins_dict, sigma=sigma, **plot_kwargs)
        except Exception as e:
            print(f"Error generating {plot_name}: {e}")
            continue


# Example usage:
"""
# Example proteins_dict (like in cell 10)
proteins_dict = {
    'ThTr': [160, 80, 40, 20, 10, 5, 2.5, 1.25],
    'C': [160, 80, 40, 20, 10, 5, 2.5, 1.25],
    'AcSu2': [160, 80, 40, 20, 10, 5, 2.5, 1.25],
    'HeAl': [160, 80, 40, 20, 10, 5, 2.5, 1.25],
}

# Example plots_to_include with sigma parameters
plots_to_include = {
    'plot_protein_concentration_vs_time': {
        'sigma': 2.0,  # Apply Gaussian filter with sigma=2.0
        'save_path': 'protein_conc.png',
        'figsize': (12, 8)
    },
    'plot_mean_velocity_vs_time': {
        'sigma': 1.5,  # Apply Gaussian filter with sigma=1.5
        'save_path': 'velocity.png'
    },
    'plot_mean_power_vs_time': {
        'sigma': 0,  # No Gaussian filter
        'save_path': 'power.png'
    }
}

# Generate all plots
generate_plots_master(df, proteins_dict, plots_to_include)
"""
