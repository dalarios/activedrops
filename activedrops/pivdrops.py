# Import plotting utilities
import os
import glob
import sys
import cv2
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from ipywidgets import interact, FloatSlider
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import colorcet as cc
from PIL import Image, ImageEnhance, ImageOps  # Added ImageOps here
from natsort import natsorted  # Import for natural sorting

"""
Title:
    pivdrops.py
Last update:
    2023-11-09
Author(s):
    David Larios
Purpose:
    This file compiles all of the relevant functions for processing raw
    PIV data from Matlab PIVlab for the ActiveDROPS project.
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
          'grid.linewidth': 0.1,
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


def df_piv(file, volume):
    """
    Processes a PIV data file and computes various parameters.

    Args:
    - file (str): Path to the PIV data file.
    - volume (float): The volume in microliters for power calculation.

    Returns:
    - DataFrame: Processed DataFrame with additional computed columns.
    """

    # Read and preprocess the dataframe
    df = pd.read_csv(file, skiprows=2).fillna(0)
    # Convert measurements to micrometers and micrometers per second
    df['x [um]'] = df['x [m]'] * 1E6
    df['y [um]'] = df['y [m]'] * 1E6
    df['u [um/s]'] = df['u [m/s]'] * 1E6
    df['v [um/s]'] = df['v [m/s]'] * 1E6
    df['magnitude [um/s]'] = df['magnitude [m/s]'] * 1E6

    # Obtain square grid of velocity magnitudes
    v = df.pivot(index='y [um]', columns='x [um]', values="magnitude [um/s]").values

    # Calculate the autocorrelation function with Fourier transform
    full_product = np.fft.fft2(v) * np.conj(np.fft.fft2(v))
    inverse = np.real(np.fft.ifft2(full_product))  # Real part of the inverse Fourier transform
    normalized_inverse = inverse / inverse[0, 0]   # Normalize the autocorrelation function

    # Define the number of r values and initialize an array for the results
    r_values = v.shape[0] // 2
    results = np.zeros(r_values)

    # Compute the autocorrelation for each r value
    for r in range(r_values):
        autocorrelation_value = (inverse[r, r] + inverse[-r, -r]) / (v.shape[0] * v.shape[1])
        results[r] = autocorrelation_value

    # Normalize the results array
    results = results / results[0]

    # Fit the results to an exponential decay model
    def exponential_decay(tau, A, B, C):
        return A * np.exp(-tau / B) + C

    params, _ = curve_fit(exponential_decay, np.arange(len(results)), results, maxfev=5000)
    A, B, C = params
    fitted_values = exponential_decay(np.arange(r_values), A, B, C)

    # Compute correlation length and other parameters
    intervector_distance_microns = (df["y [um]"].max() - df["y [um]"].min()) / v.shape[0]
    lambda_tau = -B * np.log((0.3 - C) / A) * intervector_distance_microns
    df["correlation length (µm)"] = lambda_tau

    # Calculate power
    v0 = volume * 1E-9 # µl --> m^3
    µ = 1E-3        # mPa*S
    correlation_length = lambda_tau * 1E-6 # µm --> m
    df["Power (W)"] = v0 * µ * (df["magnitude [m/s]"].mean() / correlation_length)**2
    
    
    # Calculate mean of top 30% velocity magnitudes
    n = int(0.3 * len(df))  # Top 30% of the vectors
    df["mean velocity [um/s]"] = df["magnitude [um/s]"].nlargest(n).mean()

    # # Calculate drag force
    # df["drag force (pN)"] = 6 * np.pi * µ * lambda_tau * df["magnitude [m/s]"].mean()

    # Add file name column
    df["file name"] = os.path.basename(file).split('.')[0]

    # Reorganize DataFrame
    df = pd.concat([df.iloc[:, 12:], df.iloc[:, 4:12]], axis=1)

    return df



def process_piv_files(path, volume=2, max_frame=None):
    """
    Processes multiple PIV files from a given directory.

    Args:
    - path (str): Path pattern to the PIV files.

    Returns:
    - List[DataFrame]: List of processed DataFrames.
    """
    files = sorted(glob.glob(path))
    dataframes = [df_piv(file, volume) for file in files[:max_frame]]
    return dataframes





def overlay_heatmap_on_image(image_file, df, heatmap_data, feature, vmin, vmax, time_in_minutes, output_dir=None):
    """
    Overlays a heatmap on an image and saves or renders the combined visualization.

    Args:
    - image_file (str): Path to the image file.
    - heatmap_data (np.array): Data for the heatmap.
    - feature (str): Name of the feature for the heatmap.
    - vmin (float): Minimum value for colormap scaling.
    - vmax (float): Maximum value for colormap scaling.
    - time_in_minutes (float): Time in minutes for the current frame.
    - output_dir (str, optional): Directory to save the plot. If None, the plot is shown.
    """
    # Load the image
    image = Image.open(image_file)

    # Invert colors for specific features
    if feature not in ['magnitude [um/s]', 'dcev [1]']:
        image = ImageOps.invert(image)

    # Choose colormap based on feature
    cmap = 'inferno' if feature in ['magnitude [um/s]', 'dcev [1]'] else 'RdGy'  # Change 'coolwarm' to your preferred diverging colormap


    # Create a plot to overlay the heatmap on the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap=None, extent=[-2762/2, 2762/2, -2762/2, 2762/2])  # Display the image
    im = plt.imshow(heatmap_data, cmap=cmap, origin='lower', alpha=0.7, extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax)  # Overlay the heatmap
    plt.xlabel('x [um]')
    plt.ylabel('y [um]')
    cbar = plt.colorbar(im)
    cbar.set_label(feature)
    plt.title(f'PIV Heatmap - {df["file name"][0]} || Time: {time_in_minutes:.2f} min')

    # Save or show the plot
    if output_dir:
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        plt.savefig(output_dir, format='jpg', dpi=250)
        plt.close()
    else:
        plt.show()



def piv_heatmap(df, feature, vmin, vmax, time_in_minutes, output_dir=None, image_file=None):
    """
    Generates and saves/renders a heatmap of a specified feature from the PIV data.
    Optionally overlays the heatmap on an image.
    """
    vals = df.pivot(index='y [um]', columns='x [um]', values=feature).values

    cmap = 'inferno' if feature in ['magnitude [um/s]', 'dcev [1]'] else 'coolwarm'  # Change 'coolwarm' to your preferred diverging colormap

    if image_file:
        # If an image file is provided, overlay the heatmap on the image
        overlay_heatmap_on_image(image_file, df, vals, feature, vmin, vmax, time_in_minutes, output_dir)
    elif image_file is None:
        # Regular heatmap generation (as in your original code)
        plt.figure(figsize=(10, 6))
        im = plt.imshow(vals, cmap='viridis', origin='lower', extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax)
        plt.xlabel('x [um]')
        plt.ylabel('y [um]')
        cbar = plt.colorbar(im)
        cbar.set_label(feature)
        plt.title(f'PIV Heatmap - {df["file name"][0]} || Time: {time_in_minutes:.2f} min')

        if output_dir:
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            plt.savefig(output_dir, format='jpg', dpi=250)
            plt.close()
        else:
            plt.show()




def generate_heatmaps(dataframes, feature, output_dir_base=None, vmin=None, vmax=None, seconds_per_frame=3, image_path=None):
    """
    Generates heatmaps for a list of DataFrames and a specified feature.
    Optionally overlays each heatmap on an image from a specified path.
    
    Args:
    - dataframes (List[DataFrame]): List of DataFrame objects to plot.
    - feature (str): The feature to plot.
    - output_dir_base (str, optional): Base directory to save heatmaps. If None, heatmaps are displayed.
    - vmin (float, optional): Minimum value for colormap scaling. If None, computed from data.
    - vmax (float, optional): Maximum value for colormap scaling. If None, computed from data.
    - seconds_per_frame (int, optional): Time interval in seconds for each frame. Default is 3 seconds.
    - image_path (str, optional): Path to the folder containing images for overlay. If None, regular heatmaps are generated.
    """
    # Removing units and special characters from the feature name for folder creation
    feature_name_for_folder = ''.join(filter(str.isalnum, feature.split('[')[0])).strip()

    # # Calculate vmin and vmax if not provided
    # if vmin is None or vmax is None:
    #     all_values = pd.concat([df[feature] for df in dataframes])
    #     vmin = vmin if vmin is not None else all_values.min()
    #     vmax = vmax if vmax is not None else all_values.max()

    # Get list of image files if an image path is provided
    image_files = sorted_alphanumeric(glob.glob(os.path.join(image_path, '*.tif'))) if image_path else [None] * len(dataframes)

    for i, (df, image_file) in enumerate(zip(dataframes, image_files)):
        time_in_minutes = i * seconds_per_frame / 60  # Calculate time in minutes

        output_dir = None
        if output_dir_base:
            # Creating a specific folder for the feature
            feature_folder = os.path.join(output_dir_base, feature_name_for_folder)
            os.makedirs(feature_folder, exist_ok=True)
            output_dir = os.path.join(feature_folder, f"heatmap_{i}.jpg")

        piv_heatmap(df, feature, vmin=vmin, vmax=vmax, time_in_minutes=time_in_minutes, output_dir=output_dir, image_file=image_file)





def piv_time_series(dataframes, time_interval_seconds=3):
    """
    Constructs a time series DataFrame from a list of PIV dataframes.

    This function takes a list of dataframes, each corresponding to a different time point in a PIV experiment,
    and constructs a time series dataframe. Each row in the resultant dataframe corresponds to a different time point.

    Args:
    - dataframes (list of DataFrame): A list of dataframes, each representing PIV data at a different time point.
    - time_interval_min (float, optional): Time interval between each frame in minutes. Defaults to 0.05.

    Returns:
    - DataFrame: A dataframe with time series data including file name, power, and mean velocity.
    """

    # Initialize an empty DataFrame for the time series data
    df_ts = pd.DataFrame(columns=['file name', 'Power (W)', 'mean velocity [um/s]'])

    # Loop through each dataframe in the list
    for df in dataframes:
        # Extract relevant data from the first row of each dataframe
        file_name = df.loc[0, 'file name']
        power = df.loc[0, 'Power (W)']
        velocity = df.loc[0, 'mean velocity [um/s]']

        # Create a new row with the extracted data and add it to the time series dataframe
        new_row = pd.DataFrame({'file name': [file_name], 'Power (W)': [power], 'mean velocity [um/s]': [velocity]})
        df_ts = pd.concat([df_ts, new_row], ignore_index=True)

    # Reset index to use it as the time column
    df_ts = df_ts.reset_index(drop=False)
    # Rename 'index' column to 'time (min)' and adjust it based on the time interval
    df_ts = df_ts.rename(columns={'index': 'time (min)'})
    df_ts['time (min)'] = df_ts['time (min)'] * (time_interval_seconds/60)

    df_ts['distance (um)'] = (df_ts['mean velocity [um/s]'] * df_ts['time (min)'].diff()).cumsum()
    df_ts['Work (J)'] = (df_ts['Power (W)'] * df_ts['time (min)'].diff()).cumsum()

    return df_ts


def plot_time_series(df_ts, feature, sigma=0, output_dir=None):
    """
    Plots a specified feature from the time series DataFrame and saves the plot.

    Args:
    - df_ts (DataFrame): Time series DataFrame.
    - feature (str): The feature to plot ('velocity', 'power', 'distance', or 'work').
    - sigma (float, optional): Standard deviation for Gaussian filter. Defaults to 0 (no filtering).
    - output_dir (str, optional): Directory to save the plot. If None, the plot is shown.
    """

    # Apply Gaussian filter to smooth the data if sigma is specified
    if feature == 'velocity':
        y = gaussian_filter(df_ts['mean velocity [um/s]'], sigma=sigma)
        ylabel = 'Velocity (µm/s)'
    elif feature == 'power':
        y = gaussian_filter(df_ts['Power (W)'], sigma=sigma)
        ylabel = 'Power (W)'
    elif feature == 'distance':
        y = gaussian_filter(df_ts['distance (um)'], sigma=sigma)
        ylabel = 'Distance (µm)'
    elif feature == 'work':
        y = gaussian_filter(df_ts['Work (J)'], sigma=sigma)
        ylabel = 'Work (J)'
    else:
        raise ValueError("Feature not recognized. Choose from 'velocity', 'power', 'distance', or 'work'.")

    # Plotting the time series
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_ts['time (min)'], y)
    ax.fill_between(df_ts['time (min)'], y, alpha=0.2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(ylabel)

    # Save or show the plot
    if output_dir:
        # Constructing file path
        file_path = os.path.join(output_dir, f"{feature}_time_series.jpg")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format='jpg', dpi=250)
        plt.close()
    else:
        plt.show()



def plot_combined_time_series(combined_df, feature, sigma=0, output_dir=None):
    """
    Plots a specified feature from the combined time series DataFrame for all samples and saves the plot.

    Args:
    - combined_df (DataFrame): Combined time series DataFrame with multiple samples.
    - feature (str): The feature to plot ('velocity', 'power', 'distance', or 'work').
    - sigma (float, optional): Standard deviation for Gaussian filter. Defaults to 0 (no filtering).
    - output_dir (str, optional): Directory to save the plot. If None, the plot is shown.
    """
    # Mapping feature names to dataframe column names
    feature_map = {
        'velocity': 'mean velocity [um/s]',
        'power': 'Power (W)',
        'distance': 'distance (um)',
        'work': 'Work (J)'
    }

    if feature not in feature_map:
        raise ValueError("Feature not recognized. Choose from 'velocity', 'power', 'distance', or 'work'.")

    df_column = feature_map[feature]

    # Apply Gaussian filter to smooth the data if sigma is specified
    combined_df['smoothed'] = combined_df.groupby('Condition')[df_column].transform(lambda x: gaussian_filter(x, sigma=sigma) if sigma > 0 else x)

    # Plotting the time series for each condition
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in combined_df['Condition'].unique():
        subset = combined_df[combined_df['Condition'] == condition]
        ax.plot(subset['time (min)'], subset['smoothed'], label=condition)
        ax.fill_between(subset['time (min)'], subset['smoothed'], alpha=0.2)

    ax.set_xlabel('Time (min)')
    ax.set_ylabel(df_column)
    ax.legend()

    # Save or show the plot
    if output_dir:
        file_path = os.path.join(output_dir, f"combined_{feature}_time_series.jpg")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format='jpg', dpi=250)
        plt.close()
    else:
        plt.show()


#####################

def sorted_alphanumeric(data):
    """
    Helper function to sort data in human-readable alphanumeric order.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def make_movies_from_features(base_directory, fps):
    # Find all subdirectories in the base directory
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    for feature in subdirectories:
        feature_directory = os.path.join(base_directory, feature)
        output_filename = os.path.join(base_directory, f"{feature}.avi")

        # Get all the .jpg files from the feature directory
        images = [img for img in os.listdir(feature_directory) if img.endswith(".jpg")]

        # Skip if no images are found
        if not images:
            print(f"No images found in {feature_directory}, skipping movie creation.")
            continue

        # Sort the images in alphanumeric order
        images = sorted_alphanumeric(images)

        # Read the first image to get the width and height
        frame = cv2.imread(os.path.join(feature_directory, images[0]))
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # Loop through all the images and add them to the video
        for image in images:
            frame = cv2.imread(os.path.join(feature_directory, image))
            out.write(frame)

        # Release everything when the job is finished
        out.release()

def save_dataframes(dataframes, output_dir, prefix='df'):
    """Saves a list of DataFrames to the specified directory."""
    dataframes_dir = os.path.join(output_dir, 'dataframes')
    os.makedirs(dataframes_dir, exist_ok=True)

    for i, df in enumerate(dataframes):
        df_path = os.path.join(dataframes_dir, f'{prefix}_{i}.csv')
        df.to_csv(df_path, index=False)

def load_dataframes(output_dir, prefix='df'):
    """Loads DataFrames from the specified directory."""
    dataframes_dir = os.path.join(output_dir, 'dataframes')
    if not os.path.exists(dataframes_dir):
        return None

    df_files = sorted(glob.glob(os.path.join(dataframes_dir, f'{prefix}_*.csv')))
    if not df_files:
        return None

    return [pd.read_csv(df_file) for df_file in df_files]

def convert_images(input_dir, output_dir, max_frame=None, brightness_factor=1, contrast_factor=1):
    """Converts and adjusts images from input_dir and saves them in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    input_files = sorted_alphanumeric(glob.glob(os.path.join(input_dir, '*.jpg')))

    if max_frame is None:
        input_files = input_files[:len(input_files)]
    else:
        input_files = input_files[:max_frame]

    output_files = sorted_alphanumeric(glob.glob(os.path.join(output_dir, '*.tif')))

    if len(input_files) == len(output_files):
        print(f"Conversion already completed for {output_dir}. Skipping...")
        return

    num_digits = len(str(len(input_files)))

    for i, file_name in enumerate(input_files):
        image = Image.open(file_name).convert("L")
        image_resized = image.resize((2048, 2048), Image.LANCZOS)

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image_resized)
        image_brightened = enhancer.enhance(brightness_factor)

        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image_brightened)
        image_contrasted = enhancer.enhance(contrast_factor)

        padded_index = str(i + 1).zfill(num_digits)
        base_file_name = f'converted_image_{padded_index}.tif'
        processed_image_path = os.path.join(output_dir, base_file_name)
        image_contrasted.save(processed_image_path, format='TIFF', compression='tiff_lzw')

def combine_timeseries_dataframes(base_data_dir, conditions, subconditions):
    combined_df = pd.DataFrame()

    for condition in conditions:
        for subcondition in subconditions:
            file_path = os.path.join(base_data_dir, condition, subcondition, 'plots', 'dataframes', 'timeseries_df_0.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Condition'] = f'{condition} {subcondition}'
                combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df


def process_piv_workflow(
        conditions, 
        subconditions, 
        base_data_dir, 
        feature_limits, 
        time_interval, 
        max_frame=None, 
        volume_ul=2, 
        brightness_factor=0.8, 
        contrast_factor=0.8
        ):
    """
    Main workflow to process PIV data.
    
    Args:
    - conditions (list of str): List of conditions.
    - subconditions (list of str): List of subconditions.
    - base_data_dir (str): Base directory for the data.
    - time_intervals (list of int): List of time intervals in seconds, corresponding to each condition.
    - max_frame (int, optional): Maximum number of frames to process.
    - volume_ul (int): Volume in microliters.
    - brightness_factor (float): Factor for brightness adjustment.
    - contrast_factor (float): Factor for contrast adjustment.
    """

    
    for i, condition in enumerate(conditions):
        for subcondition in subconditions:
            # Define directories
            input_image_dir = os.path.join(base_data_dir, condition, subcondition, "piv_movie")
            output_image_dir = os.path.join(base_data_dir, condition, subcondition, "piv_movie_8bit_2048x2048")
            output_dir = os.path.join(base_data_dir, condition, subcondition, "plots")

            # Convert images if not already done
            convert_images(input_image_dir, output_image_dir, max_frame, brightness_factor, contrast_factor)

            # Process PIV files
            input_dir = os.path.join(base_data_dir, condition, subcondition, "piv_data", "PIVlab_****.txt")

            if max_frame is None:
                max_frame = len(os.listdir(output_image_dir))

            # Process PIV files if not already done
            if not os.path.exists(os.path.join(output_dir, 'dataframes')):
                dataframes = process_piv_files(input_dir, volume=volume_ul, max_frame=max_frame)
                save_dataframes(dataframes, output_dir)

            # Load all dataframes
            dataframes_dir = os.path.join(output_dir, 'dataframes')
            dataframe_files = os.listdir(dataframes_dir)
            dataframes = [pd.read_csv(os.path.join(dataframes_dir, f)) for f in dataframe_files[:max_frame]]

            # Process time series data if not already done
            timeseries_df_path = os.path.join(output_dir, 'dataframes', 'timeseries_df_0.csv')
            
            # Process time series data with the specific time interval
            if not os.path.exists(timeseries_df_path):
                df = piv_time_series(dataframes, time_interval_seconds=time_interval)
                save_dataframes([df], output_dir, prefix='timeseries_df')
            else:
                df = pd.read_csv(timeseries_df_path)

            # Generate heatmaps for each feature
            for feature, (vmin, vmax) in feature_limits.items():
                generate_heatmaps(dataframes, feature, vmin=vmin, vmax=vmax, output_dir_base=output_dir, image_path=output_image_dir)


            # Make movies
            make_movies_from_features(output_dir, fps=120)
