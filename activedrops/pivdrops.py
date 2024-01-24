# Import standard libraries
import os
import re
import sys
import glob

# Import data processing libraries
import pandas as pd
import numpy as np

# Import image processing libraries
import cv2
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter

# Import plotting and visualization libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import colorcet as cc
from scipy.optimize import curve_fit
from ipywidgets import interact, FloatSlider

# Additional utilities
from natsort import natsorted  # For natural sorting
from sklearn.decomposition import PCA


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

######################################### style #########################################
# Default RP plotting style
def set_plotting_style():
    """
    Formats plotting environment to that used in Physical Biology of the Cell,
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



######################################### raw data processing #########################################


def plot_autocorrelation_values(lambda_tau, results, fitted_values, filename=None):
    """
    Plots the autocorrelation values and the fitted exponential decay with scaled x-axis.
    
    Parameters:
    - lambda_tau (float): Correlation length.
    - results (array): Array of autocorrelation values.
    - fitted_values (array): Array of fitted values.
    - filename (str, optional): If provided, the plot will be saved to this filename.
    
    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    intervector_distance_microns = 21.745

    # Scale x-axis by 20
    x_values = np.arange(len(results)) * intervector_distance_microns  # Generate scaled x-coordinates by intervector distance

    # Plot autocorrelation values and fitted exponential decay with scaled x-axis
    plt.plot(x_values, results, label='Autocorrelation Values', marker='o', linestyle='-', markersize=5)
    plt.plot(x_values, fitted_values, label='Fitted Exponential Decay', linestyle='--', color='red')
    plt.axvline(x=lambda_tau, color='green', linestyle='-.', label=f'Correlation Length = {lambda_tau:.2f} µm')

    # Adding labels, title, and legend
    plt.xlabel('Scaled Lag (µm)')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function and Fitted Exponential Decay')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1.1)

    plt.tight_layout()

    # If filename is provided, save the plot
    if filename:
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename, dpi=200, format='jpg')
        plt.close()
    else:
        plt.show()
        plt.close()


def correlation_length(df, plot_autocorrelation=True):

    # Obtain square grid of velocity magnitudes
    v = df.pivot(index='y [m]', columns='x [m]', values="velocity magnitude [m/s]").values

    # Calculate intervector distance
    intervector_distance_microns = ((df["y [m]"].max() - df["y [m]"].min()) / v.shape[0])

    # Calculate the autocorrelation function with Fourier transform
    full_product = np.fft.fft2(v) * np.conj(np.fft.fft2(v))
    inverse = np.real(np.fft.ifft2(full_product))  # Real part of the inverse Fourier transform
    normalized_inverse = inverse / inverse[0, 0]  # Normalize the autocorrelation function

    # Compute the autocorrelation for each r value
    r_values = v.shape[0] // 2
    results = np.zeros(r_values)
    for r in range(r_values):
        autocorrelation_value = (normalized_inverse[r, r] + normalized_inverse[-r, -r]) / (v.shape[0] * v.shape[1])
        results[r] = autocorrelation_value 

    # Normalize the results array
    results /= results[0]

    # Fit the results to an exponential decay model
    def exponential_decay(x, A, B, C):
        return A * np.exp(-x / B) + C

    params, _ = curve_fit(exponential_decay, np.arange(len(results)), results, maxfev=5000)
    A, B, C = params
    fitted_values = exponential_decay(np.arange(r_values), A, B, C)

    # Compute correlation length and other parameters
    lambda_tau = -B * np.log((0.3 - C) / A) * intervector_distance_microns 

    if plot_autocorrelation:
        # Plot and save autocorrelation values
        plot_autocorrelation_values(lambda_tau * 1e6, results, fitted_values)


    return lambda_tau 



def df_piv(data_path, condition, subcondition, min_frame=0, max_frame=None):
    """
    Processes Particle Image Velocimetry (PIV) data to create a DataFrame that combines mean values, 
    power calculations, and pivot matrices for each feature.

    Args:
        data_path (str): Path to the directory containing PIV data files.
        condition (str): Condition label for the data set.
        subcondition (str): Subcondition label for the data set.
        min_frame (int, optional): Minimum frame index to start processing (inclusive).
        max_frame (int, optional): Maximum frame index to stop processing (exclusive).

    Returns:
        pandas.DataFrame: A DataFrame where each row corresponds to a frame, combining mean values, 
        power calculations, and pivot matrices for each feature.
    """

    input_piv_data = os.path.join(data_path, condition, subcondition, "piv_data", "PIVlab_****.txt")
    
    # Using a for loop instead of list comprehension
    dfs = []
    for file in sorted(glob.glob(input_piv_data))[min_frame:max_frame]:
        df = pd.read_csv(file, skiprows=2).fillna(0).rename(columns={
            "magnitude [m/s]": "velocity magnitude [m/s]",
            "simple shear [1/s]": "shear [1/s]",
            "simple strain [1/s]": "strain [1/s]",
            "Vector type [-]": "data type [-]"
        })
        dfs.append(df)

    return dfs



def process_piv_data(data_path, condition, subcondition, min_frame=0, max_frame=None, plot_autocorrelation=True):
    """
    Generates a time series pivot DataFrame from input data.

    Parameters:
    data_path (str): Path to the input data file.
    condition (str): Primary condition for data filtering.
    subcondition (str): Secondary condition for further data filtering.
    min_frame (int, optional): Minimum frame to consider in the analysis. Defaults to 0.
    max_frame (int, optional): Maximum frame to consider in the analysis. If None, considers all frames. Defaults to None.
    plot_autocorrelation (bool, optional): Flag to plot autocorrelation. Defaults to True.
    time_interval (int, optional): Time interval between frames, in seconds. Defaults to 3.

    Returns:
    tuple: A tuple containing two pandas DataFrames. The first is the mean values DataFrame and the second is the pivot matrices DataFrame.
    """
    # Creating output directories
    output_directory_dfs = os.path.join(data_path, condition, subcondition, "dataframes_PIV")
    os.makedirs(output_directory_dfs, exist_ok=True)


    # Reading data frames
    # Note: Assuming df_piv and correlation_length are pre-defined functions
    data_frames = df_piv(data_path, condition, subcondition, min_frame, max_frame)


    # Calculating mean values with valid vectors only
    mean_values = []
    for data_frame in data_frames:
        data_frame["correlation length [m]"] = correlation_length(data_frame, plot_autocorrelation)
        data_frame = data_frame[data_frame["data type [-]"] == 1]
        mean_values.append(data_frame.mean(axis=0))

    # Creating mean DataFrame
    mean_data_frame = pd.DataFrame(mean_values)
    mean_data_frame.reset_index(drop=False, inplace=True)
    mean_data_frame.rename(columns={'index': 'frame'}, inplace=True)

    # Calculate power and add to DataFrame
    volume = 2E-9  # µl --> m^3
    viscosity = 1E-3  # mPa*S
    mean_data_frame["power [W]"] = volume * viscosity * (mean_data_frame["velocity magnitude [m/s]"]/mean_data_frame["correlation length [m]"])**2

    # Renaming time column
    # mean_data_frame.rename(columns={'frame': 'time [min]'}, inplace=True)

    # Remove unnecessary columns for the pivot matrices
    mean_data_frame = mean_data_frame.iloc[:, 5:]

    # Scale time appropriately
    mean_data_frame["frame"] = np.arange(len(mean_data_frame)) 



    # Creating pivot matrices for each feature
    features = data_frames[0].columns[:-1]
    pivot_matrices = {feature: [] for feature in features}

    for data_frame in data_frames:
        temporary_dictionary = {feature: data_frame.pivot(index='y [m]', columns='x [m]', values=feature).values for feature in features}
        for feature in features:
            pivot_matrices[feature].append(temporary_dictionary[feature])

    pivot_data_frame = pd.DataFrame(pivot_matrices)

    # Adjusting column names in mean_data_frame
    mean_data_frame.columns = [f"{column}_mean" if column != "frame" else column for column in mean_data_frame.columns]
    
    # Adding time column to pivot_data_frame
    pivot_data_frame["frame"] = mean_data_frame["frame"].values

    # Save DataFrames to CSV
    mean_df_output_path = os.path.join(output_directory_dfs, "mean_values.csv")
    mean_data_frame.to_csv(mean_df_output_path, index=False)

    pivot_df_output_path = os.path.join(output_directory_dfs, "features_matrices.csv")
    pivot_data_frame.to_csv(pivot_df_output_path, index=False)

    return mean_data_frame, pivot_data_frame.iloc[:, 2:]



def convert_images(data_path, condition, subcondition, max_frame=None, brightness_factor=1, contrast_factor=1):
    """
    Converts, resizes, and adjusts the brightness and contrast of images located in a specified 
    directory and saves the processed images in a new directory. The function identifies images based on 
    a specified data path, condition, and subcondition.

    This function is specifically tailored for converting PIV (Particle Image Velocimetry) movie images. 
    It supports adjustments in brightness and contrast, and checks to avoid re-processing already 
    converted images.

    Args:
    - data_path (str): Base directory where the original PIV movie images are stored.
    - condition (str): Specific condition defining a subdirectory within the data path.
    - subcondition (str): Specific subcondition defining a sub-subdirectory within the condition directory.
    - max_frame (int, optional): Maximum number of images to process. If None, all images in the directory are processed.
    - brightness_factor (float, optional): Factor to adjust the brightness of the images. Defaults to 1 (no change).
    - contrast_factor (float, optional): Factor to adjust the contrast of the images. Defaults to 1 (no change).
    """

    # Construct input and output directories based on provided path, condition, and subcondition
    input_dir = f"{data_path}/{condition}/{subcondition}/piv_movie/"
    output_dir = f"{data_path}/{condition}/{subcondition}/piv_movie_converted/"

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Gather all JPEG images from the input directory
    input_files = natsorted(glob.glob(os.path.join(input_dir, '****.jpg')))

    # Limit the processing to max_frame if specified
    input_files = input_files[:max_frame] if max_frame is not None else input_files

    # Check if the output directory already has the converted files
    output_files = natsorted(glob.glob(os.path.join(output_dir, '****.tif')))
    if len(input_files) == len(output_files):
        print(f"Conversion already completed for {output_dir}. Skipping...")
        return

    # Prepare for filename formatting
    num_digits = len(str(len(input_files)))

    # Process each image
    for i, file_name in enumerate(input_files):
        # Open and convert image to grayscale
        image = Image.open(file_name).convert("L")

        # Resize image to 2048x2048 pixels
        image_resized = image.resize((2048, 2048), Image.LANCZOS)

        # Adjust brightness and contrast
        enhancer = ImageEnhance.Brightness(image_resized)
        image_brightened = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(image_brightened)
        image_contrasted = enhancer.enhance(contrast_factor)

        # Prepare the filename and save the processed image
        padded_index = str(i + 1).zfill(num_digits)
        base_file_name = f'converted_image_{padded_index}.tif'
        processed_image_path = os.path.join(output_dir, base_file_name)
        image_contrasted.save(processed_image_path, format='TIFF', compression='tiff_lzw')



def piv_heatmap(df, data_path, condition, subcondition, feature_limits, time_interval=3):
    """
    Generates and saves heatmaps for each feature specified in the feature_limits dictionary.
    Each heatmap is overlaid on a corresponding image and saved to a structured directory.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to plot. Each column represents a feature,
                      and each row represents a frame.
    - data_path (str): Base path for reading source images and saving heatmaps.
    - condition (str): Condition name, used for directory structuring.
    - subcondition (str): Subcondition name, further specifying the directory structure.
    - feature_limits (dict): A dictionary where keys are feature names (column names in df) and
                             values are tuples (vmin, vmax) representing the limits for the heatmap.
    - time_interval (int, optional): Time interval between frames, used for time annotation in the plot title. 
                                     Default is 3.

    The function creates a directory structure under 'data_path' for each feature to store the heatmaps.
    The structure is: data_path/condition/subcondition/heatmaps_PIV/feature_name/.

    Heatmaps are generated for each frame (row in df) and saved as JPEG images.
    """
    
    for feature, limits in feature_limits.items():
        vmin, vmax = limits

        for j in range(len(df)):
            vals = df.iloc[j, df.columns.get_loc(feature)]

            output_directory_heatmaps = os.path.join(data_path, condition, subcondition, "heatmaps_PIV", f"{feature.split()[0]}", f"{feature.split()[0]}_heatmap_{j}.jpg")
            image_files_pattern = f"{data_path}/{condition}/{subcondition}/piv_movie_converted/converted_image_****.tif"
            image_files = sorted(glob.glob(image_files_pattern))[j]
            image = Image.open(image_files)

            plt.figure(figsize=(10, 6))
            plt.imshow(image, cmap=None, extent=[-2762/2, 2762/2, -2762/2, 2762/2]) # piv image
            im = plt.imshow(vals, cmap='inferno', origin='lower', alpha=0.7, extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax) # heatmap
            plt.xlabel('x [um]')
            plt.ylabel('y [um]')
            cbar = plt.colorbar(im)
            cbar.set_label(feature)
            time = df.iloc[j, -1]
            plt.title(f"PIV - {feature}  ||  time: {time * time_interval/60} min")

            os.makedirs(os.path.dirname(output_directory_heatmaps), exist_ok=True)
            plt.savefig(output_directory_heatmaps, format='jpg', dpi=250)
            plt.close()



def create_heatmap_movies(data_path, condition, subcondition, feature_limits, frame_rate=120, max_frame=None):
    """
    Creates heatmap video files from heatmap images stored in a specified directory.

    Args:
    - data_path (str): Base path where the heatmap images are stored.
    - condition (str): Condition under which the heatmap images are stored.
    - subcondition (str): Subcondition under which the heatmap images are stored.
    - feature_limits (dict): Dictionary specifying the limits for each feature.
    - frame_rate (int, optional): Frame rate for the output video. Defaults to 120.
    - max_frame (int, optional): Maximum number of frames to be included in the video. If None, all frames are included.

    The function reads heatmap images from the specified directory and creates a video file for each feature.
    """

    plots_dir = f"{data_path}/{condition}/{subcondition}/heatmaps_PIV/"
    for feature in feature_limits.keys():
        feature_name_for_file = feature.split()[0]
        heatmap_dir = os.path.join(data_path, condition, subcondition, "heatmaps_PIV", f"{feature.split()[0]}", f"{feature.split()[0]}_heatmap_****.jpg")
        heatmap_files = natsorted(glob.glob(heatmap_dir))

        if not heatmap_files:
            continue

        # Limit the number of files if max_frame is specified
        heatmap_files = heatmap_files[:max_frame] if max_frame is not None else heatmap_files

        # Get the resolution of the first image (assuming all images are the same size)
        first_image = cv2.imread(heatmap_files[0])
        video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(f'{plots_dir}{feature_name_for_file}.avi', fourcc, frame_rate, video_resolution)

        for file in heatmap_files:
            img = cv2.imread(file)
            out.write(img)  # Write the image as is, without resizing

        out.release()


def plot_features(dfs, data_paths, conditions, subconditions, time_intervals, sigma=2):
    """
    Plots each feature with respect to frame for multiple DataFrames.

    Parameters:
    - dfs (list of DataFrame): List of DataFrames to plot.
    - data_paths (list of str): List of base paths for saving the plots, corresponding to each DataFrame.
    - conditions (list of str): List of condition names corresponding to each DataFrame.
    - subconditions (list of str): List of subcondition names corresponding to each DataFrame.
    - time_intervals (list of int): List of time intervals between frames, used for x-axis scaling, corresponding to each DataFrame.
    - sigma (int, optional): Standard deviation for Gaussian filter applied to the data. Default is 2.

    The function creates a plot for each feature in the DataFrame(s), combining data from all provided
    DataFrames. Plots are saved as JPEG images in the specified data_paths.
    """

    for feature in dfs[0].columns[:-1]:
        plt.figure(figsize=(10, 6))

        for df, data_path, condition, subcondition, time_interval in zip(dfs, data_paths, conditions, subconditions, time_intervals):
            output_directory_plots = os.path.join(data_path, condition, subcondition, "plots_PIV", f"{feature.split()[0]}_plot.jpg")
            os.makedirs(os.path.dirname(output_directory_plots), exist_ok=True)
            filtered_values = gaussian_filter(df[feature], sigma=sigma)
            plt.plot(df["frame"] * (time_interval/60), filtered_values, marker='o', linestyle='-', markersize=2, linewidth=2, label=f'{condition}_{subcondition}')

        plt.xlabel('Time (minutes)')
        plt.ylabel(feature)
        plt.title(f"PIV - {feature}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(output_directory_plots, format='jpg', dpi=250)
        plt.close()



def plot_pca(dfs, data_paths, conditions, subconditions, features):
    # Perform PCA and Plot
    plt.figure(figsize=(10, 6))

    # Get colors from Seaborn's "colorblind" color palette
    sns.set_palette("colorblind", color_codes=True)
    colors = sns.color_palette("colorblind", n_colors=len(data_paths))

    for group_index, (df, data_path, condition, subcondition) in enumerate(zip(dfs, data_paths, conditions, subconditions)):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(df.loc[:, features])
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

        # Scaling alpha to increase with respect to the frame index
        num_points = principalDf.shape[0]
        alphas = np.linspace(0.01, 1, num_points)  # Alpha values linearly spaced from 1 to 0.01

        # Plotting each line segment with increasing alpha
        for i in range(1, num_points):
            plt.plot(principalDf['principal component 1'][i-1:i+1], principalDf['principal component 2'][i-1:i+1], 
                     alpha=alphas[i], linestyle='-', linewidth=1, color=colors[group_index])

        # Plotting the points
        plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], 
                    alpha=0.5, label=f'{condition}_{subcondition}', s=10, color=colors[group_index])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of PIV Features (All Samples)')
    plt.legend()
    plt.grid(True)

    output_dir_pca = os.path.join(data_paths[-1], conditions[-1], subconditions[-1], "plots_PIV", "PCA.jpg")
    os.makedirs(os.path.dirname(output_dir_pca), exist_ok=True)
    plt.savefig(output_dir_pca, format='jpg', dpi=250)
    plt.close()



# def read_piv_dataframe(file):
#     """
#     Reads a PIV (Particle Image Velocimetry) data file and preprocesses it by converting units.
    
#     Args:
#     - file (str): Path to the PIV data file.
    
#     Returns:
#     - DataFrame: A preprocessed DataFrame with unit-converted columns.
#     """
#     # Read and preprocess the dataframe
#     df = pd.read_csv(file, skiprows=2).fillna(0)
    
#     # Convert measurements to micrometers and micrometers per second
#     df['x [um]'] = df['x [m]'] * 1E6
#     df['y [um]'] = df['y [m]'] * 1E6
#     df['u [um/s]'] = df['u [m/s]'] * 1E6
#     df['v [um/s]'] = df['v [m/s]'] * 1E6
#     df['magnitude [um/s]'] = df['magnitude [m/s]'] * 1E6
    
#     return df


# def plot_autocorrelation_values(lambda_tau, results, fitted_values, filename=None):
#     """
#     Plots the autocorrelation values and the fitted exponential decay with scaled x-axis.
    
#     Parameters:
#     - lambda_tau (float): Correlation length.
#     - results (array): Array of autocorrelation values.
#     - fitted_values (array): Array of fitted values.
#     - filename (str, optional): If provided, the plot will be saved to this filename.
    
#     Returns:
#     - None
#     """
#     plt.figure(figsize=(10, 6))
#     intervector_distance_microns = 21.745

#     # Scale x-axis by 20
#     x_values = np.arange(len(results)) * intervector_distance_microns  # Generate scaled x-coordinates by intervector distance

#     # Plot autocorrelation values and fitted exponential decay with scaled x-axis
#     plt.plot(x_values, results, label='Autocorrelation Values', marker='o', linestyle='-', markersize=5)
#     plt.plot(x_values, fitted_values, label='Fitted Exponential Decay', linestyle='--', color='red')
#     plt.axvline(x=lambda_tau, color='green', linestyle='-.', label=f'Correlation Length = {lambda_tau:.2f} µm')

#     # Adding labels, title, and legend
#     plt.xlabel('Scaled Lag (µm)')
#     plt.ylabel('Autocorrelation')
#     plt.title('Autocorrelation Function and Fitted Exponential Decay')
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.ylim(0, 1.1)

#     plt.tight_layout()

#     # If filename is provided, save the plot
#     if filename:
#         directory = os.path.dirname(filename)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         plt.savefig(filename, dpi=200, format='jpg')
#         plt.close()
#     else:
#         plt.show()
#         plt.close()




# def calculate_autocorrelation_and_power(file, volume, plot_dir, plot_autocorrelation=True):
#     """
#     Computes the correlation length and completes the DataFrame with additional computed parameters.
    
#     Args:
#     - df (DataFrame): The DataFrame containing the preprocessed PIV data.
#     - file (str): Path to the PIV data file.
#     - volume (float): The volume in microliters for power calculation.
#     - plot_dir (str): Directory to save the autocorrelation plot.
    
#     Returns:
#     - DataFrame: A DataFrame with added columns for computed parameters like correlation length and power.
#     """

#     df = read_piv_dataframe(file)

#     # Obtain square grid of velocity magnitudes
#     v = df.pivot(index='y [um]', columns='x [um]', values="magnitude [um/s]").values

#     # Calculate intervector distance
#     intervector_distance_microns = (df["y [um]"].max() - df["y [um]"].min()) / v.shape[0]

#     # Calculate the autocorrelation function with Fourier transform
#     full_product = np.fft.fft2(v) * np.conj(np.fft.fft2(v))
#     inverse = np.real(np.fft.ifft2(full_product))  # Real part of the inverse Fourier transform
#     normalized_inverse = inverse / inverse[0, 0]  # Normalize the autocorrelation function

#     # Compute the autocorrelation for each r value
#     r_values = v.shape[0] // 2
#     results = np.zeros(r_values)
#     for r in range(r_values):
#         autocorrelation_value = (normalized_inverse[r, r] + normalized_inverse[-r, -r]) / (v.shape[0] * v.shape[1])
#         results[r] = autocorrelation_value 

#     # Normalize the results array
#     results /= results[0]

#     # Fit the results to an exponential decay model
#     def exponential_decay(x, A, B, C):
#         return A * np.exp(-x / B) + C

#     params, _ = curve_fit(exponential_decay, np.arange(len(results)), results, maxfev=5000)
#     A, B, C = params
#     fitted_values = exponential_decay(np.arange(r_values), A, B, C)

#     # Compute correlation length and other parameters
#     lambda_tau = -B * np.log((0.3 - C) / A) * intervector_distance_microns
#     df["correlation length (µm)"] = lambda_tau

#     # Calculate power
#     v0 = volume * 1E-9  # µl --> m^3
#     µ = 1E-3  # mPa*S
#     correlation_length = lambda_tau * 1E-6  # µm --> m
#     df["Power (W)"] = v0 * µ * (df[df["magnitude [m/s]"] > 0]["magnitude [m/s]"].mean() / correlation_length)**2

#     # Calculate the mean of non-zero velocity magnitudes
#     df["mean velocity [um/s]"] = df[df["magnitude [um/s]"] > 0]["magnitude [um/s]"].mean()

#     # Take the last four digits of the file name and convert to integer
#     df["frame"] = int(os.path.basename(file).split('.')[0][-4:]) - 1
#     # # Convert time to minutes
#     # df["time [min]"] = df["time [min]"] * seconds_interval / 60

#     # Reorganize DataFrame
#     columns = list(df.columns[-5:]) + list(df.columns[:-5])
#     df = df[columns]

#     if plot_autocorrelation:
#         # Plot and save autocorrelation values
#         plot_autocorrelation_values(r_values, results, fitted_values, plot_dir, file)

#     return df


# def process_and_save_piv_files(data_path, condition, subcondition, volume=2, max_frame=None, save_csv=True, plot_autocorrelation=True):
#     """
#     Processes PIV data files and optionally saves them into separate CSV files. The function reads files 
#     from a specified directory, processes each using the df_piv function, and conditionally saves each 
#     resulting DataFrame as a CSV file in a specified output directory.

#     Args:
#     - data_path (str): Base directory where PIV data files are stored.
#     - condition (str): The specific condition (subdirectory) under which the PIV data is stored.
#     - subcondition (str): The subcondition (sub-subdirectory) under which the PIV data is stored.
#     - volume (float, optional): Volume parameter for the df_piv function. Defaults to 2.
#     - max_frame (int, optional): Maximum number of files to process. If None, all files are processed.
#     - save_csv (bool, optional): Flag to decide whether to save the processed DataFrames as CSV files. Defaults to True.

#     Returns:
#     - List[DataFrame]: A list of DataFrames, each corresponding to a processed PIV file.
#     """
    
#     # Find input directory
#     input_piv_data = os.path.join(data_path, condition, subcondition, "piv_data", "PIVlab_****.txt")

#     # Define output directory based on input parameters
#     output_dir = os.path.join(data_path, condition, subcondition, "dataframes_PIV")
    
#     # Define plot directory
#     plot_dir = os.path.join(data_path, condition, subcondition, "plots", "autocorrelation")
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)

#     # Ensure the output directory exists
#     if save_csv and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Find and process files
#     files = sorted(glob.glob(input_piv_data))
#     dataframes = []
#     for i, file in enumerate(files[:max_frame]):
#         df = calculate_autocorrelation_and_power(file, volume, plot_dir, plot_autocorrelation)  
#         if save_csv:
#             df.to_csv(os.path.join(output_dir, f"PIV_dataframe_{i}.csv"), index=False)
#         dataframes.append(df)

#     return dataframes



# ######################################### image processing #########################################

# def convert_images(data_path, condition, subcondition, max_frame=None, brightness_factor=1, contrast_factor=1):
#     """
#     Converts, resizes, and adjusts the brightness and contrast of images located in a specified 
#     directory and saves the processed images in a new directory. The function identifies images based on 
#     a specified data path, condition, and subcondition.

#     This function is specifically tailored for converting PIV (Particle Image Velocimetry) movie images. 
#     It supports adjustments in brightness and contrast, and checks to avoid re-processing already 
#     converted images.

#     Args:
#     - data_path (str): Base directory where the original PIV movie images are stored.
#     - condition (str): Specific condition defining a subdirectory within the data path.
#     - subcondition (str): Specific subcondition defining a sub-subdirectory within the condition directory.
#     - max_frame (int, optional): Maximum number of images to process. If None, all images in the directory are processed.
#     - brightness_factor (float, optional): Factor to adjust the brightness of the images. Defaults to 1 (no change).
#     - contrast_factor (float, optional): Factor to adjust the contrast of the images. Defaults to 1 (no change).
#     """

#     # Construct input and output directories based on provided path, condition, and subcondition
#     input_dir = f"{data_path}{condition}/{subcondition}/piv_movie/"
#     output_dir = f"{data_path}{condition}/{subcondition}/piv_movie_converted/"

#     # Create the output directory if it does not exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Gather all JPEG images from the input directory
#     input_files = natsorted(glob.glob(os.path.join(input_dir, '****.jpg')))

#     # Limit the processing to max_frame if specified
#     input_files = input_files[:max_frame] if max_frame is not None else input_files

#     # Check if the output directory already has the converted files
#     output_files = natsorted(glob.glob(os.path.join(output_dir, '****.tif')))
#     if len(input_files) == len(output_files):
#         print(f"Conversion already completed for {output_dir}. Skipping...")
#         return

#     # Prepare for filename formatting
#     num_digits = len(str(len(input_files)))

#     # Process each image
#     for i, file_name in enumerate(input_files):
#         # Open and convert image to grayscale
#         image = Image.open(file_name).convert("L")

#         # Resize image to 2048x2048 pixels
#         image_resized = image.resize((2048, 2048), Image.LANCZOS)

#         # Adjust brightness and contrast
#         enhancer = ImageEnhance.Brightness(image_resized)
#         image_brightened = enhancer.enhance(brightness_factor)
#         enhancer = ImageEnhance.Contrast(image_brightened)
#         image_contrasted = enhancer.enhance(contrast_factor)

#         # Prepare the filename and save the processed image
#         padded_index = str(i + 1).zfill(num_digits)
#         base_file_name = f'converted_image_{padded_index}.tif'
#         processed_image_path = os.path.join(output_dir, base_file_name)
#         image_contrasted.save(processed_image_path, format='TIFF', compression='tiff_lzw')



# # def piv_heatmap(df, feature, vmin=None, vmax=None, time_interval=3, output_dir=None):


# #     vals = df.pivot(index='y [um]', columns='x [um]', values=feature).values
    
# #     plt.figure(figsize=(10, 6))
# #     im = plt.imshow(vals, cmap='viridis', origin='lower', extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax)
# #     plt.xlabel('x [um]')
# #     plt.ylabel('y [um]')
# #     cbar = plt.colorbar(im)
# #     cbar.set_label(feature)
# #     plt.title(f'{feature} - time: {df["frame"][0] * time_interval / 60} min')

# #     if output_dir:
# #         os.makedirs(os.path.dirname(output_dir), exist_ok=True)
# #         plt.savefig(output_dir, format='jpg', dpi=250)
# #         plt.close()
# #     else:
# #         plt.show()




# def piv_heatmap_piv(df, feature, vmin=None, vmax=None, time_interval=3, output_directory=None, image_file_path=None):
#     """
#     Generates a heatmap from Particle Image Velocimetry (PIV) data and either overlays it on an image or creates a standalone heatmap.

#     Args:
#         data_frame (pandas.DataFrame): The DataFrame containing the PIV data.
#         feature (str): The feature for which to generate the heatmap.
#         minimum_value (float): Minimum value for colormap scaling in the heatmap.
#         maximum_value (float): Maximum value for colormap scaling in the heatmap.
#         time_in_minutes (float): Time in minutes for the current frame.
#         output_directory (str, optional): Directory to save the plot. If None, the plot is displayed.
#         image_file_path (str, optional): Path to an image file on which to overlay the heatmap.

#     If an image file path is provided, the heatmap is overlaid on the image. Otherwise, a standalone heatmap is generated.
#     """

#     # Prepare the heatmap data
#     heatmap_data = df.pivot(index='y [um]', columns='x [um]', values=feature).values

#     # Check if image file is provided for overlay
#     if image_file_path:
#         # Load the image
#         image = Image.open(image_file_path)

#         # Create a plot to overlay the heatmap on the image
#         plt.figure(figsize=(10, 6))
#         plt.imshow(image, cmap=None, extent=[-2762/2, 2762/2, -2762/2, 2762/2])  # Display the image
#         im = plt.imshow(heatmap_data, cmap='inferno', origin='lower', alpha=0.7, extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax)  # Overlay the heatmap
#     else:
#         # Generate a standalone heatmap
#         plt.figure(figsize=(10, 6))
#         im = plt.imshow(heatmap_data, cmap='viridis', origin='lower', extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax)

#     # Set the labels and title
#     plt.xlabel('x [um]')
#     plt.ylabel('y [um]')
#     cbar = plt.colorbar(im)
#     cbar.set_label(feature)
#     title = f'{feature} - time: {df["frame"][0] * time_interval / 60} min'
#     plt.title(title)

#     # Save or show the plot
#     if output_directory:
#         os.makedirs(os.path.dirname(output_directory), exist_ok=True)
#         plt.savefig(output_directory, format='jpg', dpi=250)
#     else:
#         plt.show()


        



# def overlay_heatmap_on_image(image_file, df, heatmap_data, feature, vmin, vmax, time_in_minutes, output_dir=None):
#     """
#     Overlays a heatmap on an image and either saves or displays the combined visualization.

#     Args:
#     - image_file (str): Path to the image file.
#     - df (DataFrame): The DataFrame containing the PIV data.
#     - heatmap_data (np.array): Data for the heatmap.
#     - feature (str): Name of the feature for which the heatmap is generated.
#     - vmin (float): Minimum value for colormap scaling.
#     - vmax (float): Maximum value for colormap scaling.
#     - time_in_minutes (float): Time in minutes for the current frame.
#     - output_dir (str, optional): Directory to save the plot. If None, the plot is displayed.

#     The function loads the image, applies the heatmap on top with specified parameters, and either
#     saves or displays the combined image, based on the provided output directory.
#     """

#     # Load the image
#     image = Image.open(image_file)

#     # Create a plot to overlay the heatmap on the image
#     plt.figure(figsize=(10, 6))
#     plt.imshow(image, cmap=None, extent=[-2762/2, 2762/2, -2762/2, 2762/2])  # Display the image
#     im = plt.imshow(heatmap_data, cmap='inferno', origin='lower', alpha=0.7, extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax)  # Overlay the heatmap
#     plt.xlabel('x [um]')
#     plt.ylabel('y [um]')
#     cbar = plt.colorbar(im)
#     cbar.set_label(feature)
#     plt.title(f'PIV Heatmap - {df["file name"][0]} || Time: {time_in_minutes:.2f} min')

#     # Save or show the plot
#     if output_dir:
#         os.makedirs(os.path.dirname(output_dir), exist_ok=True)
#         plt.savefig(output_dir, format='jpg', dpi=250)
#         plt.close()
#     else:
#         plt.show()



# # def piv_heatmap(df, feature, vmin, vmax, time_in_minutes, output_dir=None, image_file=None):
# #     """
# #     Generates a heatmap for a specific feature from PIV data and optionally overlays it on an image.

# #     Args:
# #     - df (DataFrame): The DataFrame containing the PIV data.
# #     - feature (str): The feature for which to generate the heatmap.
# #     - vmin (float): Minimum value for colormap scaling.
# #     - vmax (float): Maximum value for colormap scaling.
# #     - time_in_minutes (float): Time in minutes for the current frame.
# #     - output_dir (str, optional): Directory to save the plot. If None, the plot is displayed.
# #     - image_file (str, optional): Path to an image file on which to overlay the heatmap.

# #     The function creates a heatmap from the provided DataFrame and feature. If an image file is provided,
# #     the heatmap is overlaid on the image; otherwise, a standalone heatmap is generated.
# #     """

# #     # Extract values for the heatmap
# #     vals = df.pivot(index='y [um]', columns='x [um]', values=feature).values

# #     if image_file:
# #         # Overlay the heatmap on the image if an image file is provided
# #         overlay_heatmap_on_image(image_file, df, vals, feature, vmin, vmax, time_in_minutes, output_dir)
# #     else:
# #         # Generate a standalone heatmap
# #         plt.figure(figsize=(10, 6))
# #         im = plt.imshow(vals, cmap='viridis', origin='lower', extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax)
# #         plt.xlabel('x [um]')
# #         plt.ylabel('y [um]')
# #         cbar = plt.colorbar(im)
# #         cbar.set_label(feature)
# #         plt.title(f'PIV Heatmap - Time: {df[0].iloc[:1, 4:5].values} min')

# #         if output_dir:
# #             os.makedirs(os.path.dirname(output_dir), exist_ok=True)
# #             plt.savefig(output_dir, format='jpg', dpi=250)
# #             plt.close()
# #         else:
# #             plt.show()



# # def generate_heatmaps_for_features(data_path, condition, subcondition, feature_limits, dfs):
# #     """
# #     Automates the generation of heatmaps for multiple features across different data frames.

# #     Args:
# #     - data_path (str): Base directory where PIV data and images are stored.
# #     - condition (str): Specific condition defining a subdirectory within the data path.
# #     - subcondition (str): Specific subcondition defining a sub-subdirectory within the condition directory.
# #     - feature_limits (dict): Dictionary mapping features to their corresponding value limits (vmin, vmax).
# #     - dfs (List[DataFrame]): List of DataFrames containing PIV data.

# #     This function iterates over each DataFrame in 'dfs', generating heatmaps for each feature specified
# #     in 'feature_limits'. The heatmaps are either saved or displayed based on the provided output directory.
# #     """

# #     # Retrieve the list of converted image files
# #     image_files_pattern = f"{data_path}{condition}/{subcondition}/piv_movie_converted/converted_image_****.tif"
# #     image_files = natsorted(glob.glob(image_files_pattern))

# #     # Iterate over each DataFrame and feature to generate heatmaps
# #     for i, df in enumerate(dfs):
# #         for feature, limits in feature_limits.items():
# #             vmin, vmax = limits
# #             feature_name_for_file = re.sub(r"\s*\[.*?\]\s*", "", feature).replace(" ", "_").lower()
            
# #             # Zero-pad the frame number
# #             frame_number = str(i).zfill(4)  # Adjust the number 3 based on the maximum number of frames

# #             heatmap_output = f"{data_path}{condition}/{subcondition}/plots/{feature_name_for_file}/heatmap_{frame_number}.jpg"
# #             piv_heatmap(df, feature, vmin=vmin, vmax=vmax, time_in_minutes=i, image_file=image_files[i], output_dir=heatmap_output)
        

        



# def create_heatmap_movies(data_path, condition, subcondition, feature_limits, frame_rate=120, max_frame=None):
#     """
#     Creates heatmap video files from heatmap images stored in a specified directory.

#     Args:
#     - data_path (str): Base path where the heatmap images are stored.
#     - condition (str): Condition under which the heatmap images are stored.
#     - subcondition (str): Subcondition under which the heatmap images are stored.
#     - feature_limits (dict): Dictionary specifying the limits for each feature.
#     - frame_rate (int, optional): Frame rate for the output video. Defaults to 120.
#     - max_frame (int, optional): Maximum number of frames to be included in the video. If None, all frames are included.

#     The function reads heatmap images from the specified directory and creates a video file for each feature.
#     """

#     plots_dir = f"{data_path}{condition}/{subcondition}/plots/"
#     for feature in feature_limits.keys():
#         feature_name_for_file = re.sub(r"\s*\[.*?\]\s*", "", feature).replace(" ", "_").lower()
#         heatmap_dir = f"{data_path}{condition}/{subcondition}/plots/{feature_name_for_file}/"
#         heatmap_files = natsorted(glob.glob(f"{heatmap_dir}heatmap_****.jpg"))

#         if not heatmap_files:
#             continue

#         # Limit the number of files if max_frame is specified
#         heatmap_files = heatmap_files[:max_frame] if max_frame is not None else heatmap_files

#         # Get the resolution of the first image (assuming all images are the same size)
#         first_image = cv2.imread(heatmap_files[0])
#         video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

#         # Define the codec and create VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#         out = cv2.VideoWriter(f'{plots_dir}{feature_name_for_file}.avi', fourcc, frame_rate, video_resolution)

#         for file in heatmap_files:
#             img = cv2.imread(file)
#             out.write(img)  # Write the image as is, without resizing

#         out.release()




# # def calculate_mean_over_time(data_path, condition, subcondition, seconds_interval, min_frame=None, max_frame=None):
# #     """
# #     Calculates the mean of specific columns over time from a list of DataFrames.

# #     Args:
# #     - dfs (List[DataFrame]): List of DataFrames to process.
# #     - seconds_interval (int): Interval in seconds between each DataFrame in the list.

# #     Returns:
# #     - DataFrame: A DataFrame containing the mean values of specific columns from each DataFrame,
# #       along with a corresponding time column in minutes.
# #     """

# #     processed_piv_files = os.path.join(data_path, condition, subcondition, "dataframes_PIV", "PIV_dataframe_*.csv")

# #     # Load all CSV files into a list of pandas dataframes
# #     dfs = [pd.read_csv(file) for file in sorted(glob.glob(processed_piv_files))]

# #     means_list = []
# #     # Iterate over each DataFrame, calculating mean for specific columns
# #     for df in dfs[min_frame:max_frame]:
# #         # Selecting specific columns and calculating the mean
# #         means = df.iloc[:, 4:8].join(df.iloc[:, 10:17]).mean(axis=0)
# #         means_list.append(means)

# #     # Concatenate all Series in the list into a single DataFrame
# #     result_df = pd.concat(means_list, axis=1).T

# #     # Reset index and convert index to time in minutes
# #     result_df = result_df.reset_index().rename(columns={'index': 'time [min]'})
# #     result_df['time [min]'] = result_df['time [min]'] * seconds_interval / 60

# #     return result_df






# # def process_piv_data(data_path, condition, subcondition, max_frame, feature_limits, volume=2, frame_rate=120, save_csv=True, heatmaps=True):
# #     """
# #     Processes PIV data, converts images, generates heatmaps for features, and creates heatmap movies.

# #     Args:
# #     - data_path (str): Base path where the PIV data files and images are stored.
# #     - condition (str): Specific condition defining a subdirectory within the data path.
# #     - subcondition (str): Specific subcondition defining a sub-subdirectory within the condition directory.
# #     - max_frame (int): Maximum number of frames/files to process.
# #     - feature_limits (dict): Dictionary specifying the limits for each feature.
# #     - volume (float, optional): Droplet volume in microliters µl. Defaults to 2.
# #     - frame_rate (int, optional): Frame rate for the output video. Defaults to 120.
# #     - save_csv (bool, optional): Whether to save the processed PIV files as CSV. Defaults to True.
# #     """

# #     if save_csv:
# #         # Process and save PIV files but avoid re-processing if already completed
# #         process_and_save_piv_files(data_path, condition, subcondition, max_frame=max_frame, volume=volume, save_csv=True)

# #     # Load all CSV files into a list of dataframes
# #     saved_processed_dfs = f"{data_path}{condition}/{subcondition}/dataframes_PIV/PIV_dataframe_****.csv"
# #     dfs = [pd.read_csv(file) for file in sorted(glob.glob(saved_processed_dfs))]

# #     if heatmaps:
# #         # Convert images
# #         convert_images(data_path, condition, subcondition, max_frame=max_frame)

# #         # Generate heatmaps for features
# #         generate_heatmaps_for_features(data_path, condition, subcondition, feature_limits, dfs)

# #         # Create heatmap movies
# #         create_heatmap_movies(data_path, condition, subcondition, feature_limits, max_frame=max_frame, frame_rate=frame_rate)

# #     return dfs



# # def plot_piv_features(data_paths, conditions, subconditions, seconds_intervals, sigma=1, min_frame=None, max_frame=None):
# #     """
# #     Performs plotting of mean features over time and Principal Component Analysis (PCA) for multiple 
# #     Particle Image Velocimetry (PIV) DataFrames.

# #     Args:
# #     - data_paths (List[str]): List of base directories for each DataFrame.
# #     - conditions (List[str]): List of condition labels for each DataFrame.
# #     - subconditions (List[str]): List of subcondition labels for each DataFrame.
# #     - seconds_intervals (List[int]): List of time intervals in seconds for each DataFrame.
# #     - sigma (float, optional): Standard deviation for the Gaussian filter, used in smoothing feature trends. Defaults to 1.
# #     - min_frame (int, optional): Minimum frame index to consider for calculation. If None, starts from the first frame. Defaults to None.
# #     - max_frame (int, optional): Maximum frame index to consider for calculation. If None, considers up to the last frame. Defaults to None.

# #     The function first calculates the mean values over time for each DataFrame and then plots these mean values.
# #     It then performs Principal Component Analysis (PCA) on each DataFrame and plots the first two principal components.
# #     The size and alpha of the PCA plot points are scaled to represent the progression of time or index.
# #     """

# #     # Calculating mean values over time for each DataFrame
# #     calculated_dfs = [calculate_mean_over_time(data_path, condition, subcondition, seconds_interval, min_frame, max_frame) 
# #                       for data_path, condition, subcondition, seconds_interval in zip(data_paths, conditions, subconditions, seconds_intervals)]

# #     # Plot Mean Over Time for Each Feature
# #     features = calculated_dfs[0].columns[1:]
# #     for feature in features:
# #         plt.figure(figsize=(10, 6))
# #         for df, data_path, condition, subcondition in zip(calculated_dfs, data_paths, conditions, subconditions):
# #             filtered_values = gaussian_filter(df[feature], sigma=sigma)
# #             plt.plot(df['time [min]'], filtered_values, label=f'{condition}_{subcondition}')
        
# #         plt.xlabel('Time [min]')
# #         plt.ylabel(feature)
# #         plt.title(f'Mean {feature} vs Time')
# #         plt.legend()
# #         plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# #         output_dir = os.path.join(data_path, condition, subcondition, "plots", "features_vs_time")
# #         if not os.path.exists(output_dir):
# #             os.makedirs(output_dir)

# #         output_file = f"{feature.split()[0]}.jpg"
# #         plt.savefig(os.path.join(output_dir, output_file), format='jpg', dpi=250)
# #         plt.close()

# #     # Perform PCA and Plot
# #     plt.figure(figsize=(10, 6))

# #     # Get colors from Seaborn's "colorblind" color palette
# #     sns.set_palette("colorblind", color_codes=True)
# #     colors = sns.color_palette("colorblind", n_colors=len(data_paths))


# #     for group_index, (df, data_path, condition, subcondition) in enumerate(zip(calculated_dfs, data_paths, conditions, subconditions)):
# #         pca = PCA(n_components=2)
# #         principalComponents = pca.fit_transform(df.iloc[:, 2:])
# #         principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# #         # Scaling alpha to decrease with respect to the frame index
# #         num_points = principalDf.shape[0]
# #         alphas = np.linspace(1, 0.01, num_points)  # Alpha values linearly spaced from 1 to 0.01

# #         # Use the color from Seaborn's palette for each group but vary the alpha
# #         for i in range(num_points):
# #             plt.scatter(principalDf['principal component 1'][i], principalDf['principal component 2'][i], 
# #                         label=f'{condition}_{subcondition}' if i == 0 else "", 
# #                         alpha=alphas[i], s=10, color=colors[group_index])

# #     plt.xlabel('Principal Component 1')
# #     plt.ylabel('Principal Component 2')
# #     plt.title('PCA of PIV Features (All Samples)')
# #     plt.legend()
# #     plt.grid(True)

# #     output_dir_pca = os.path.join(data_paths[-1], conditions[-1], subconditions[-1], "plots", "PCA.jpg")
# #     if not os.path.exists(os.path.dirname(output_dir_pca)):
# #         os.makedirs(os.path.dirname(output_dir_pca))
# #     plt.savefig(output_dir_pca, format='jpg', dpi=250)
# #     plt.close()