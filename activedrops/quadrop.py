# Import standard libraries
import os
import re
import sys
import glob
import shutil


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
import imageio.v2 as imageio

# Import plotting utilities
import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
import imageio.v2 as imageio
import seaborn as sns
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d


"""
Name: 
    Quantitative Active Drops Phenotyping (QuADroP)
Title:
    quadrop.py
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
    sns.set_palette("viridis", color_codes=True)
    sns.set_context('notebook', rc=rc)




######################################### raw data processing #########################################
    
# optional
def reorgTiffsToOriginal(data_path, conditions, subconditions):
    """
    Args:
        data_path (_type_): _description_
        conditions (_type_): _description_
        subconditions (_type_): _description_
        
        
    Activate when you have your subconditions inside the conditions folder. 
    This function renames the subconditions as PosX and moves the raw data do "original" folder.
    """
    
    
    for condition in conditions:
        # Get the actual subconditions in the directory
        actual_subconditions = [name for name in os.listdir(os.path.join(data_path, condition)) if os.path.isdir(os.path.join(data_path, condition, name))]
        
        # Rename the actual subconditions to match the subconditions in your list
        for i, actual_subcondition in enumerate(sorted(actual_subconditions)):
            os.rename(os.path.join(data_path, condition, actual_subcondition), os.path.join(data_path, condition, subconditions[i]))
        
        for subcondition in subconditions:
            # Construct the path to the subcondition directory
            subcondition_path = os.path.join(data_path, condition, subcondition)
            
            # Create the path for the "original" directory within the subcondition directory
            original_dir_path = os.path.join(subcondition_path, "original")
            
            # Always create the "original" directory
            os.makedirs(original_dir_path, exist_ok=True)
            
            # Iterate over all files in the subcondition directory
            for filename in os.listdir(subcondition_path):
                # Check if the file is a .tif file
                if filename.endswith(".tif"):
                    # Construct the full path to the file
                    file_path = os.path.join(subcondition_path, filename)
                    
                    # Construct the path to move the file to
                    destination_path = os.path.join(original_dir_path, filename)
                    
                    # Move the file to the "original" directory
                    shutil.move(file_path, destination_path)
            print(f"Moved .tif files from {subcondition_path} to {original_dir_path}")


# reorgTiffsToOriginal(data_path, conditions, subconditions)



def plot_fluorescence(data_path, conditions, subconditions, channel, time_intervals, min_frame, max_frame,
                      skip_frames=1, line_slope=1, line_intercept=0, log_scale=False, timescale="h", averaged=False, 
                      legend_loc='upper right', subtract_first_datapoint=True):
    """
    Computes and plots the fluorescence intensity over time for a given set of images across multiple conditions and subconditions.
    Can also average the subconditions within each condition if 'averaged' is True, after converting A.U. to µg/ml.
    Time is displayed based on the specified timescale and time interval for each condition.
    The final plot, including all curves, is saved as a JPG file.

    Parameters:
    - data_path (str): Base path where the images are stored.
    - conditions (list of str): List of condition names.
    - subconditions (list of str): List of subcondition names.
    - channel (str): Channel name.
    - time_intervals (list of int): List of time intervals between frames in seconds, one for each condition.
    - min_frame (int): Minimum frame number to process.
    - max_frame (int): Maximum frame number to process.
    - skip_frames (int): Number of frames to skip between plotted points.
    - log_scale (bool): Whether to plot the y-axis in log scale.
    - timescale (str): The unit of time to display on the x-axis ('h' for hours, 'min' for minutes).
    - averaged (bool): Whether to average the fluorescence intensity across subconditions.
    """
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('inferno')
    condition_colors = cmap(np.linspace(0, 1, len(conditions) + 1)[:-1])

    # Line equation to convert A.U. to concentration
    line_slope = line_slope
    line_intercept = line_intercept

    # Define conversion factor based on the timescale
    if timescale == "h":
        time_conversion_factor = 1 / 3600  # Convert seconds to hours
        x_label = "Time (hours)"
    elif timescale == "min":
        time_conversion_factor = 1 / 60  # Convert seconds to minutes
        x_label = "Time (minutes)"
    else:
        raise ValueError("Invalid timescale. Choose either 'h' for hours or 'min' for minutes.")

    if len(time_intervals) != len(conditions):
        raise ValueError("The number of time intervals must match the number of conditions.")

    for condition_idx, (condition, time_interval) in enumerate(zip(conditions, time_intervals)):
        all_concentrations = []  # List to hold all concentrations for averaging

        for sub_idx, subcondition in enumerate(subconditions):
            directory_path = os.path.join(data_path, condition, subcondition, "original")
            current_time_interval = time_interval * time_conversion_factor  # Convert time_interval to the correct timescale

            if channel == "cy5":
                image_files = sorted(glob.glob(os.path.join(directory_path, "*cy5*.tif")))[min_frame:max_frame:skip_frames]
            elif channel == "gfp":
                image_files = sorted(glob.glob(os.path.join(directory_path, "*gfp*.tif")))[min_frame:max_frame:skip_frames]

            intensities = []
            for i, image_file in enumerate(image_files):
                img = imageio.imread(image_file) / 2**16  # Normalize to 16-bit
 
                # Select ROI
                mean_intensity = np.mean(img[750:1250, 750:1250])

                # Subtract the minimum intensity to remove background noise
                mean_intensity = mean_intensity - np.min(img)


                intensities.append(mean_intensity)

            # Convert A.U. to concentration using the line equation
            concentrations = [(intensity - line_intercept) / line_slope for intensity in intensities]
            
            # apply gaussian filter to smooth the curve
            concentrations = gaussian_filter1d(concentrations, sigma=2)

            # make sure first value is always 0 by subtracting the first value from all values
            if subtract_first_datapoint == True:
                concentrations = np.array(concentrations) - concentrations[0]

            if averaged:
                all_concentrations.append(concentrations)
            else:
                frames = np.array([i * skip_frames * current_time_interval for i in range(len(concentrations))])

                alpha = 0.3 + (sub_idx / len(subconditions)) * 0.7
                color = condition_colors[condition_idx] * np.array([1, 1, 1, alpha])
                plt.plot(frames, concentrations, color=color, marker='o', linestyle='-', label=f"{condition} - {subcondition}")

        if averaged:
            avg_concentrations = np.mean(all_concentrations, axis=0)
            frames = np.array([i * skip_frames * current_time_interval for i in range(len(avg_concentrations))])
            plt.plot(frames, avg_concentrations, color=condition_colors[condition_idx], marker='o', linestyle='-', label=condition)

    # plt.title(f"Fluorescence expression over time - {channel}")
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Protein Concentration (ng/μl)", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc=legend_loc, fontsize=16)

    # Increase the size of x and y ticks
    plt.tick_params(axis='both', which='major', labelsize=10)

    if log_scale:
        plt.yscale('log')

    # else:
    #     plt.ylim(bottom=-0.5)

    output_path = os.path.join(data_path, f"{channel}_{'averaged' if averaged else 'mean'}_fluorescence_vs_time.jpg")
    plt.savefig(output_path, format='jpg', dpi=200)
    plt.show()




# plot the raw image as heatmap of fluorescence intensity
def fluorescence_heatmap(data_path, condition, subcondition, channel, time_interval, min_frame, max_frame, vmax, skip_frames=1, line_slope=1, line_intercept=0):
    """
    Reads each image as a matrix, creates and saves a heatmap representing the normalized pixel-wise fluorescence intensity.

    Args:
    - data_path (str): Base directory where the images are stored.
    - condition (str): Condition defining a subdirectory within the data path.
    - subcondition (str): Subcondition defining a further subdirectory.
    - channel (str): Channel specifying the fluorescence ('cy5' or 'gfp').
    - time_interval (int): Time interval in seconds between frames.
    - min_frame (int): Minimum frame number to start processing from.
    - max_frame (int, optional): Maximum frame number to stop processing at.
    """
    # Determine the directory paths based on the channel
    input_directory_path = os.path.join(data_path, condition, subcondition, "original")
    output_directory_path = os.path.join(data_path, condition, subcondition, f"intensity_heatmap_{channel}")
    
    # Create the output directory if it doesn't exist but also deletes the directory if it already exists to start fresh
    if os.path.exists(output_directory_path):
        shutil.rmtree(output_directory_path)
    os.makedirs(output_directory_path, exist_ok=True)


    # Get all .tif files in the folder
    image_files = sorted(glob.glob(os.path.join(input_directory_path, "*.tif")))[min_frame:max_frame:skip_frames] 

    if channel == "cy5":
        image_files = sorted(glob.glob(os.path.join(input_directory_path, "*cy5*.tif")))[min_frame:max_frame:skip_frames]
    elif channel == "gfp":
        image_files = sorted(glob.glob(os.path.join(input_directory_path, "*gfp*.tif")))[min_frame:max_frame:skip_frames]
            
    # # Calibration curve parameters
    # line_slope = 0.0004203353275461814
    # line_intercept = 0.0015873751623883166
    
    # Loop through each image file and create a heatmap
    for i, image_file in enumerate(image_files, start=min_frame):
        # Read the image into a numpy array
        intensity_matrix = imageio.imread(image_file) / 2**16  # Normalize the 16-bit image to 1.0

        # Convert intensity values to protein concentration using the calibration curve
        concentration_matrix = (intensity_matrix - line_intercept) / line_slope

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(concentration_matrix, cmap='gray', interpolation='nearest', extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=0, vmax=vmax)
        
        plt.colorbar(im, ax=ax, label='Protein concentration (ng/µl)')
        plt.title(f"Time (min): {(i - min_frame) * time_interval * skip_frames / 60:.2f} \nTime (hours): {(i - min_frame) * time_interval * skip_frames / 3600:.2f} \n{condition} - {subcondition} - {channel}", fontsize=14)
        plt.xlabel('x [µm]')
        plt.ylabel('y [µm]')
        
        # Save the heatmap
        heatmap_filename = f"heatmap_frame_{i}.tif"
        heatmap_path = os.path.join(output_directory_path, heatmap_filename)
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)


# create a movie from the processed images -- general function
def create_movies(data_path, condition, subcondition, channel, movie_type, frame_rate, feature_limits=None, max_frame=None):
    """
    Creates video files from processed and annotated images stored in a specified directory.

    Args:
    - data_path (str): Base path where the annotated images are stored.
    - condition (str): Condition under which the annotated images are stored.
    - subcondition (str): Subcondition under which the annotated images are stored.
    - channel (str): The specific channel being processed ('cy5' or 'gfp').
    - movie_type (str): Type of movie to create ('single', 'grid', or 'PIV').
    - feature_limits (dict, optional): Dictionary specifying the limits for each feature (only for 'PIV' movie type).
    - frame_rate (int, optional): Frame rate for the output video. Defaults to 120.
    - max_frame (int, optional): Maximum number of frames to be included in the video. If None, all frames are included.
    """

    if movie_type == 'single':
        output_dir = os.path.join(data_path, f"single_movies_{channel}")
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(data_path, condition, subcondition, f"intensity_heatmap_{channel}")
        image_files = natsorted(glob.glob(os.path.join(images_dir, "*.tif")))
        out_path = os.path.join(output_dir, f"{condition}_{subcondition}-{channel}.avi")
    elif movie_type == 'grid':
        images_dir = os.path.join(data_path, f"grid_heatmaps_{channel}")
        image_files = natsorted(glob.glob(os.path.join(images_dir, "*.png")))
        out_path = os.path.join(data_path, f"grid-{channel}.avi")
    elif movie_type == 'PIV':
        plots_dir = f"{data_path}/{condition}/{subcondition}/heatmaps_PIV/"
        for feature in feature_limits.keys():
            feature_name_for_file = feature.split()[0]
            heatmap_dir = os.path.join(data_path, condition, subcondition, "heatmaps_PIV", f"{feature.split()[0]}", f"{feature.split()[0]}_heatmap_****.jpg")
            image_files = natsorted(glob.glob(heatmap_dir))

            if not image_files:
                continue

            # Limit the number of files if max_frame is specified
            image_files = image_files[:max_frame] if max_frame is not None else image_files

            # Get the resolution of the first image (assuming all images are the same size)
            first_image = cv2.imread(image_files[0])
            video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(f'{plots_dir}{feature_name_for_file}.avi', fourcc, frame_rate, video_resolution)

            for file in image_files:
                img = cv2.imread(file)
                out.write(img)  # Write the image as is, without resizing

            out.release()
        return

    if not image_files:
        print("No images found for video creation.")
        return

    # Limit the number of files if max_frame is specified
    image_files = image_files[:max_frame] if max_frame is not None else image_files

    # Get the resolution of the first image (assuming all images are the same size)
    first_image = cv2.imread(image_files[0])
    video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_path, fourcc, frame_rate, video_resolution)

    for file_path in image_files:
        img = cv2.imread(file_path)
        out.write(img)  # Write the image frame to the video

    out.release()
    print(f"Video saved to {out_path}")

# generate movies from individual fluorescence intensity heatmaps
def single_fluorescence_movies(data_path, conditions, subconditions, channel, time_intervals, min_frame, max_frame, vmax, skip_frames, frame_rate, line_slope=1, line_intercept=0):
    for i, condition in enumerate(conditions):
        time_interval = time_intervals[i]
        for subcondition in subconditions:
            # Create heatmaps for each condition and subcondition
            fluorescence_heatmap(
                data_path=data_path,
                condition=condition,
                subcondition=subcondition,
                channel=channel,
                time_interval=time_interval,
                min_frame=min_frame,
                max_frame=max_frame,
                vmax=vmax,
                skip_frames=skip_frames,
                line_intercept=line_intercept,
                line_slope=line_slope
            )

            # Create annotated image movies for each condition and subcondition
            create_movies(
                data_path=data_path,
                condition=condition,
                subcondition=subcondition,
                channel=channel,
                movie_type='single',
                frame_rate=frame_rate,
                max_frame=max_frame
            )

# combine all the movies into a single movie (heatmaps in a grid)
def grid_heatmaps(data_path, conditions, subconditions, channel, frame_rate, figsize):
    output_dir = os.path.join(data_path, f"grid_heatmaps_{channel}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all image paths for the specified channel across all conditions and subconditions
    all_image_paths = []
    for condition in conditions:
        for subcondition in subconditions:
            image_dir = os.path.join(data_path, condition, subcondition, f"intensity_heatmap_{channel}")
            image_paths = natsorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
            all_image_paths.extend(image_paths)
    
    # Number of frames is based on the unique frames across all conditions and subconditions
    num_frames = len(set(os.path.basename(path) for path in all_image_paths))
    
    # Iterate over the number of unique frames
    for frame_idx in range(0, num_frames):
        fig, ax = plt.subplots(len(subconditions), len(conditions), figsize=figsize)
        
        for i, subcondition in enumerate(subconditions):
            for j, condition in enumerate(conditions):
                # Construct the path for the current heatmap for each condition and subcondition
                image_path = os.path.join(data_path, condition, subcondition, f"intensity_heatmap_{channel}", f"heatmap_frame_{frame_idx}.tif")
                
                # Ensure the file exists before attempting to read it
                if os.path.exists(image_path):
                    im = imageio.imread(image_path)
                    ax[i, j].imshow(im, cmap='gray')
                    ax[i, j].axis('off')
                else:
                    # Handle missing files (optional) by clearing or placing a placeholder
                    ax[i, j].axis('off')
        
        plt.subplots_adjust(wspace=0, hspace=0)
        output_path = os.path.join(output_dir, f"heatmap_grid_frame_{frame_idx}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=250)
        plt.close(fig)

    create_movies(data_path, condition=None, subcondition=None, channel=channel, movie_type='grid', frame_rate=frame_rate)



######################################### PIV #########################################

# convert pivlab images to the right size 
def convert_images(data_path, conditions, subconditions, max_frame, brightness_factor=1, contrast_factor=1, skip_frames=1):
    """
    Converts, resizes, and adjusts the brightness and contrast of images for multiple conditions and 
    subconditions, then saves the processed images in new directories.

    Args:
    - data_path (str): Base directory where the original PIV movie images are stored.
    - conditions (list of str): List of conditions defining subdirectories within the data path.
    - subconditions (list of str): List of subconditions defining sub-subdirectories within each condition directory.
    - max_frame (int, optional): Maximum number of images to process. If None, all images in the directory are processed.
    - brightness_factor (float, optional): Factor to adjust the brightness of the images. Defaults to 1 (no change).
    - contrast_factor (float, optional): Factor to adjust the contrast of the images. Defaults to 1 (no change).
    - skip_frames (int, optional): Number of frames to skip between processing. Defaults to 1 (no skipping).
    """
    for condition in conditions:
        for subcondition in subconditions:
            input_dir = os.path.join(data_path, condition, subcondition, "piv_movie")
            output_dir = os.path.join(data_path, condition, subcondition, "piv_movie_converted")

            os.makedirs(output_dir, exist_ok=True)

            input_files = natsorted(glob.glob(os.path.join(input_dir, '*.jpg')))

            if max_frame:
                input_files = input_files[:max_frame]

            # Apply frame skipping
            input_files = input_files[::skip_frames]

            output_files = natsorted(glob.glob(os.path.join(output_dir, '*.tif')))
            if len(input_files) <= len(output_files):
                print(f"Conversion might already be completed or partial for {output_dir}. Continuing...")
                # Optional: Add logic to check and continue incomplete work.

            num_digits = len(str(len(input_files)))

            for i, file_name in enumerate(input_files):
                image = Image.open(file_name).convert("L")
                image_resized = image.resize((2048, 2048), Image.LANCZOS)

                enhancer = ImageEnhance.Brightness(image_resized)
                image_brightened = enhancer.enhance(brightness_factor)
                enhancer = ImageEnhance.Contrast(image_brightened)
                image_contrasted = enhancer.enhance(contrast_factor)

                padded_index = str(i + 1).zfill(num_digits)
                base_file_name = f'converted_image_{padded_index}.tif'
                processed_image_path = os.path.join(output_dir, base_file_name)
                image_contrasted.save(processed_image_path, format='TIFF', compression='tiff_lzw')

# helper function to plot autocorrelation
def plot_autocorrelation_values(data_path, condition, subcondition, frame_id, lambda_tau, results, fitted_values, intervector_distance_microns):
    output_directory_dfs = os.path.join(data_path, condition, subcondition, "autocorrelation_plots")
    os.makedirs(output_directory_dfs, exist_ok=True)

    plt.figure(figsize=(10, 6))

    x_values = np.arange(len(results)) * intervector_distance_microns * 1E6

    plt.plot(x_values, results, label='Autocorrelation Values', marker='o', linestyle='-', markersize=5)
    plt.plot(x_values, fitted_values, label='Fitted Exponential Decay', linestyle='--', color='red')
    plt.axvline(x=lambda_tau, color='green', linestyle='-.', label=f'Correlation Length = {lambda_tau:.2f} µm')

    plt.xlabel('Scaled Lag (µm)')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation Function and Fitted Exponential Decay (Frame {frame_id})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.ylim(0, 1.1)

    plt.tight_layout()

    filename = os.path.join(output_directory_dfs, f'autocorrelation_frame_{frame_id}.jpg')
    plt.savefig(filename, dpi=200, format='jpg')
    plt.close()

# helper function to calculate correlation length
def correlation_length(data_frame):
    # Reshaping the data frame to a 2D grid and normalizing
    v = data_frame.pivot(index='y [m]', columns='x [m]', values="velocity magnitude [m/s]").values
    v -= np.mean(v)  # Centering the data

    # FFT to find the power spectrum and compute the autocorrelation
    fft_v = np.fft.fft2(v)
    autocorr = np.fft.ifft2(fft_v * np.conj(fft_v))
    autocorr = np.real(autocorr) / np.max(np.real(autocorr))  # Normalize the autocorrelation

    # Preparing to extract the autocorrelation values along the diagonal
    r_values = min(v.shape) // 2
    results = np.zeros(r_values)
    for r in range(r_values):
        # Properly average over symmetric pairs around the center
        autocorrelation_value = (autocorr[r, r] + autocorr[-r, -r]) / 2
        results[r] = autocorrelation_value

    # Normalize the results to start from 1
    results /= results[0]

    # Exponential decay fitting to extract the correlation length
    def exponential_decay(x, A, B, C):
        return A * np.exp(-x / B) + C

    # Fit parameters and handling potential issues with initial parameter guesses
    try:
        params, _ = curve_fit(exponential_decay, np.arange(len(results)), results, p0=(1, 10, 0), maxfev=5000)
    except RuntimeError:
        # Handle cases where the curve fit does not converge
        params = [np.nan, np.nan, np.nan]  # Use NaN to indicate the fit failed

    A, B, C = params
    fitted_values = exponential_decay(np.arange(r_values), *params)

    # Calculate the correlation length
    intervector_distance_microns = ((data_frame["y [m]"].max() - data_frame["y [m]"].min()) / v.shape[0])
    if B > 0 and A != C:  # Ensure valid values for logarithmic calculation
        lambda_tau = -B * np.log((0.3 - C) / A) * intervector_distance_microns
    else:
        lambda_tau = np.nan  # Return NaN if parameters are not suitable for calculation

    return lambda_tau, results, fitted_values, intervector_distance_microns


# load PIV data from PIVlab into dataframes
def load_piv_data(data_path, condition, subcondition, min_frame=0, max_frame=None, skip_frames=1):
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
    for file in sorted(glob.glob(input_piv_data))[min_frame:max_frame:skip_frames]:
        df = pd.read_csv(file, skiprows=2).fillna(0).rename(columns={
            "magnitude [m/s]": "velocity magnitude [m/s]",
            "simple shear [1/s]": "shear [1/s]",
            "simple strain [1/s]": "strain [1/s]",
            "Vector type [-]": "data type [-]"
        })
        dfs.append(df)

    return dfs

# store pivlab output as dataframes
def generate_dataframes_from_piv_data(data_path, condition, subcondition, min_frame=0, max_frame=None, skip_frames=1, plot_autocorrelation=True):
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

    # Load PIV data
    data_frames = load_piv_data(data_path, condition, subcondition, min_frame, max_frame, skip_frames)


    # Calculating mean values with valid vectors only
    mean_values = []
    for frame_id, data_frame in enumerate(data_frames):
        lambda_tau, results, fitted_values, intervector_distance_microns = correlation_length(data_frame)
        if plot_autocorrelation:
            plot_autocorrelation_values(data_path, condition, subcondition, frame_id, lambda_tau * 1E6, results, fitted_values, intervector_distance_microns)
        data_frame["correlation length [m]"] = lambda_tau
        data_frame = data_frame[data_frame["data type [-]"] == 1]
        mean_values.append(data_frame.mean(axis=0))

    # Creating mean DataFrame
    mean_data_frame = pd.DataFrame(mean_values)
    mean_data_frame.reset_index(drop=False, inplace=True)
    mean_data_frame.rename(columns={'index': 'frame'}, inplace=True)

    # Calculate power and add to DataFrame
    volume = 2.5E-9  # µl --> m^3
    viscosity = 1E-3  # mPa*S
    mean_data_frame["power [W]"] = volume * viscosity * (mean_data_frame["velocity magnitude [m/s]"]/mean_data_frame["correlation length [m]"])**2

    # Renaming time column
    # mean_data_frame.rename(columns={'frame': 'time [min]'}, inplace=True)

    # Remove unnecessary columns for the pivot matrices
    # mean_data_frame = mean_data_frame.iloc[:, 5:]

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

    # return mean_data_frame, pivot_data_frame, average_values
    return mean_data_frame, pivot_data_frame

# plot the pivlab output as heatmaps
def generate_heatmaps_from_dataframes(df, data_path, condition, subcondition, feature_limits, time_interval=3):
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
            im = plt.imshow(vals, cmap='inferno', origin='upper', alpha=0.7, extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax) # heatmap
            plt.xlabel('x [um]')
            plt.ylabel('y [um]')
            cbar = plt.colorbar(im)
            cbar.set_label(feature)
            time = df.iloc[j, -1]
            plt.title(f"PIV - {feature}  ||  time: {int(time * time_interval/60)} min")

            os.makedirs(os.path.dirname(output_directory_heatmaps), exist_ok=True)
            plt.savefig(output_directory_heatmaps, format='jpg', dpi=250)
            plt.close()

# turn heatmaps into movies 
def process_piv_data(data_path, conditions, subconditions, feature_limits, time_intervals, min_frame=0, max_frame=None, skip_frames=1, plot_autocorrelation=True, frame_rate=120, heatmaps=True):
    """Process PIV data for all conditions and subconditions, then average and save results.

    Args:
        data_path (str): Base directory for PIV data and output.
        conditions (list): List of conditions.
        subconditions (list): List of subconditions.
        feature_limits (dict): Dictionary of feature limits.
        time_intervals (list): List of time intervals matching the conditions.
        min_frame (int, optional): Minimum frame number to process. Defaults to 0.
        max_frame (int, optional): Maximum frame number to process. Defaults to None.
        skip_frames (int, optional): Number of frames to skip between processed frames. Defaults to 1.
        plot_autocorrelation (bool, optional): Whether to plot autocorrelation. Defaults to True.
        frame_rate (int, optional): Frame rate for the movies. Defaults to 120.
    """
    for i, condition in enumerate(conditions):
        time_interval = time_intervals[i] * skip_frames
        results = []
        for subcondition in subconditions:
            m, p = generate_dataframes_from_piv_data(data_path, condition, subcondition, min_frame, max_frame, skip_frames, plot_autocorrelation)
            results.append(m)

            if heatmaps == True:
                generate_heatmaps_from_dataframes(p, data_path, condition, subcondition, feature_limits, time_interval)
                create_movies(data_path, condition, subcondition, channel=None, movie_type='PIV', feature_limits=feature_limits, frame_rate=frame_rate, max_frame=max_frame)

        # Averaging and saving the results for the current condition
        save_path = os.path.join(data_path, condition, 'averaged')
        average_df = sum(results) / len(results)
        
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        average_df.to_csv(os.path.join(save_path, f"{condition}_average.csv"))

# generate PCA from pivlab output
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
        alphas = np.linspace(0.001, 1, num_points)  # Alpha values linearly spaced from 1 to 0.01
        
        # Plotting each line segment with increasing alpha
        for i in range(1, num_points):
            plt.plot(principalDf['principal component 1'][i-1:i+1], principalDf['principal component 2'][i-1:i+1], 
                     alpha=alphas[i], linestyle='-', linewidth=2, color=colors[group_index])

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

def plot_features(data_paths, conditions, subconditions, features, time_intervals, sigma=2, min_frame=None, max_frame=None):
    dfs = []

    for data_path, condition, subcondition, time_interval in zip(data_paths, conditions, subconditions, time_intervals):
        file_path = os.path.join(data_path, condition, subcondition, "dataframes_PIV", "mean_values.csv")
        df = pd.read_csv(file_path)

        df.iloc[:, :] = df.iloc[:, :].apply(lambda x: gaussian_filter(x, sigma=sigma))

        df = df.rename(columns={"frame": "time [min]"})
        df["time [min]"] = (df["time [min]"] - df["time [min]"].min()) * time_interval / 60

        df = df.rename(columns={"data type [-]_mean": "work [J]", "correlation length [m]_mean": "correlation length [um]", "velocity magnitude [m/s]_mean": "velocity magnitude [um/s]"})
        df["work [J]"] = df["power [W]_mean"].cumsum()
        df["correlation length [um]"] = df["correlation length [um]"] * 1e6
        df["velocity magnitude [um/s]"] = df["velocity magnitude [um/s]"] * 1e6

        # make "power [W]_mean" the first column
        cols = list(df.columns)
        # cols = [cols[-1]] + cols[:-1]
        df = df[cols]

        df = df.iloc[min_frame:max_frame, :]

        dfs.append(df)

    plot_pca(dfs, data_paths, conditions, subconditions, features)

    for feature in dfs[0].columns[:]:
        plt.figure(figsize=(10, 6))

        for df, data_path, condition, subcondition, time_interval in zip(dfs, data_paths, conditions, subconditions, time_intervals):
            output_directory_plots = os.path.join(data_path, condition, subcondition, "plots_PIV", f"{feature.split()[0]}_plot.jpg")
            os.makedirs(os.path.dirname(output_directory_plots), exist_ok=True)
            plt.plot(df["time [min]"], df[feature], marker='o', linestyle='-', markersize=1, linewidth=1, label=f'{condition}_{subcondition}')

        plt.xlabel('Time (minutes)')
        plt.ylabel(feature)
        plt.title(f"PIV - {feature}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(output_directory_plots, format='jpg', dpi=300)
        plt.close()

# plot features and PCA averaged over subconditions
def plot_features_averages(data_paths, conditions, subconditions, features, time_intervals, sigma=2, min_frame=None, max_frame=None):
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
    
    dfs = []

    for data_path, condition, subcondition, time_interval in zip(data_paths, conditions, subconditions, time_intervals):
        file_path = os.path.join(data_path, condition, subcondition, f"{condition}_average.csv")
        df = pd.read_csv(file_path)

        df.iloc[:, :] = df.iloc[:, :].apply(lambda x: gaussian_filter(x, sigma=sigma))

        df = df.rename(columns={"frame": "time [min]"})
        df["time [min]"] = (df["time [min]"] - df["time [min]"].min()) * time_interval / 60

        df = df.rename(columns={"data type [-]_mean": "work [J]", "correlation length [m]_mean": "correlation length [um]", "velocity magnitude [m/s]_mean": "velocity magnitude [um/s]"})
        df["work [J]"] = df["power [W]_mean"].cumsum()
        df["correlation length [um]"] = df["correlation length [um]"] * 1e6
        df["velocity magnitude [um/s]"] = df["velocity magnitude [um/s]"] * 1e6

        df = df.iloc[min_frame:max_frame, :]

        dfs.append(df)

    plot_pca(dfs, data_paths, conditions, subconditions, features)

    for feature in dfs[0].columns[:]:
        plt.figure(figsize=(10, 6))

        for df, data_path, condition, subcondition, time_interval in zip(dfs, data_paths, conditions, subconditions, time_intervals):
            output_directory_plots = os.path.join(data_path, condition, subcondition, "plots_PIV", f"{feature.split()[0]}_plot.jpg")
            os.makedirs(os.path.dirname(output_directory_plots), exist_ok=True)
            plt.plot(df["time [min]"], df[feature], marker='o', linestyle='-', markersize=1, linewidth=1, label=f'{condition}_{subcondition}')

        plt.xlabel('Time (minutes)')
        plt.ylabel(feature)
        plt.title(f"PIV - {feature}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(output_directory_plots, format='jpg', dpi=300)
        plt.close()


def plot_PIV_features(data_path, conditions, subconditions, features_pca, time_intervals, sigma=10, min_frame=0, max_frame=None, averages=True):
    # Plot features for individual subconditions
    for condition in conditions:
        data_paths = [data_path] * len(subconditions)
        condition_list = [condition] * len(subconditions)
        plot_features(
            data_paths,
            condition_list,
            subconditions,
            features_pca,
            time_intervals=[time_intervals[conditions.index(condition)]] * len(subconditions),
            sigma=sigma,
            min_frame=min_frame,
            max_frame=max_frame,
        )

    # Plot features for all subconditions together
    data_paths = [data_path] * len(conditions) * len(subconditions)
    condition_list = [condition for condition in conditions for _ in range(len(subconditions))]
    subcondition_list = subconditions * len(conditions)
    time_interval_list = [time_interval for time_interval in time_intervals for _ in range(len(subconditions))]

    plot_features(
        data_paths,
        condition_list,
        subcondition_list,
        features_pca,
        time_intervals=time_interval_list,
        sigma=sigma,
        min_frame=min_frame,
        max_frame=max_frame,
    )

    if averages == True:
        # Plot features for averaged subconditions
        data_paths = [data_path] * len(conditions)
        subcondition_list = ['averaged'] * len(conditions)
        time_interval_list = time_intervals

        plot_features_averages(
            data_paths,
            conditions,
            subcondition_list,
            features_pca,
            time_intervals=time_interval_list,
            sigma=sigma,
            min_frame=min_frame,
            max_frame=max_frame,
        )