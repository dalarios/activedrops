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


def plot_mean_fluorescence_over_time(data_path, conditions, subconditions, channel, time_interval=3, min_frame=0, max_frame=None, skip_frames=1, log_scale=False):
    """
    Computes and plots the mean fluorescence intensity over time for a given set of images across multiple conditions and subconditions,
    with visual grouping by condition and improved legend. Time is displayed in hours. The final plot, including all curves, is saved as a JPG file.

    Parameters:
    - data_path (str): Base path where the images are stored.
    - conditions (list of str): List of condition names.
    - subconditions (list of str): List of subcondition names.
    - channel (str): Channel name.
    - time_interval (int): Time interval between frames in minutes.
    - min_frame (int): Minimum frame number to process.
    - max_frame (int): Maximum frame number to process.
    - skip_frames (int): Number of frames to skip between plotted points.
    - log_scale (bool): Whether to plot the y-axis in log scale.
    """
    plt.figure(figsize=(12, 8))

    # Use a colormap to generate distinct colors for each condition
    cmap = plt.get_cmap('inferno')
    condition_colors = cmap(np.linspace(0, 1, len(conditions) + 1)[:-1])

    # Generate shades for subconditions within each condition
    for condition_idx, condition in enumerate(conditions):
        base_color = condition_colors[condition_idx]
        
        for sub_idx, subcondition in enumerate(subconditions):
            directory_path = os.path.join(data_path, condition, subcondition, "original")
            
            # files can be either img_00000****_cy5-4x_000 or img_00000****_gfp-4x_000.tif so load the appropriate channel
            if channel == "cy5":
                image_files = sorted(glob.glob(os.path.join(directory_path, "*cy5-4x_000.tif")))[min_frame:max_frame:skip_frames]
            elif channel == "gfp":
                image_files = sorted(glob.glob(os.path.join(directory_path, "*gfp-4x_000.tif")))[min_frame:max_frame:skip_frames]
            
            intensities = []
            frames = []
            
            for i, image_file in enumerate(image_files):
                img = imageio.imread(image_file) #/ 2**16  # Normalize to 16-bit
                mean_intensity = np.mean(img[750:1250, 750:1250]) 
                frames.append(i * skip_frames)  # Adjust frames slightly for visual separation
                intensities.append(mean_intensity)
            
            results_df = pd.DataFrame({
                "frame": frames,
                "mean_intensity": intensities - np.min(intensities)
            })
            
            # Apply Gaussian filter to mean_intensity for smoothing
            smoothed_intensities = gaussian_filter1d(results_df["mean_intensity"], sigma=1)  # Sigma controls the smoothing strength
            
            # Calculate shade for subcondition
            alpha = 0.3 + (sub_idx / len(subconditions)) * 0.7  # Adjust alpha for subcondition shading
            color = base_color * np.array([1, 1, 1, alpha])

            # Plot each condition and subcondition with smoothed data
            plt.plot(results_df["frame"] * time_interval / 60 / 60, smoothed_intensities, color=color, marker='o', linestyle='-', label=f"{condition} - {subcondition}")

    plt.title(f"Fluorescence expression over time - {channel}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Normalized Mean Fluorescence Intensity (A.U.)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    if log_scale:
        plt.yscale('log')  # Set y-axis scale to log scale

    # Determine the output path for saving the plot
    output_path = os.path.join(data_path, f"{channel}_mean_fluorescence_vs_time.jpg")
    plt.savefig(output_path, format='jpg', dpi=200)
    plt.show()  # Close the figure after saving to free resources


def fluorescence_heatmap(data_path, condition, subcondition, channel, time_interval=3, min_frame=0, max_frame=None, vmax=None, skip_frames=1):
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
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    # Get all .tif files in the folder
    image_files = sorted(glob.glob(os.path.join(input_directory_path, "*.tif")))[min_frame:max_frame:skip_frames] 
    
    # Loop through each image file and create a heatmap
    for i, image_file in enumerate(image_files, start=min_frame):
        # Read the image into a numpy array
        intensity_matrix = imageio.imread(image_file) 

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(intensity_matrix, cmap='gray', interpolation='nearest', extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Normalized Fluorescence Intensity (A.U.)')
        plt.title(f"Time (min): {(i - min_frame) * time_interval / 60:.2f}. \n{condition} - {subcondition} - {channel}")
        plt.xlabel('x [µm]')
        plt.ylabel('y [µm]')
        
        # Save the heatmap
        heatmap_filename = f"heatmap_frame_{i}.tif"
        heatmap_path = os.path.join(output_directory_path, heatmap_filename)
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)  # Close the figure to free memory
 
def create_movies(data_path, condition, subcondition, channel, frame_rate=30, max_frame=None):
    """
    Creates video files from processed and annotated images stored in a specified directory.

    Args:
    - data_path (str): Base path where the annotated images are stored.
    - condition (str): Condition under which the annotated images are stored.
    - subcondition (str): Subcondition under which the annotated images are stored.
    - channel (str): The specific channel being processed ('cy5' or 'gfp').
    - frame_rate (int, optional): Frame rate for the output video. Defaults to 30.
    - max_frame (int, optional): Maximum number of frames to be included in the video. If None, all frames are included.
    """


    images_dir = os.path.join(data_path, condition, subcondition, f"intensity_heatmap_{channel}")

    image_files = natsorted(glob.glob(os.path.join(images_dir, "*.tif")))

    if max_frame is not None:
        image_files = image_files[:max_frame]

    if not image_files:
        print("No images found for video creation.")
        return

    # Get the resolution of the first image (assuming all images are the same size)
    first_image = cv2.imread(image_files[0])
    video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_path = os.path.join(data_path, condition, subcondition, f"{condition}_{subcondition}-{channel}.avi")
    out = cv2.VideoWriter(out_path, fourcc, frame_rate, video_resolution)

    for file_path in image_files:
        img = cv2.imread(file_path)
        out.write(img)  # Write the image frame to the video

    out.release()
    print(f"Video saved to {out_path}")


def process_all_conditions_and_subconditions(data_path, conditions, subconditions, channel, time_interval, skip_frames, vmax, frame_rate, min_frame, max_frame):
    """
    Wrapper function to create heatmaps and movies for all combinations of conditions and subconditions.

    Args:
    - data_path (str): Base directory where the images are stored.
    - conditions (list of str): List of condition names.
    - subconditions (list of str): List of subcondition names.
    - channel (str): Channel specifying the fluorescence ('cy5' or 'gfp').
    - time_interval (int): Time interval in seconds between frames for heatmap.
    - skip_frames (int): Number of frames to skip between each processed frame for heatmap.
    - vmax (int): Maximum value for normalization in the heatmap.
    - frame_rate (int): Frame rate for the output video.
    """
    for condition in conditions:
        for subcondition in subconditions:
            # Create heatmaps for each condition and subcondition
            fluorescence_heatmap(
                data_path=data_path,
                condition=condition,
                subcondition=subcondition,
                channel=channel,
                time_interval=time_interval * skip_frames / 2, # this /2 is to consider that we are always at least be skipping one frame 
                min_frame=min_frame,
                max_frame=max_frame,
                vmax=vmax,
                skip_frames=skip_frames
            )
            
            # Create annotated image movies for each condition and subcondition
            create_movies(
                data_path=data_path,
                condition=condition,
                subcondition=subcondition,
                channel=channel,
                frame_rate=frame_rate,
                max_frame=max_frame
            )
