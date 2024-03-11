# QuADroP: Quantitative Active Drops Phenotyping

## Overview
QuADroP (Quantitative Active Drops Phenotyping) is a Python-based tool designed for the detailed analysis of experimental data obtained from ActiveDROPS experiments. It encompasses a suite of functions for processing raw Particle Image Velocimetry (PIV) data, fluorescence imaging, and other quantitative analyses crucial for understanding the dynamics and properties of active droplets.

## How to Use
To adapt QuADroP to your research, start by cloning the repository to your local machine. Update the environment and install necessary dependencies as per the provided requirements file.

```
git clone https://github.com/your-repository/QuADroP.git
cd QuADroP
pip install -r requirements.txt
```

## Structure
The codebase is structured to facilitate easy access and manipulation of data:

- **`data_processing`**: For raw data transformation into analyzable formats.
- **`analysis`**: Functions to draw insights from processed data.
- **`visualization`**: Tools for generating plots and heatmaps to visualize the data.
- **`utilities`**: Helper functions for data manipulation and analysis.

## Features
- **Fluorescence Analysis**: Quantify and plot fluorescence intensity over time.
- **Heatmap Generation**: Visualize spatial distribution of fluorescence.
- **PIV Data Analysis**: Extract and analyze vector fields from PIV data.
- **Autocorrelation and PCA**: Tools for understanding dynamics and correlations in active droplet motion.

## Required Files
Ensure the following files are included in your project for complete functionality:
- **`requirements.txt`**: List of Python packages required.
- **`README.md`**: Guide on using QuADroP, including setup, usage instructions, and feature descriptions.

## License
QuADroP is open source, made available under the MIT License, allowing for wide distribution and modification with appropriate credit to the original authors.

To contribute or for more information, please visit the [QuADroP GitHub repository](https://github.com/your-repository/QuADroP).
# License Information

<p xmlns:dct="http://purl.org/dc/terms/" xmlns:vcard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <a rel="license"
     href="http://creativecommons.org/publicdomain/zero/1.0/">
    <img src="http://i.creativecommons.org/p/zero/1.0/88x31.png" style="border-style: none;" alt="CC0" />
  </a>
  <br />
  To the extent possible under law,
  <a rel="dct:publisher"
     href="github.com/gchure/reproducible_research">
    <span property="dct:title">Griffin Chure</span></a>
  has waived all copyright and related or neighboring rights to
  <span property="dct:title">A template for using git as a platform for reproducible scientific research</span>.
This work is published from:
<span property="vcard:Country" datatype="dct:ISO3166"
      content="US" about="github.com/gchure/reproducible_research">
  United States</span>.
</p>