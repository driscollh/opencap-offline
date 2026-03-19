# OpenCap-Offline

A fully offline, localized implementation of Stanford NMBL's OpenCap. This version adapts the original cloud-based processing pipeline to run entirely on a local machine, ensuring strict data privacy and allowing for use in environments without internet access.
This version of the software requires a sufficiently powerful GPU, but cannot run on blackwell architecture cards (i.e. Nvidia 50-series cards). 
I recommend use with Nvidia 30-series or 40-series cards with minimum 12GB graphics ram.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/driscollh)

## Prerequisites
* Python 3.9.25
* Git

## Installation & Setup

**1. Download the Code**
Clone this repository to your local machine:
`git clone https://github.com/driscollh/opencap-offline.git`

**2. Install Python Packages**
Navigate into the main folder and install the required Python environment packages:
`pip install -r requirements.txt`

**3. Download the Local Dependencies (Required)**
Because the machine learning models and background engines (OpenPose, FFmpeg) are too large for GitHub, they are hosted securely on Zenodo.
* Go to the Zenodo archive: [https://doi.org/10.5281/zenodo.18995971](https://doi.org/10.5281/zenodo.18995971)
* Download the `dependencies.zip` file.
* Extract the contents directly into the empty `dependencies/` folder inside this project. This should result in three separate subfolders, one for each key dependency.

## Usage
Once the environment is set up and the dependencies are in place, you can launch the local processing by running the main Python script in your environment:

`python pyqt5_launcher_improved.py` 
or
`python simple_launcher.py`

## VideoRecording
Unlike the online version of OpenCap, this offline version requires additional video capture for calibration of camera intrinsics and extrinsics.
Please consult the `recording_practices.pdf` file prior to data collection.

## Acknowledgments and Licensing
This project is built upon the original open-source OpenCap project by the Stanford Neuromuscular Biomechanics Laboratory. 
* **Original OpenCap:** [https://github.com/opencap-org/opencap-core](https://github.com/opencap-org/opencap-core)
* **License:** Distributed under the Apache 2.0 License.
