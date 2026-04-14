# OpenCap-Offline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19562446.svg)](https://doi.org/10.5281/zenodo.19562446)  
  
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/driscollh)

A fully offline, localized implementation of Stanford NMBL's OpenCap. This version adapts the original cloud-based processing pipeline to run entirely on a local machine, ensuring strict data privacy and allowing for use in environments without internet access.
This version of the software requires a sufficiently powerful GPU, but cannot run on blackwell architecture cards (i.e. Nvidia 50-series cards). 
I recommend use with Nvidia 30-series or 40-series cards with minimum 12GB graphics ram.

## 🔔 Stay Updated
**Never miss a new feature or bug fix.** Click the **Watch** button at the top right of this GitHub page, select **Custom**, and check the box for **Releases**. GitHub will automatically send you an email notification whenever a new version of OpenCap Offline is published!

## Prerequisites
* Python 3.9.25
* Git
* NVIDIA GPU (RTX 3000 series or 4000 series recommended) - this setup is not yet suitable for GPUs with Blackwell architecture
* Anaconda or Miniconda

## Installation & Setup

**1. Download the Code**  
Clone this repository to your local machine:  
`git clone https://github.com/driscollh/opencap-offline.git`

**2. Create a Python Environment**  
`conda create -n opencap_env python=3.9 -y`  
`conda activate opencap_env`  

**3. Install the Correct Chumpy Version**  
Navigate into the main folder  
`pip install chumpy==0.70 --no-build-isolation`

**4. Install CPython and OpenSim Packages**  
Navigate into the main folder and install the required Python environment packages, followed by opensim:  
`pip install -r requirements.txt`
`conda install -c opensim-org opensim=4.4 -y`

**5. Download the Local Dependencies (Required)**  
Because the machine learning models and background engines (OpenPose, FFmpeg) are too large for GitHub, they are hosted securely on Zenodo.  
* Go to the V3 Zenodo archive: [https://doi.org/10.5281/zenodo.19447679](https://doi.org/10.5281/zenodo.19447679)  
* Download the `dependencies.zip` file.  
* Extract the contents directly into the empty `dependencies/` folder inside this project. This should result in three separate subfolders, one for each key dependency.
* Extract the `RTMPose.zip` file directly into the main project folder next to `pyqt5_launcher_improved.py`. This folder should be titled "Blackwell_RTMPose", and contain a `dependencies/` folder and `mmcv/` folder.

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

## Citation

If you use OpenCap Offline in your research or clinical workflow, please cite it using the following DOI:

**APA:**
> Driscoll, H. G. (2026). OpenCap Offline (Version 2.1.1) [Computer software]. https://doi.org/10.5281/zenodo.19562446

**BibTeX:**
```bibtex
@software{opencap_offline_2026,
  author       = {Driscoll, Harry G.},
  title        = {OpenCap Offline},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {2.1.1},
  doi          = {10.5281/zenodo.19562446},
  url          = {[https://doi.org/10.5281/zenodo.19562446](https://doi.org/10.5281/zenodo.19562446)}
}
  
