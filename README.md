# SARDetect — Dual Method SAR Vessel Detection

A Streamlit web application that detects vessels in Sentinel-1 SAR imagery using two complementary methods:

- **CFAR** — physics-based detection via elevated radar backscatter from vessel hulls
- **YOLOv8-OBB** — deep learning wake detection trained on the OpenSARWake dataset

## Live App

👉 [Open SARDetect on Streamlit Cloud](https://sardetect.streamlit.app)

## Supported Input Formats

- GeoTIFF (`.tif` / `.tiff`) — calibrated or raw
- PNG — pre-processed SAR chips
- Zipped Sentinel-1 SAFE folder (`.zip`) — full radiometric calibration applied automatically

## Methods

### CFAR (Constant False Alarm Rate)
Replicates the ArcGIS Detect Bright Ocean Objects workflow in pure Python. Converts raw DN values to gamma-naught (γ0) backscatter, applies a Lee speckle filter, then runs adaptive CFAR thresholding. Detects vessels via their elevated radar cross-section relative to the surrounding sea clutter.

### YOLOv8-OBB Wake Detection
Oriented bounding box detector trained on the OpenSARWake dataset (Xu & Wang, 2024). Detects vessels indirectly via the hydrodynamic wake signature they produce in the water surface. Large scene inference handled via SAHI (Slicing Aided Hyper Inference) to preserve 1024×1024 training chip resolution.

## Getting Started

### Run locally

```bash
git clone https://github.com/Yelow47/GE7090
cd GE7090
pip install -r requirements.txt
streamlit run app.py
```

### Model weights

Download `best.pt` from the [Releases page](https://github.com/Yelow47/GE7090/releases/tag/v1.0) and place it in a `weights/` folder, or use the Download Weights button in the app sidebar.

## Getting Sentinel-1 Data

Free Sentinel-1 IW mode GRD data is available from the [Copernicus Data Space](https://dataspace.copernicus.eu). Select:
- Mode: IW (Interferometric Wide Swath)
- Product: GRD
- Polarization: VV+VH

Busy shipping lanes recommended: Strait of Gibraltar, English Channel, North Sea.

## Citation

Dataset:
```
Xu, C. & Wang, X. (2024). OpenSARWake: A Large-Scale SAR Dataset for Ship Wake 
Recognition with a Feature Refinement Oriented Detector. 
IEEE Geoscience and Remote Sensing Letters, 21. 
DOI: 10.1109/LGRS.2024.3392681
```

## Course Context

Developed as a final project for GE7090 Advanced Remote Sensing, Stockholm University.  
Research question: *How does physics-based CFAR detection compare to deep learning wake-based detection for SAR vessel identification, and under what conditions does each method succeed or fail?*
