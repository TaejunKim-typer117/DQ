# DQ Viewer

Data Quality Viewer for WindBlade and SolarPanel datasets with web-based interface.

## Quick Start

### 1. Download Data
```bash
python3 download_data.py
```
Downloads images and labels from S3 using `sampled_files_*.csv`.

### 2. Run Viewer
```bash
python3 dq_viewer.py
```
Open http://localhost:5001 in your browser.

## Requirements

- Python 3.x
- Flask, Pillow (see venv)
- exiftool (install via `brew install exiftool`)

## Features

- Browse WindBlade and SolarPanel datasets
- View images with annotation overlays
- Inspect JSON metadata and EXIF data
- Navigate with keyboard shortcuts (← →)
