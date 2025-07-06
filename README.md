# Pythonized Bag of Visual Words

This project is a Python implementation of the Bag of Visual Words model, inspired by the MATLAB code from the [ICCV of 2005](https://people.csail.mit.edu/fergus/iccv2005/bagwords.html). It's currently a work in progress, with some components still being translated from MATLAB to Python.

## Project Status

⚠️ **Work in Progress** ⚠️
This repository is actively being developed. Some features are still being implemented and tested.

## Project Highlights

- **Feature Extraction**: Uses SIFT (Scale-Invariant Feature Transform) for detecting points of interest
- **Edge Detection**: Implements Canny edge detection through the [`vgg_xcv_segment`](vgg_xcv_segment.py) function
- **Data Processing**: Utilizes PyTorch's [`DataLoader`](ImageProcessor.py) for efficient batch processing
- **Modern Python Stack**: Built with NumPy, PyTorch, and OpenCV

## Project Structure

### Implemented Components
- [`EdgeSamplingNV.py`](EdgeSamplingNV.py): Edge sampling implementation
- [`ExtractSIFTFeatures.py`](ExtractSIFTFeatures.py): SIFT feature extraction
- [`ImageProcessor.py`](ImageProcessor.py): Image loading and processing pipeline
- [`ProcessingConfig.py`](ProcessingConfig.py): Configuration management
- [`main.py`](main.py): Main execution script

### Legacy/Reference Components
- `compute_descriptors.ln`
- `discrete_sampler.m`
- `vgg_*.m` files
- Various MEX files
