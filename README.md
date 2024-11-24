# Package Sorting System

## Overview

The **Package Sorting System** is an innovative solution designed to streamline the sorting of packages in response to the growing demand in the e-commerce and logistics industries. This system leverages **Computer Vision (CV)**, **Optical Character Recognition (OCR)**, and **Machine Learning** techniques to automate and optimize the process of sorting packages based on addresses, zip codes, and item classifications. 

## Features

- **Automated Label Recognition**: Utilizes OCR (via PyTesseract) for text extraction from package labels.
- **Dynamic Sorting Logic**: Implements efficient data management and validation using Pandas.
- **Real-Time Processing**: Processes package data and sorts them in real-time for high throughput.
- **Scalable Design**: Adaptable to various sorting requirements and scales for warehouse operations.

## Project Components

### Technologies Used
- **Python Libraries**:
  - `OpenCV`: For image processing and object detection.
  - `PyTesseract`: For OCR to extract text from package labels.
  - `Pandas`: For data analysis and management.
  - `NumPy`: For numerical computations.
- **Hardware Requirements**:
  - Camera system for capturing package images.
  - Conveyor belt or sorting mechanism (optional, for demonstration).

### Algorithms
1. **Dynamic Image Adjustment**: Pre-processes images for improved OCR accuracy.
2. **Object Detection**: Detects and extracts label information using OpenCV.
3. **Sorting Logic**: Validates and categorizes items based on extracted data (e.g., zip codes and item types).

## Flowchart

For an illustrative project flow, refer to [this flowchart](https://drive.google.com/file/d/1YG1yhXGP2dJqlWmsKvJ3G0gSO5Z95xEq/view?usp=drive_link).

## Installation Guide

1. **Dependencies**: Install required libraries using pip:
   ```bash
   pip install opencv-python pytesseract pandas numpy
   ```
2. **Setup OCR**: Ensure Tesseract-OCR is installed and properly configured in the system.
3. **Run Script**: Execute the main script to start the sorting process.

## Dataset

The dataset includes:
- **Images**: Sample package images with labels.
- **Text Data**: Zip codes and item names for validation.

## Usage

1. **Capture Package Image**: Use the camera system to capture package images.
2. **Process Data**: The system extracts and validates package details.
3. **Sort Packages**: Based on rules (e.g., zip codes), the system categorizes and sorts packages.

## Future Enhancements

- Integration with **IoT Devices** for real-time tracking.
- Use of **Advanced Object Recognition** for non-standard label formats.
- Implementation of **Predictive Algorithms** for faster sorting.

## Team Members

- Kunlanith Busabong  
- Akesit Akkharasasiri  
- Paveetida Tiranatwittayakul  
- Rattapol Kitirak  

## License

This project is licensed under the MIT License.
