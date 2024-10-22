# MRL Eye Dataset

## Overview

The **MRL Eye Dataset** is a large-scale dataset of human eye images, designed for tasks such as eye detection, gaze estimation, and blink detection in computer vision. The dataset includes infrared images in various lighting conditions and resolutions, captured by different devices.

## Dataset Contents

- **Subject ID**: Data from 37 individuals (33 men and 4 women).
- **Image ID**: The dataset contains 84,898 images.
- **Gender**: 0 for male, 1 for female.
- **Glasses**: 0 (no glasses), 1 (glasses).
- **Eye State**: 0 (closed), 1 (open).
- **Reflections**: 0 (none), 1 (small), 2 (big).
- **Lighting Conditions**: 0 (bad), 1 (good).
- **Sensor ID**: Data captured by three sensors:
  - Intel RealSense RS 300 (640x480 resolution)
  - IDS Imaging (1280x1024 resolution)
  - Aptina sensor (752x480 resolution)

## Downloads

- **Dataset**: [Download Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)
- **Pupil Annotations**: [Download Pupil Annotations](http://mrl.cs.vsb.cz/data/eyedataset/pupil.txt)

## Contact

For any questions about the dataset, please contact [Radovan Fusek](http://mrl.cs.vsb.cz//people/fusek/).

## Example Images

The dataset includes both open and closed eye images. Below are examples:

![Eye Image Example](http://mrl.cs.vsb.cz/images/eyedataset/eyedataset01.png)

## References

- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- [Kaggle: MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)
