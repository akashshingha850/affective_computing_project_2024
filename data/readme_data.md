# MRL Infrared Eye Images Dataset for Drowsiness Detection (Forked Version)

This dataset is a **forked version** of the original MRL Eye Dataset, containing infrared eye images categorized into **Open** and **Closed** states. It is split into training, validation, and test sets, comprising over **85,000 images** captured under various lighting conditions using multiple sensors. This dataset is tailored for tasks such as eye detection, gaze estimation, blink detection, and drowsiness analysis in computer vision.

## Dataset Structure

- **Train**: Open (25,770), Closed (25,167)
- **Validation**: Open (8,591), Closed (8,389)
- **Test**: Open (8,591), Closed (8,390)

## Directory Tree
```plaintext
data/
├── train/
│   ├── open/
│   └── closed/
├── val/
│   ├── open/
│   └── closed/
└── test/
    ├── open/
    └── closed/ 
```



## Metadata

- **subject_id**: Unique identifier for each subject (37 subjects)
- **Attributes**: Eye state, gender, glasses, reflections, lighting, and sensor ID

## Original Dataset Overview

The **MRL Eye Dataset** is a large-scale dataset of human eye images designed for tasks such as eye detection, gaze estimation, and blink detection in computer vision. The dataset includes infrared images in various lighting conditions and resolutions, captured by different devices.

## Dataset Contents

- **Total Images**: 84,898 images in the original dataset.
- **Subject ID**: Data from 37 individuals (33 men and 4 women).
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
- **GitHub Release**: [Dataset Release](https://github.com/akashshingha850/affective_computing_project/releases/tag/dataset)
- **Kaggle Dataset**: [Kaggle: MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)


- **Original Dataset**: [Download Original Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)



## Contact

For any questions about the dataset, please contact [Radovan Fusek](http://mrl.cs.vsb.cz//people/fusek/).

## Example Images

The dataset includes both open and closed eye images. Below are examples:

![Eye Image Example](http://mrl.cs.vsb.cz/images/eyedataset/eyedataset01.png)

## References

- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- [Kaggle: MRL Eye Dataset (forked)](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)