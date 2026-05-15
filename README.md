# Panorama Stitching Using Homography Estimation

This project implements an automated image stitching pipeline to create high-quality panoramas from multiple source images. It was developed for the **BMI4247 Computer Vision and Imaging** course.

## 🛠 Features & Methodology
The project follows a 5-stage computer vision pipeline:
1. **Feature Detection:** Utilizing **SIFT** (Scale-Invariant Feature Transform) to find robust keypoints.
2. **Feature Matching:** Implementing **FLANN-based Matcher** with **Lowe's Ratio Test** for accurate point pairing.
3. **Homography Estimation:** Using the **RANSAC** algorithm to estimate the transformation matrix and filter outliers.
4. **Perspective Warping:** Transforming images into a common coordinate system.
5. **Blending:** Applying **Feather Blending** (Distance Transform) to smooth seams and ensure seamless transitions.

## 👥 Team Members (Grup-3)
* **Ümit Sevil** - Feature Extraction & Project Coordination
* **Onur Kutan** - Feature Matching
* **Mustafa Yaman** - Homography Estimation
* **Sudenaz Ustabaş** - Blending & Optimization

## 🚀 How to Run
1. Ensure you have Python 3.10+ installed.
2. Install dependencies:
   ```bash
   pip install opencv-python numpy
