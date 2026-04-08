# PokemonChess ♟️💤

Summoning Snorlax on a chessboard using OpenCV Pose Estimation.

## 📺 Demo

![Snorlax AR Demo](./demo.mov)

## 📝 Description

This project is a Python application that utilizes computer vision techniques to track a real-world chessboard and overlay a virtual Pokemon (Snorlax) using Augmented Reality (AR).

## 🚀 Key Features

- **Camera Pose Estimation:** Calculates the real-time 3D translation and rotation (pose) of the camera using OpenCV's `findChessboardCorners` and `solvePnP` algorithms.
- **Dynamic AR Visualization:** Projects 3D spatial coordinates onto a 2D screen space (`projectPoints`) based on the camera's dynamic pose, rendering a natural, perspective-aware AR image.

## 🛠️ How it Works

1. Applies the pre-calibrated intrinsic camera matrix ($K$) and distortion coefficients.
2. Extracts 2D corner points from the physical chessboard in every video frame.
3. Estimates the extrinsic camera parameters using 2D-3D point correspondences.
4. Calculates the pixel distance to dynamically adjust the scale and position of the 2D Snorlax image (`.png`), anchoring it to the 3D center of the chessboard with accurate perspective.

## ⚙️ Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
