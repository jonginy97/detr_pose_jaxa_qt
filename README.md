
# Robot End Effector 6D Pose Estimation with Modified DETR

This repository provides a modified version of [DETR (DEtection TRansformers)](https://github.com/facebookresearch/detr), designed specifically for 6D pose estimation of robot end effectors. In this setup:

- The package predicts both the position and orientation of the robot’s end effector.
- Orientation is predicted in quaternion format for more precise rotational control.

## Installation

1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. **Remove the existing Git history (optional):**
    If you cloned the repository and want to start fresh without any existing `.git` history:
    ```bash
    rm -rf .git
    git init
    ```

## Training

To train the model with multiple processes, use the following command:
```bash
torchrun --nproc_per_node=n main.py
```
Replace `n` with the number of processes you want to run concurrently.

## Data Format

Ensure your data follows the expected format:
- **Position:** Provided in 3D coordinates `[x, y, z]`.
- **Orientation:** Represented as a quaternion `[w, x, y, z]`.

## Key Files

- `main.py` - Main training script for running the modified DETR model.
- `dataset.py` - Custom dataset loading script designed to handle 6D pose data for end effectors.
- `models/` - Contains model architecture and modifications applied to DETR.

## Notes

- The model is designed to work with quaternion outputs for orientation, which avoids singularities associated with Euler angles and allows continuous and unambiguous rotational representation.

For further information or troubleshooting, please refer to the code comments and function docstrings provided in the source files.

