# ComfyUI-Yolo-Person-Tracker

A custom node for **ComfyUI** that uses **YOLO (v8/v11)** Instance Segmentation to track specific people in video frames. It generates high-quality tracking masks and visualized overlays.  
Unlike simple object detection, this tracker uses **Color Histogram Re-Identification** to maintain a lock on a specific person even if they are temporarily occluded or the camera cuts, provided their appearance remains similar.  

<p align="center">
  <img src="https://ai.static.ad2.cc/tracker1.png" width="700" />
</p>

<p align="center">
  <img src="https://ai.static.ad2.cc/tracker2.png" width="700" />
</p>

---

## ✨ Features

- **Instance Segmentation**: Uses yolov8-seg or yolo11-seg models for precise pixel-level masks (not just bounding boxes).
- **Target Locking**: Select a specific person based on their position in the first frame (e.g., "The 2nd person from the right").
- **Robust Tracking**: Implements HSV Color Histogram comparison to re-identify the target across frames, robust against minor light changes and occlusions.
- **Edge Smoothing**: Automatically applies morphological operations and Gaussian blur to output smooth, organic masks (no jagged edges).
- **Dual Output**: Outputs both a visual preview (original footage + translucent green overlay) and a clean black-and-white mask video sequence.

---

## 🛠️ Installation

### 1. Clone the Repository

Navigate to your **ComfyUI custom_nodes** directory and clone this repo:

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/YourUsername/ComfyUI-Yolo-Person-Tracker.git


```

### 2. Install Dependencies

Make sure you have the required Python packages installed (YOLO requires ultralytics):

```bash
pip install ultralytics opencv-python numpy
```

(Note: torch is already required by ComfyUI, so it is assumed to be installed.)

## 📂 Model Setup

This node requires YOLO segmentation models (.pt files). Navigate to your ComfyUI models directory. Create a new folder named `yolo`. Download the supported models and place them in `ComfyUI/models/yolo/`.

### Directory Structure:

```
ComfyUI/
└── models/
    └── yolo/
        ├── yolov8l-seg.pt
        ├── yolov8x-seg.pt
        └── ...
```

### Supported Models:

The node supports the following models (it will attempt to download them automatically via ultralytics if not found, but manual placement is recommended):

- `yolov8x-seg.pt` (Highest accuracy, slower)
- `yolov8l-seg.pt` (Good balance)
- `yolov8m-seg.pt`
- `yolo11x-seg.pt` (Newer architecture)
- `yolo11l-seg.pt`

## 🧩 Usage

Search for the node "YOLO Person Tracker" in ComfyUI (Category: Tracking).

### Inputs

| Parameter        | Description                                                                                                                                                  |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `images`         | The input image sequence (video frames). Usually from Load Video (Upload) or VHS_LoadVideo.                                                                 |
| `fps`            | Frame rate of the input video (affects output format metadata).                                                                                              |
| `yolo_model`     | Select which model to use. `yolov8x-seg` or `yolo11x-seg` are recommended for best edge quality.                                                             |
| `sort_direction` | Defines how to count people in the scene to find your target. Options:                                                                                       |
|                  | - `left-to-right`: Count from the left side.                                                                                                                  |
|                  | - `right-to-left`: Count from the right side.                                                                                                                 |
| `target_index`   | The index of the person to track (0-based).                                                                                                                  |
|                  | - 0 = The 1st person.                                                                                                                                         |
|                  | - 1 = The 2nd person, etc.                                                                                                                                   |
| Example: To track the person on the far right, set `right-to-left` and index 0.                                                                               |
| `conf_threshold` | YOLO detection confidence (0.0 - 1.0). Default 0.5. Increase if it detects background objects as people.                                                    |
| `sim_threshold`  | Similarity threshold (0.0 - 1.0) for the color tracker. Default 0.5. Lower this if the tracker loses the target when lighting changes.                         |

### Outputs

| Output           | Description                                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------------------------|
| `vis_frames`     | Original video frames with a semi-transparent green overlay on the tracked person. Useful for previewing accuracy. |
| `mask_frames`    | Black and white RGB images (White = Target, Black = Background). Can be used with VHS_VideoCombine or processed further. |
| `fps`            | Passes through the input FPS.                                                                                      |

## 🚀 Example Workflow

1. **Load Video**: Use VHS_LoadVideo or standard Load Image Sequence.
2. **Connect Tracker**: Connect the image output to YOLO Person Tracker.
3. **Preview/Save**: 
    - Connect `vis_frames` to VHS_VideoCombine to see the tracking result.
    - Connect `mask_frames` to VHS_VideoCombine to save the matte.
    - Or use Preview Image to check individual frames.

## ⚙️ How it Works

### Initialization:
In the first frame where people are detected, the node sorts all detected persons based on the `sort_direction`. It picks the person at `target_index`.

### Feature Extraction:
It extracts a color histogram (HSV) of that specific person's clothing/appearance.

### Tracking Loop:
For every subsequent frame, it detects all people and compares their appearance to the saved target feature. The person with the highest similarity (above `sim_threshold`) is identified as the target.

### Refinement:
The raw YOLO mask is processed with morphological dilation/erosion and Gaussian blur to create smooth, professional-looking edges.

## 📝 Changelog

- **v1.0.0**: Initial release. Support for YOLOv8/v11 segmentation, Left/Right selection logic, and edge smoothing.

## 🤝 Contributing

Feel free to submit issues or pull requests if you have ideas for improvements (e.g., adding DeepSort or ResNet Re-ID features).

## 📄 License

This project is open-source. Please ensure you comply with the licenses of the underlying libraries (Ultralytics YOLO, OpenCV, ComfyUI).
