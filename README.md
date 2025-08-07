# Comprehensive Football Analysis

This project focuses on analyzing football (soccer) games through video footage. Using object detection models, the project tracks players and the ball, providing a variety of insights such as player positions, ball movement, and even generating bird's-eye view images of the field. The core of the project is based on YOLO (You Only Look Once) for object detection and other techniques like perspective transformation and player tracking.

## Features

* **Player Detection & Tracking**: Identifies and tracks players throughout the match using deep learning models (YOLOv5 and DeepSORT).
* **Ball Detection**: Detects the position of the ball on the field in real-time.
* **Bird's Eye View**: Generates an overhead view of the football field based on detected player positions.
* **Frame Extraction & Video Processing**: Extracts frames from football match videos for further analysis and creates new video outputs with player markings.
* **Customization**: Flexible settings for detecting and processing football videos.

## Prerequisites

To use this project, ensure you have the following installed:

* **Python 3.x** or higher
* **Libraries**:

  * `opencv-python-headless`
  * `matplotlib`
  * `torch`
  * `ultralytics`
  * `tqdm`
  * `color-thief`
  * `numpy`
  * `tensorflow` (optional, if using for model loading)

To install dependencies, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/comprehensive-football-analysis.git
   cd comprehensive-football-analysis
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv5** (if not included by default):

   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   ```

4. **Download the dataset or input video**:

   * Ensure you have a video file (`football.mp4`) placed in the directory for analysis.

## Usage

1. **Video Processing and Player Marking**:
   To process a video and track players, use the `player_marking` function. For example:

   ```python
   player_marking("input_video.mp4", "output_video.mp4")
   ```

2. **Run Image Extraction**:
   The `run_image_extraction` function allows you to extract frames from the video, focusing on specific scenes, such as when the green color dominates (indicating the field). For example:

   ```python
   run_image_extraction("input_video.mp4", "output_video.mp4")
   ```

3. **Bird’s Eye View Generation**:
   The project can generate a bird’s-eye view of the football field. This requires configuring the homography matrix and applying perspective transformations based on player and ball detections.

4. **Annotations & Output**:
   Annotated frames can be saved in the output folder. You can visualize player locations, ball tracking, and more in the generated video:

   ```python
   annotated_video = player_marking("input_video.mp4", "annotated_output.mp4")
   ```

## Code Overview

### **Key Modules**

* `BirdsEyeView_`:

  * Handles perspective transformation and detection.
  * Generates the bird's-eye view of the football field based on tracked player positions.

* `imageextract.py`:

  * Extracts frames from the input video based on a given frame rate.
  * Saves selected frames based on specific conditions (e.g., dominant green color).

* `player_marking.py`:

  * Handles player detection, marking, and tracking.
  * Saves annotated frames or videos showing player positions, ball detection, and other game metrics.

* `main.py`:

  * The main script that coordinates video input, processing, and saving outputs.
  * Initializes models for player and ball detection, and performs perspective transformations.

### **Functions**

* `generate_frames`: Extracts frames from a video.
* `plot_image`: Visualizes images using `matplotlib`.
* `filter_detections_by_class`: Filters detections based on class names (e.g., players or ball).
* `draw_rect`, `draw_ellipse`, `draw_polygon`: Annotates frames with shapes to represent objects.

### **Classes**

* **Detection**: Represents the detection data of an object, including its bounding box, class ID, and confidence score.
* **BaseAnnotator**: A class responsible for annotating images with various shapes and texts based on detection data.

## Project Structure

```
├── BirdsEyeView_/
│   ├── deep_sort_pytorch/
│   ├── elements/
│   ├── inference/
│   ├── perspective_transform/
│   ├── yolov5/
│   ├── .gitignore
│   ├── README.md
│   ├── arguments.py
│   ├── main.py
│   └── requirements.txt
├── imageextract.py
├── player_marking.py
├── requirements.txt
├── README.md
└── main.py
```

## Example Output

* **Annotated Video**: The video output with annotations such as player positions, ball location, and other analysis.
* **Bird's Eye View Image**: An overhead view of the football field highlighting player and ball positions.

## Troubleshooting

* Ensure all dependencies are installed correctly.
* If the `yolov5` repository is missing, clone it manually into the project directory.
* Adjust the `conf_thresh` (confidence threshold) and `iou_thresh` (Intersection Over Union threshold) for YOLOv5 if detections are not accurate.

