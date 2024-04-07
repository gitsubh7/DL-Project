#!pip install colorthief

# @title
from colorthief import ColorThief

# Commented out IPython magic to ensure Python compatibility.
# @title
import os
HOME = os.getcwd()
print(HOME)
# %cd {HOME}
# !git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# %pip install -r requirements.txt
# %cd {HOME}
import torch
from typing import Generator
import matplotlib.pyplot as plt
import numpy as np
import cv2
# %matplotlib inline
def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        yield frame

    video.release()
def plot_image(image: np.ndarray, size: int = 12) -> None:
    plt.figure(figsize=(size, size))
    plt.imshow(image[...,::-1])
    plt.show()

# @title
import torch
#!pip install -U ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# @title
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
@dataclass(frozen=True)
class Point:
    x: float
    y: float
    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)
@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float
    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)
    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )

    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y


@dataclass
class Detection:
    rect: Rect
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None
    @classmethod
    def from_results(cls, pred: np.ndarray, names: Dict[int, str]) -> List[Detection]:
        result = []
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            class_id=int(class_id)
            result.append(Detection(
                rect=Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence)
            ))
        return result

def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
    return [
        detection
        for detection
        in detections
        if detection.class_name == class_name
    ]


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r
    @classmethod
    def from_hex_string(cls, hex_string: str) -> Color:
        r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return Color(r=r, g=g, b=b)


def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image

def draw_polygon(image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
    return image

def draw_text(image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.putText(image, text, anchor.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.bgr_tuple, thickness, 2, False)
    return image

def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image

@dataclass
class BaseAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection],cols) -> np.ndarray:
        annotated_image = image.copy()
        for i,detection in enumerate(detections):
            annotated_image = draw_rect(
                image=image,
                rect=detection.rect,
                color=self.colors[cols[i]],
            )
            annotated_image = draw_text(
                image=annotated_image,
                anchor = Point(detection.rect.x,detection.rect.y),
                text="Teamred" if self.colors[cols[i]].bgr_tuple[-1]>120 else "Teamblue",
                color=self.colors[cols[i]],
            )
            annotated_image = draw_ellipse(
                image=annotated_image,
                rect = detection.rect,
                color=self.colors[cols[i]],
            )

        return annotated_image

# @title
cols =  ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#800080", "#FFC0CB", "#A52A2A", "#808080", "#FFFFFF",
        "#000000", "#00FFFF", "#FF00FF", "#8B008B", "#4B0082", "#40E0D0", "#800000", "#808000", "#000080", "#C0C0C0"]
COLORS = [Color.from_hex_string(i) for i in cols]
THICKNESS = 4

# @title
annotator = BaseAnnotator(
    colors=COLORS,
    thickness=THICKNESS)
frame_iterator = iter(generate_frames(video_file="/content/CityUtdR.mp4"))

frame = next(frame_iterator)

results = model(frame, size=1280)

detections = Detection.from_results(
    pred=results.pred[0].cpu().numpy(),
    names=model.names)
cols = []

def matching(col, image):
    hex_color = col.lstrip('#')
    color_values = np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])
    image_values = np.mean(image, axis=(0, 1))
    diff = np.abs(color_values - image_values)
    confidence = 1 - np.mean(diff) / 255.0
    return confidence

for drec in detections:
  rec = drec.rect
  x_start, x_end, y_start, y_end = int(rec.x), int(rec.x + rec.width), int(rec.y), int(rec.y + rec.height)
  x_end = min(x_end,frame.shape[1])
  y_end = min(y_end,frame.shape[0])
  image = frame[y_start:y_end, x_start:x_end, :]

  maxcol = 0
  maxconf = 0
  for i, col in enumerate(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#800080", "#FFC0CB", "#A52A2A", "#808080", "#FFFFFF",
        "#000000", "#00FFFF", "#FF00FF", "#8B008B", "#4B0082", "#40E0D0", "#800000", "#808000", "#000080", "#C0C0C0"]):
    if matching(col,image)>maxconf:
      maxconf=matching(col,image)
      maxcol = i
  cols.append(maxcol)

annotated_image = annotator.annotate(
    image=frame,
    detections=detections,
    cols=cols)

plot_image(annotated_image, 16)
num_of_players = 0
for det in detections:
  num_of_players+=(det.class_id==0)
print(f"there are {num_of_players} number of players there")

# @title
#video
import cv2
import warnings
from tqdm import tqdm


def player_marking(video_path, output_video_path='annotated_video.mp4'):

  warnings.filterwarnings("ignore", category=RuntimeWarning)
  annotator = BaseAnnotator(
    colors=COLORS,
    thickness=THICKNESS)
  frame_iterator = iter(generate_frames(video_file="/content/CityUtdR.mp4"))

  frame = next(frame_iterator)

  results = model(frame, size=1280)

  detections = Detection.from_results(
      pred=results.pred[0].cpu().numpy(),
      names=model.names)
  cols = []

  for drec in detections:
    rec = drec.rect
    x_start, x_end, y_start, y_end = int(rec.x), int(rec.x + rec.width), int(rec.y), int(rec.y + rec.height)
    x_end = min(x_end,frame.shape[1])
    y_end = min(y_end,frame.shape[0])
    image = frame[y_start:y_end, x_start:x_end, :]

    maxcol = 0
    maxconf = 0
    for i, col in enumerate(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#800080", "#FFC0CB", "#A52A2A", "#808080", "#FFFFFF",
          "#000000", "#00FFFF", "#FF00FF", "#8B008B", "#4B0082", "#40E0D0", "#800000", "#808000", "#000080", "#C0C0C0"]):
      if matching(col,image)>maxconf:
        maxconf=matching(col,image)
        maxcol = i
    cols.append(maxcol)

  annotated_image = annotator.annotate(
      image=frame,
      detections=detections,
      cols=cols)

  # plot_image(annotated_image, 16)
  num_of_players = 0
  for det in detections:
    num_of_players+=(det.class_id==0)
  print(f"there are {num_of_players} players there")

  fps = 30
  width, height = frame.shape[1], frame.shape[0]
  video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

  bar = tqdm(frame_iterator)
  for frame in bar:
      results = model(frame, size=1280)
      detections = Detection.from_results(
          pred=results.pred[0].cpu().numpy(),
          names=model.names)

      cols = []

      for drec in detections:
          rec = drec.rect
          x_start, x_end, y_start, y_end = int(rec.x), int(rec.x + rec.width), int(rec.y), int(rec.y + rec.height)
          image = frame[x_start:x_end, y_start:y_end, :]
          image = image[:, :image.shape[1] // 2, :]
          maxcol = 0
          maxconf = 0
          for i, col in enumerate(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#800080", "#FFC0CB", "#A52A2A",
                                  "#808080", "#FFFFFF", "#000000", "#00FFFF", "#FF00FF", "#8B008B", "#4B0082", "#40E0D0",
                                  "#800000", "#808000", "#000080", "#C0C0C0"]):
              if matching(col, image) > maxconf:
                  maxconf = matching(col, image)
                  maxcol = i
          cols.append(maxcol)

      annotated_image = annotator.annotate(
          image=frame,
          detections=detections,
          cols=cols)
      video_writer.write(annotated_image)
  video_writer.release()
  return f"Video saved to {output_video_path}"

player_marking("/content/CityUtdR.mp4", output_video_path="savehere.mp4")

