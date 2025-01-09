import os
import json
import cv2
import numpy as np
import supervisely as sly
from supervisely.video_annotation.key_id_map import KeyIdMap

video_path = r'/home/luc/Downloads/Contrail_US000F 2024-11-09 to 2024-12-05_US000F_20241130-335_frames_timelapse.mp4'

annotations_path = r'/home/luc/Downloads/Contrail_US000F 2024-11-09 to 2024-12-05_US000F_20241130-335_frames_timelapse.mp4.json'
meta_path = r'/home/luc/Downloads/91_3_1_Contrail/Contrail/meta.json'

# Base directories for output
jpeg_base_dir = "/home/luc/Projects/Samples/train/JPEGImages"
png_base_dir = "/home/luc/Projects/Samples/train/Annotations"

# Extract video name from path (without extension)
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Create output directories for this video
frames_dir = os.path.join(jpeg_base_dir, video_name)
masks_dir = os.path.join(png_base_dir, video_name)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Load ProjectMeta
project_meta = sly.ProjectMeta.from_json(sly.json.load_json_file(meta_path))

# Load annotation JSON
with open(annotations_path, 'r') as f:
    annotation_json = json.load(f)

video_ann = sly.VideoAnnotation.from_json(annotation_json, project_meta, key_id_map=KeyIdMap())

# Create object mapping (assuming code from previous steps if needed)
obj_mapping = {}
for obj_json, video_object in zip(annotation_json['objects'], video_ann.objects):
    original_id = obj_json['id']
    obj_mapping[video_object] = original_id

def get_unique_color(object_id: int):
    r = (object_id * 37) % 256
    g = (object_id * 73) % 256
    b = (object_id * 109) % 256
    return [r, g, b]

object_id_to_color = {obj_mapping[o]: get_unique_color(obj_mapping[o]) for o in obj_mapping}

def process_video_annotations(video_path, video_annotation: sly.VideoAnnotation, frames_dir: str, masks_dir: str):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ann_height, ann_width = video_annotation.img_size
    frame_index_to_figures = {frame.index: frame.figures for frame in video_annotation.frames}

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if necessary
        if (frame_height, frame_width) != (ann_height, ann_width):
            frame = cv2.resize(frame, (ann_width, ann_height))

        # Zero-padded frame name
        frame_name = f"{frame_idx:05d}"

        # Save the raw frame as JPEG
        frame_output_path = os.path.join(frames_dir, f"{frame_name}.jpg")
        cv2.imwrite(frame_output_path, frame)

        # Create a blank mask
        mask = np.zeros((ann_height, ann_width, 3), dtype=np.uint8)

        ann_frame = frame_index_to_figures.get(frame_idx, [])
        for figure in ann_frame:
            obj_id = obj_mapping[figure.video_object]
            color = object_id_to_color[obj_id]
            figure.geometry.draw(mask, color=color, thickness=-1)

        # Save mask as PNG
        mask_output_path = os.path.join(masks_dir, f"{frame_name}.png")
        cv2.imwrite(mask_output_path, mask)

    cap.release()
    print("Processing completed.")

process_video_annotations(video_path, video_ann, frames_dir, masks_dir)
