import os
import json
import cv2
import numpy as np
import supervisely as sly
from supervisely.video_annotation.key_id_map import KeyIdMap
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class VideoSegment:
    start_frame: int
    end_frame: int
    segment_id: int

def find_annotation_segments(video_annotation: sly.VideoAnnotation, frame_count: int, buffer_frames: int = 10) -> List[VideoSegment]:
    frame_index_to_figures = {frame.index: frame.figures for frame in video_annotation.frames}
    segments = []
    current_segment = None
    
    for frame_idx in range(frame_count):
        has_annotation = frame_idx in frame_index_to_figures and len(frame_index_to_figures[frame_idx]) > 0
        
        if has_annotation and current_segment is None:
            start = max(0, frame_idx - buffer_frames)
            current_segment = VideoSegment(start, start, len(segments))
        elif not has_annotation and current_segment is not None:
            current_segment.end_frame = min(frame_count - 1, frame_idx + buffer_frames)
            segments.append(current_segment)
            current_segment = None
    
    if current_segment is not None:
        current_segment.end_frame = min(frame_count - 1, frame_idx + buffer_frames)
        segments.append(current_segment)
    
    return segments

def process_video_segment(cap: cv2.VideoCapture, segment: VideoSegment, video_annotation: sly.VideoAnnotation, 
                         jpeg_base_dir: str, png_base_dir: str, object_id_to_color: dict, obj_mapping: dict):
    segment_name = f"{video_name}_segment_{segment.segment_id}_frames_{segment.start_frame}-{segment.end_frame}"
    frames_dir = os.path.join(jpeg_base_dir, segment_name)
    masks_dir = os.path.join(png_base_dir, segment_name)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    ann_height, ann_width = video_annotation.img_size
    frame_index_to_figures = {frame.index: frame.figures for frame in video_annotation.frames}
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
    
    for frame_idx in range(segment.start_frame, segment.end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_height, frame_width = frame.shape[:2]
        if (frame_height, frame_width) != (ann_height, ann_width):
            frame = cv2.resize(frame, (ann_width, ann_height))
            
        rel_frame_idx = frame_idx - segment.start_frame
        frame_name = f"{rel_frame_idx:05d}"
        
        frame_output_path = os.path.join(frames_dir, f"{frame_name}.jpg")
        cv2.imwrite(frame_output_path, frame)
        
        mask = np.zeros((ann_height, ann_width, 3), dtype=np.uint8)
        ann_frame = frame_index_to_figures.get(frame_idx, [])
        for figure in ann_frame:
            obj_id = obj_mapping[figure.video_object]
            color = object_id_to_color[obj_id]
            figure.geometry.draw(mask, color=color, thickness=-1)
            
        mask_output_path = os.path.join(masks_dir, f"{frame_name}.png")
        cv2.imwrite(mask_output_path, mask)

def main():
    global video_name
    video_path = r'/home/luc/Downloads/Contrail_US000F 2024-11-09 to 2024-12-05_US000F_20241130-335_frames_timelapse.mp4'
    annotations_path = r'/home/luc/Downloads/Contrail_US000F 2024-11-09 to 2024-12-05_US000F_20241130-335_frames_timelapse.mp4.json'
    meta_path = r'/home/luc/Downloads/91_3_1_Contrail/Contrail/meta.json'
    
    jpeg_base_dir = "/home/luc/Projects/Samples/train/JPEGImages"
    png_base_dir = "/home/luc/Projects/Samples/train/Annotations"
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    project_meta = sly.ProjectMeta.from_json(sly.json.load_json_file(meta_path))
    with open(annotations_path, 'r') as f:
        annotation_json = json.load(f)
    video_ann = sly.VideoAnnotation.from_json(annotation_json, project_meta, key_id_map=KeyIdMap())
    
    obj_mapping = {}
    for obj_json, video_object in zip(annotation_json['objects'], video_ann.objects):
        obj_mapping[video_object] = obj_json['id']
    
    def get_unique_color(object_id: int):
        return [(object_id * 37) % 256, (object_id * 73) % 256, (object_id * 109) % 256]
    
    object_id_to_color = {obj_mapping[o]: get_unique_color(obj_mapping[o]) for o in obj_mapping}
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    segments = find_annotation_segments(video_ann, frame_count)
    print(f"Found {len(segments)} segments with annotations")
    
    for segment in segments:
        print(f"Processing segment {segment.segment_id} (frames {segment.start_frame}-{segment.end_frame})")
        process_video_segment(cap, segment, video_ann, jpeg_base_dir, png_base_dir, 
                            object_id_to_color, obj_mapping)
    
    cap.release()
    print("Processing completed.")

if __name__ == "__main__":
    main()