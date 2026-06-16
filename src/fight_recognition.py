"""
Three-stage pipeline for violent behavior detection on video:

    YOLOv11 (person detection + tracking)
        -> DBSCAN (spatial clustering of detections, adaptive eps)
        -> UniFormer (spatio-temporal classification on 16-frame crops)

Usage:
    python fight_recognition.py CONFIG CHECKPOINT LABEL VIDEO \
        --device cuda:0 \
        --yolo-version yolo11m.pt \
        --output output/result.mp4
"""
import argparse
import logging
import os
import time
from collections import defaultdict
from queue import Queue

import cv2
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import init_recognizer
from mmaction.utils import get_str_type

from ultralytics import YOLO
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.DEBUG)


FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)
RED_COLOR = (0, 0, 255)
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

# Pipeline thresholds, factored out of the code so they can be tuned
# without touching the logic. Values correspond to the ones used in the
# experiments described in the thesis (chapter 2-3).
CONFIDENCE_THRESHOLD = 0.75      # below this, prediction is forced to NORMAL
CLUSTER_EPS_COEFFICIENT = 0.5    # multiplier for mean bbox diagonal in DBSCAN
CLIPS_PER_DECISION = 3           # number of clip predictions averaged per ID
CROP_SIZE = (224, 224)           # input size expected by the classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description='Violence detection: YOLOv11 + DBSCAN + UniFormer pipeline'
    )
    parser.add_argument('config', help='test config file path (mmaction2 config)')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('label', help='label file path')
    parser.add_argument('video', help='video file')
    parser.add_argument('--device', type=str, default='cuda:0',
                         help='CPU/CUDA device option, e.g. cuda:0 or cpu')
    parser.add_argument('--yolo-version', type=str,
                         default='yolo11m.pt', help='YOLO checkpoint for person detection')
    parser.add_argument('--output', type=str, default='output/result.mp4',
                         help='path to the annotated output video')
    args = parser.parse_args()
    return args


class VideoProcessor:
    def __init__(self, args):
        """
        Class for video processing and action recognition.

        Initialization parameters are passed with parse_args() function.

        Attributes:
            args : Parsed CLI (config, checkpoint, label, video,
                device, yolo-version, output).
            device (torch.device): Computational device.
            label (list[str]): Loaded text labels.
            yolo_model: YOLO model for people detection.
            frame_queues (dict[Queue]): Frame buffer for building clips.
            clips (dict[list]): Collected clips.
            stop_flag (bool): Flag that should stop process.

        Methods:
            _init_video_streams(): Initializes video stream.
            _init_model(): Initialize action recognition model.
        """
        self.args = args
        self.device = torch.device(args.device)
        self.label = self._load_labels(args.label)
        self.yolo_model = YOLO(args.yolo_version).to(self.device)
        self._init_video_streams()
        self._init_model(args)
        self.frame_queues = defaultdict(
            lambda: Queue(maxsize=self.sample_length))
        self.clips = defaultdict(list)
        self.stop_flag = False

    def _load_labels(self, label_path):
        with open(label_path, 'r') as f:
            return [line.strip() for line in f]

    def _init_video_streams(self):
        self.cap = cv2.VideoCapture(self.args.video)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        output_path = self.args.output
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def _init_model(self, args):
        cfg = Config.fromfile(args.config)
        self.model = init_recognizer(cfg, args.checkpoint, device=args.device)
        self.data = dict(img_shape=None, modality='RGB', label=-1)
        sample_length = None

        pipeline = cfg.test_pipeline.copy()
        pipeline_ = pipeline.copy()
        for step in pipeline:
            if 'SampleFrames' in get_str_type(step['type']):
                sample_length = 16
                self.data['num_clips'] = 1
                self.data['clip_len'] = 16
                pipeline_.remove(step)
            if get_str_type(step['type']) in EXCLUED_STEPS:
                pipeline_.remove(step)

        assert sample_length > 0

        self.sample_length = sample_length
        self.test_pipeline = Compose(pipeline_)

    @staticmethod
    def get_diag(width_crop, height_crop):
        # bounding box diagonal
        diag = (width_crop ** 2 + height_crop ** 2) ** 0.5
        return diag

    def get_eps(self, boxes):
        # mean diagonal for clustering
        sum_of_diag = 0
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            sum_of_diag += self.get_diag(width, height)
        return sum_of_diag / len(boxes) * CLUSTER_EPS_COEFFICIENT

    def merge_boxes_dbscan(self, boxes, track_ids):
        """
        Merges nearby or overlapping bounding boxes using DBSCAN clustering.

        The function computes the center of each box, clusters them with DBSCAN
        and produces a single merged bounding box for each cluster.

        Args:
            boxes (list[list[int]]): List of bounding boxes in [x1, y1, x2, y2] format.
            track_ids (list[int]): Track IDs associated with each bounding box.

        Returns:
            tuple:
                merged_boxes (list[list[int]]): List of merged bounding boxes.
                merged_ids (list[list[int]]): Track IDs belonging to each merged box.
        """
        if len(boxes) == 0:
            return [], []

        boxes = np.array(boxes)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)

        centers = np.column_stack((
            (boxes[:, 0] + boxes[:, 2]) / 2.0,
            (boxes[:, 1] + boxes[:, 3]) / 2.0
        ))

        eps = self.get_eps(boxes)

        dbscan = DBSCAN(eps=eps, min_samples=1)
        cluster_labels = dbscan.fit_predict(centers)

        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        merged_boxes = []
        merged_ids = []

        for label in unique_labels:
            idxs = np.where(cluster_labels == label)[0]

            cluster_boxes = boxes[idxs]
            cluster_track_ids = [track_ids[i] for i in idxs]

            if cluster_boxes.ndim == 1:
                cluster_boxes = np.expand_dims(cluster_boxes, axis=0)

            x1_min = np.min(cluster_boxes[:, 0])
            y1_min = np.min(cluster_boxes[:, 1])
            x2_max = np.max(cluster_boxes[:, 2])
            y2_max = np.max(cluster_boxes[:, 3])

            merged_boxes.append([x1_min, y1_min, x2_max, y2_max])
            merged_ids.append(cluster_track_ids)

        return merged_boxes, merged_ids

    def detect_with_yolo(self, frame):
        """
        Runs YOLO tracking, merges nearby detections, and extracts crops for each tracked object.

        Args:
            frame (np.ndarray): Input video frame.

        Returns:
            tuple:
                detections (list[dict]): List of dictionaries containing:
                    - 'bbox': (x1, y1, x2, y2) merged bounding box
                    - 'track_id': ID of the tracked object
                crops (dict[int, np.ndarray]): Dictionary of track_id-cropped image.
        """
        results = self.yolo_model.track(
            frame, persist=True, classes=[0], device=self.device)

        if not results or len(results[0].boxes) == 0:
            return [], {}

        boxes_data = results[0].boxes
        boxes = []
        boxes_ids = []

        for box in boxes_data:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            track_id = int(box.id[0].cpu().numpy())
            boxes.append((x1, y1, x2, y2))
            boxes_ids.append(track_id)
        if boxes:
            merged_boxes, merged_ids = self.merge_boxes_dbscan(
                boxes, boxes_ids)
        else:
            merged_boxes, merged_ids = [], []

        detections = []
        crops = {}
        for box, ids in zip(merged_boxes, merged_ids):
            x1, y1, x2, y2 = map(int, box)
            for track_id in ids:
                detections.append(
                    {'bbox': (x1, y1, x2, y2), 'track_id': track_id})
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, CROP_SIZE)
                crops[track_id] = crop
        return detections, crops

    def processing_clips(self, clips: list):
        # Aggregates clip prediction and return action label
        labels_dict = defaultdict(int)
        for label in clips:
            if label == "fight":
                return "fight"
            labels_dict[label.lower()] += 1

        return "normal"

    def show_results(self):
        """
         Function that performs detection, clip buffering, action inference, and visualization.

         - Read frame from video
         - Run YOLO tracking
         - For each track ID, store cropped frames in a fixed-length queue
         - When the queue reaches `sample_length`, run `inference`
         - Accumulate clip-level predictions and determine the final label using `processing_clips`
         - Draw bounding boxes and predicted labels on the frame
         - Write annotated frame to the output video
         """
        frame_counter = 0
        id_bbox = {}
        last_results = {}
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_flag = True
                break

            frame_counter += 1

            yolo_results, yolo_crops = self.detect_with_yolo(frame)
            current_ids = set()
            for track_id, crop in yolo_crops.items():
                if self.frame_queues[track_id].qsize() == self.sample_length:
                    self.frame_queues[track_id].get()

                self.frame_queues[track_id].put_nowait(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            for det in yolo_results:
                x1, y1, x2, y2 = det['bbox']
                track_id = det['track_id']
                id_bbox[track_id] = det['bbox']
                current_ids.add(track_id)

            for track_id, queue in self.frame_queues.items():
                if queue.qsize() == self.sample_length:
                    ti = time.time()
                    result = self.inference(queue)
                    logging.debug('inference time: %.4f s', time.time() - ti)
                    if result[0][1] < CONFIDENCE_THRESHOLD:
                        result = [('NORMAL', result[0][1])]

                    self.clips[track_id].append(result[0][0])
                    if track_id not in last_results:
                        last_results[track_id] = None

            for track_id in current_ids:
                if track_id in last_results and track_id in id_bbox:
                    x1, y1, x2, y2 = id_bbox[track_id]

                    if len(self.clips[track_id]) == CLIPS_PER_DECISION:
                        selected_label = self.processing_clips(self.clips[track_id])
                        self.clips[track_id].clear()

                        last_results[track_id] = selected_label
                    elif last_results[track_id] is None:
                        continue
                    else:
                        selected_label = last_results[track_id]
                    if selected_label == "fight":
                        color = RED_COLOR
                    else:
                        color = FONTCOLOR
                    location = (x1, y1 - 25)
                    text = f'ID {track_id}: {selected_label}'
                    logging.debug(text)
                    cv2.putText(frame, text, location, FONTFACE,
                                FONTSCALE, color, THICKNESS, LINETYPE)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            self.out.write(frame)

        self.out.release()

    def inference(self, queue):
        """
        Performs action recognition on a sequence of frames stored in a queue.

        Args:
            queue (Queue): A queue containing frames used to form a clip.

        Returns:
            list[tuple[str, float]]:
                A list containing a tuple (label, score)
        """

        frames = list(queue.queue)

        if self.data['img_shape'] is None and frames:
            self.data['img_shape'] = frames[0].shape[:2]

        cur_data = self.data.copy()
        cur_data['imgs'] = frames
        cur_data = self.test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])

        results = [('NORMAL', 0.0)]
        try:
            with torch.no_grad():
                result = self.model.test_step(cur_data)[0]
            scores = np.array(result.pred_score.tolist())
            score_tuples = list(zip(self.label, scores))
            score_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
            num_selected_labels = min(len(self.label), 1)
            results = score_sorted[:num_selected_labels]

        except Exception as e:
            logging.error(f"Error during inference: {e}")

        num_to_remove = len(frames) // 2
        for _ in range(num_to_remove):
            queue.get()

        return results


def main():
    args = parse_args()
    processor = VideoProcessor(args)
    processor.show_results()


if __name__ == '__main__':
    main()
