import torch
import numpy as np
import cv2
import os
import folder_paths
from ultralytics import YOLO

class YoloTrackNode:
    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "fps": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "yolo_model": (["yolov8x-seg.pt", "yolov8l-seg.pt",  "yolov8m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"],),
                "sort_direction": (["left-to-right", "right-to-left"],),
                "target_index": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],),
                "conf_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "sim_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("vis_frames", "mask_frames", "fps")
    FUNCTION = "track_objects"
    CATEGORY = "Tracking"

    def get_color_histogram(self, image, mask_polygon):
        """提取HSV直方图特征"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(mask_polygon, dtype=np.int32)], 255)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def initialize_target(self, frame, results, model, sort_direction, target_index):
        """初始化目标"""
        if results[0].masks is None:
            return None, False

        candidates = []
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            if model.names[cls_id] != 'person':
                continue
            
            x_center = box.xywh[0][0].item()
            mask_poly = results[0].masks.xy[i]
            
            if len(mask_poly) == 0: continue

            candidates.append({
                'index': i,
                'x': x_center,
                'poly': mask_poly
            })
        
        if not candidates:
            return None, False

        is_reverse = (sort_direction == 'right-to-left')
        candidates.sort(key=lambda c: c['x'], reverse=is_reverse)

        if target_index < len(candidates):
            selected = candidates[target_index]
            target_hist = self.get_color_histogram(frame, selected['poly'])
            return target_hist, True
        
        return None, False

    def track_objects(self, images, fps, yolo_model, sort_direction, target_index, conf_threshold, sim_threshold):
        
        yolo_dir = os.path.join(folder_paths.models_dir, "yolo")
        
        if not os.path.exists(yolo_dir):
            os.makedirs(yolo_dir)
            print(f"Created directory: {yolo_dir}")

        model_path = os.path.join(yolo_dir, yolo_model)

        if self.model is None or self.current_model_name != yolo_model:
            print(f"Loading YOLO model from: {model_path}")
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                print(f"Warning: Model not found at {model_path}.")
                print("Attempting to download via Ultralytics (will save to default cache)...")
                try:
                    self.model = YOLO(yolo_model) 
                except Exception as e:
                    raise FileNotFoundError(f"无法加载模型，请确保 '{yolo_model}' 存在于 '{yolo_dir}' 目录下。错误: {e}")

            self.current_model_name = yolo_model

        input_frames_np = []
        for img in images:
            i = 255. * img.cpu().numpy()
            img_np = np.clip(i, 0, 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            input_frames_np.append(img_np)

        vis_frames = []
        mask_frames = []
        
        target_hist = None
        target_found = False
        
        for idx, frame in enumerate(input_frames_np):
            vis_frame = frame.copy()
            mask_frame = np.zeros_like(frame)

            # 开启 retina_masks=True 以获得高质量边缘
            results = self.model(frame, verbose=False, conf=conf_threshold, retina_masks=True)

            target_poly = None

            if not target_found:
                target_hist, found = self.initialize_target(frame, results, self.model, sort_direction, target_index)
                if found:
                    target_found = True
                    # print(f"Frame {idx}: Target initialized.")

            elif results[0].masks is not None:
                best_score = -1
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    if self.model.names[cls_id] != 'person':
                        continue
                    
                    poly = results[0].masks.xy[i]
                    if len(poly) == 0: continue

                    current_hist = self.get_color_histogram(frame, poly)
                    score = cv2.compareHist(target_hist, current_hist, cv2.HISTCMP_CORREL)
                    
                    if score > best_score:
                        best_score = score
                        target_poly = poly
                
                if best_score < sim_threshold:
                    target_poly = None
            
            if target_poly is not None:
                pts = np.array(target_poly, dtype=np.int32)
                
                temp_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(temp_mask, [pts], 255)
                
                temp_mask = cv2.GaussianBlur(temp_mask, (9, 9), 0)
                _, temp_mask = cv2.threshold(temp_mask, 127, 255, cv2.THRESH_BINARY)
                
                mask_rgb = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR)
                mask_frame = mask_rgb

                mask_indices = temp_mask == 255
                vis_frame[mask_indices] = vis_frame[mask_indices] * 0.5 + np.array([0, 255, 0]) * 0.5
                contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)

            vis_frames.append(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            mask_frames.append(cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB))

        vis_tensor = np.array(vis_frames).astype(np.float32) / 255.0
        vis_tensor = torch.from_numpy(vis_tensor)
        
        mask_tensor = np.array(mask_frames).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_tensor)

        return (vis_tensor, mask_tensor, fps)
    
NODE_CLASS_MAPPINGS = {
    "YoloTrackNode": YoloTrackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloTrackNode": "YOLO Person Tracker"
}