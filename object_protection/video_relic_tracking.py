#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频文物跟踪系统
实时检测、选择和跟踪文物，保持目标ID
"""

import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from object_protection.mindyolo_adapter import MindYOLODetector, create_default_detector

EXCLUDED_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'hair drier', 'toothbrush'
}

HIGH_ANTIQUITY_CLASSES = {'bottle', 'wine glass', 'cup', 'bowl', 'vase', 'book', 'clock', 'scissors'}
MEDIUM_ANTIQUITY_CLASSES = {'teddy bear', 'potted plant'}

class SimpleTracker:
    """简单的目标跟踪器，基于质心距离的贪心匹配。"""

    def __init__(self, max_disappeared: int = 10, max_distance: float = 100.0):
        self.next_object_id = 0
        self.objects: Dict[int, Dict[str, object]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        """注册新目标"""
        self.objects[self.next_object_id] = {
            'centroid': tuple(centroid),
            'bbox': list(bbox),
            'last_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        """注销目标"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
    
    def update(self, detections: Sequence[Dict[str, object]]):
        """更新跟踪器"""
        if not detections:
            # 没有检测到目标，增加所有目标的消失计数
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_bboxes = [list(det['bbox']) for det in detections]
        input_centroids = np.array([
            ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            for bbox in input_bboxes
        ], dtype=np.float32)

        if not self.objects:
            for centroid, bbox in zip(input_centroids, input_bboxes):
                self.register(centroid, bbox)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array([
            self.objects[obj_id]['centroid'] for obj_id in object_ids
        ], dtype=np.float32)

        distance_matrix = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2,
        )

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row in np.argsort(distance_matrix.min(axis=1)):
            col = distance_matrix[row].argmin()
            if row in used_rows or col in used_cols:
                continue
            if distance_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id]['centroid'] = tuple(input_centroids[col])
            self.objects[object_id]['bbox'] = input_bboxes[col]
            self.objects[object_id]['last_seen'] = time.time()
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distance_matrix.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(distance_matrix.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col], input_bboxes[col])

        return self.objects

class VideoRelicTracker:
    def __init__(
        self,
        detector: MindYOLODetector,
        confidence_threshold: float = 0.1,
        window_name: Optional[str] = None,
    ):
        self.detector = detector
        self.tracker = SimpleTracker(max_disappeared=10)
        self.selected_relics = set()  # 选中的文物ID
        self.relic_detections = []  # 当前帧的文物检测结果
        self.tracked_objects = {}  # 跟踪的目标
        self.window_name = (
            window_name
            if window_name is not None
            else "文物跟踪系统 - 点击选择文物，按Enter确认，按ESC退出"
        )
        self.confidence_threshold = confidence_threshold
        self.class_names: Sequence[str] = getattr(self.detector, "class_names", ())

        # 创建窗口（在无图形界面环境下自动降级为无GUI模式）
        self.gui_enabled = True
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
        except cv2.error as exc:
            self.gui_enabled = False
            print(
                "警告: OpenCV 无法创建显示窗口，系统将以无GUI模式运行。"
                f" 详细信息: {exc}"
            )

    @staticmethod
    def _calculate_antiquity_score(
        *,
        area: float,
        confidence: float,
        class_name: str,
        bronze_ratio: float,
        edge_density: float,
    ) -> float:
        score = 0.0

        if area > 100000:
            score += 0.8
        elif area > 50000:
            score += 0.6
        elif area > 10000:
            score += 0.4
        else:
            score += 0.3

        if confidence > 0.8:
            score += 0.3
        elif confidence > 0.6:
            score += 0.2
        elif confidence > 0.4:
            score += 0.1

        if class_name in HIGH_ANTIQUITY_CLASSES:
            score += 0.4
        elif class_name in MEDIUM_ANTIQUITY_CLASSES:
            score += 0.2
        else:
            score += 0.1

        if bronze_ratio > 0.01 or edge_density > 0.05:
            score += 0.3

        return min(score, 1.0)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查点击是否在某个检测框内
            clicked_relic = self.get_clicked_relic(x, y)
            if clicked_relic is not None:
                if clicked_relic in self.selected_relics:
                    # 如果已选中，则取消选择
                    self.selected_relics.remove(clicked_relic)
                    print(f"取消选择文物 {clicked_relic}")
                else:
                    # 如果未选中，则选择
                    self.selected_relics.add(clicked_relic)
                    print(f"选择文物 {clicked_relic}")
    
    def get_clicked_relic(self, x, y):
        """获取点击的文物ID"""
        for detection in self.relic_detections:
            x1, y1, x2, y2 = detection['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return detection.get('track_id', None)
        return None
    
    def _detect_all_objects(self, frame: np.ndarray) -> List[Dict[str, object]]:
        """运行一次YOLO检测并返回所有检测结果。"""
        try:
            raw_detections = self.detector.detect(
                frame,
                conf_threshold=self.confidence_threshold,
            )
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent failure
            print(f"MindYOLO推理失败: {exc}")
            return []

        detections: List[Dict[str, object]] = []
        for det in raw_detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(det.get('class_id', -1))
            class_name = det.get('class_name')
            if not class_name and 0 <= class_id < len(self.class_names):
                class_name = str(self.class_names[class_id])
            if not class_name:
                class_name = f'class_{class_id}'

            area = det.get('area')
            if area is None:
                area = float(max(0, (x2 - x1) * (y2 - y1)))

            detections.append(
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(det.get('confidence', 0.0)),
                    'class_id': class_id,
                    'class_name': class_name,
                    'area': float(area),
                }
            )

        return detections

    def detect_relics(
        self,
        frame: np.ndarray,
        detections: Optional[Sequence[Dict[str, object]]] = None,
    ) -> List[Dict[str, object]]:
        """检测文物"""
        h, w = frame.shape[:2]

        # 分析图片特征
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bronze_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
        bronze_ratio = float(np.count_nonzero(bronze_mask)) / (h * w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (h * w)

        relic_detections: List[Dict[str, object]] = []
        if detections is None:
            detections = self._detect_all_objects(frame)

        for detection in detections:
            class_name = str(detection['class_name'])
            if class_name in EXCLUDED_CLASSES:
                continue

            class_id = int(detection['class_id'])
            confidence = float(detection['confidence'])
            x1, y1, x2, y2 = map(int, detection['bbox'])
            bbox_area = float(detection.get('area', max(0, (x2 - x1) * (y2 - y1))))
            score = self._calculate_antiquity_score(
                area=bbox_area,
                confidence=confidence,
                class_name=class_name,
                bronze_ratio=bronze_ratio,
                edge_density=edge_density,
            )

            if score < 0.3:
                continue

            relic_detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name,
                'area': bbox_area,
                'antiquity_score': score,
            })

        return relic_detections
    
    def update_tracking(self, detections):
        """更新跟踪"""
        # 更新跟踪器
        self.tracked_objects = self.tracker.update(detections)
        
        if not self.tracked_objects:
            for detection in detections:
                detection['track_id'] = None
            return

        tracked_ids = list(self.tracked_objects.keys())
        tracked_centroids = np.array([
            self.tracked_objects[track_id]['centroid'] for track_id in tracked_ids
        ], dtype=np.float32)

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            distances = np.linalg.norm(tracked_centroids - centroid, axis=1)

            best_index = distances.argmin()
            if distances[best_index] < 50:
                detection['track_id'] = tracked_ids[best_index]
            else:
                detection['track_id'] = None
    
    def draw_detections(self, frame):
        """绘制检测结果"""
        result_frame = frame.copy()
        
        for detection in self.relic_detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection.get('track_id', None)
            is_selected = track_id in self.selected_relics if track_id is not None else False
            
            # 确保坐标在图片范围内
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # 选择颜色和样式
            if is_selected:
                color = (0, 255, 0)  # 绿色
                thickness = 4
            else:
                color = (0, 0, 255)  # 红色
                thickness = 2
            
            # 绘制边界框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制跟踪ID
            if track_id is not None:
                cv2.putText(
                    result_frame, 
                    f"ID:{track_id}", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color, 
                    2
                )
            
            # 如果被选中，绘制红色电子栅栏
            if is_selected:
                fence_info = self.calculate_safety_fence([x1, y1, x2, y2], frame.shape)
                fx1, fy1, fx2, fy2 = fence_info['fence_bbox']
                cv2.rectangle(result_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
        
        return result_frame
    
    def calculate_safety_fence(self, relic_bbox, frame_shape, safety_margin=0.3):
        """计算文物的安全栅栏范围"""
        x1, y1, x2, y2 = relic_bbox
        h, w = frame_shape[:2]
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        relic_width = x2 - x1
        relic_height = y2 - y1
        
        safety_width = int(relic_width * (1 + safety_margin * 2))
        safety_height = int(relic_height * (1 + safety_margin * 2))
        
        fence_x1 = max(0, center_x - safety_width // 2)
        fence_y1 = max(0, center_y - safety_height // 2)
        fence_x2 = min(w, center_x + safety_width // 2)
        fence_y2 = min(h, center_y + safety_height // 2)
        
        return {
            'fence_bbox': [fence_x1, fence_y1, fence_x2, fence_y2],
            'relic_center': [center_x, center_y],
            'safety_margin': safety_margin,
            'fence_area': (fence_x2 - fence_x1) * (fence_y2 - fence_y1)
        }
    
    def process_video(self, video_source=0):
        """处理视频"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return
        
        print("=== 视频文物跟踪系统 ===")
        print("操作说明:")
        print("1. 点击红色框选择文物")
        print("2. 点击绿色框取消选择")
        print("3. 按Enter键确认选择")
        print("4. 按ESC键退出")
        print("5. 按S键保存当前帧")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测文物
            all_detections = self._detect_all_objects(frame)
            self.relic_detections = self.detect_relics(frame, all_detections)

            # 更新跟踪
            self.update_tracking(self.relic_detections)
            
            # 绘制检测结果
            result_frame = self.draw_detections(frame)
            
            # 显示状态信息
            status_text = f"已选择: {len(self.selected_relics)} 个文物"
            cv2.putText(
                result_frame, 
                status_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            
            # 显示帧数
            cv2.putText(
                result_frame, 
                f"Frame: {frame_count}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # 显示图片
            cv2.imshow(self.window_name, result_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
            elif key == 13:  # Enter键
                print(f"确认选择 {len(self.selected_relics)} 个文物")
            elif key == ord('s') or key == ord('S'):  # S键保存
                filename = f"tracking_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"保存帧到: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='视频文物跟踪系统')
    parser.add_argument('--source', type=str, default='0', help='视频源 (0=摄像头, 或视频文件路径)')
    parser.add_argument('--conf', type=float, default=0.1, help='置信度阈值')
    
    args = parser.parse_args()
    
    print("=== 视频文物跟踪系统 ===")
    print("实时检测、选择和跟踪文物")
    
    detector = create_default_detector(conf_threshold=args.conf)
    if detector is None:
        return

    tracker = VideoRelicTracker(detector, confidence_threshold=args.conf)
    
    # 处理视频
    try:
        video_source = int(args.source) if args.source.isdigit() else args.source
        tracker.process_video(video_source)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"处理视频时出错: {e}")

if __name__ == "__main__":
    main()
