# -*- coding: utf-8 -*-
"""
高尔夫挥杆分析模块

本模块提供高尔夫挥杆视频分析功能，包括：
1. 人体姿态检测和关键点提取
2. 高尔夫挥杆动作识别（准备、上杆顶点、击球、收杆）
3. 关键帧提取和分析
4. 角度计算和可视化
5. 结果视频生成

主要功能：
- analyze_golf_swing: 主要分析函数，处理整个视频分析流程
- 各种角度计算函数：计算身体各部位的角度
- 动作识别函数：识别不同的高尔夫挥杆阶段
- 可视化函数：在图像上绘制关键点、骨架和角度

依赖：
- OpenCV: 图像处理和视频操作
- Ultralytics YOLO: 人体姿态检测
- NumPy: 数值计算
- PyTorch: 深度学习框架
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import PoseModel
import json
import math
import os
from collections import deque
import threading
import datetime
from golfswingsAssistant import assistant_answer, batch_assistant_analysis

from config import GolfAnalysisConfig


def analyze_golf_swing(input_video_path, output_dir, progress_callback=None):
    """
    分析高尔夫挥杆视频
    
    Args:
        input_video_path (str): 输入视频文件路径
        output_dir (str): 输出目录路径
        progress_callback (callable, optional): 进度回调函数
        
    Returns:
        dict: 分析结果，包含错误信息或成功状态
    """
    cap = None  # 在函数开头初始化
    out = None  # 初始化视频写入器变量

    try:
        # 配置部分
        VIDEO_PATH = input_video_path

        # 检查输入视频是否存在
        if not os.path.exists(VIDEO_PATH):
            error_msg = f"Input video file not found: {VIDEO_PATH}"
            print(error_msg)
            return {'error': error_msg}

        # 获取输入视频的文件名和扩展名
        input_filename = os.path.basename(input_video_path)
        filename, ext = os.path.splitext(input_filename)

        # 确保扩展名有效
        if not ext:
            ext = ".mp4"

        # 创建输出目录
        result_video_dir = os.path.join(output_dir, 'result_video')
        keyframe_dir = os.path.join(output_dir, 'key_frames')
        os.makedirs(result_video_dir, exist_ok=True)
        os.makedirs(keyframe_dir, exist_ok=True)

        # 输出视频文件路径
        if filename.startswith("analyzed_"):
            output_video_name = f"{filename}{ext}"
        else:
            output_video_name = f"analyzed_{filename}{ext}"
        OUTPUT_VIDEO_PATH = os.path.join(result_video_dir, output_video_name)

        # 检查输出目录是否可写
        test_file = os.path.join(result_video_dir, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            error_msg = f"Output directory not writable: {result_video_dir} - {str(e)}"
            print(error_msg)
            return {'error': error_msg}

        # YOLO模型路径
        MODEL_PATH = r"best.pt"

        # 检查模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model file not found: {MODEL_PATH}"
            print(error_msg)
            return {'error': error_msg}

        # 打印路径信息
        print(f"Input video: {VIDEO_PATH}")
        print(f"Output video: {OUTPUT_VIDEO_PATH}")
        print(f"Output keyframes: {keyframe_dir}")

        # 使用配置常量
        CONFIDENCE_THRESHOLD = GolfAnalysisConfig.CONFIDENCE_THRESHOLD
        SHOW_KEYPOINTS = GolfAnalysisConfig.SHOW_KEYPOINTS
        SHOW_SKELETON = GolfAnalysisConfig.SHOW_SKELETON
        SHOW_ANGLES = GolfAnalysisConfig.SHOW_ANGLES
        ANGLE_COLORS = GolfAnalysisConfig.ANGLE_COLORS
        ANGLE_SHORT_NAMES = GolfAnalysisConfig.ANGLE_SHORT_NAMES

        # 关键点索引定义
        KEYPOINT_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "middle_club", "head_club",
            "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        # 创建名称到索引的映射字典
        kp_name_to_index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

        # 关键点索引常量
        RIGHT_SHOULDER_IDX = kp_name_to_index["right_shoulder"]
        RIGHT_ELBOW_IDX = kp_name_to_index["right_elbow"]
        RIGHT_WRIST_IDX = kp_name_to_index["right_wrist"]
        RIGHT_HIP_IDX = kp_name_to_index["right_hip"]
        RIGHT_KNEE_IDX = kp_name_to_index["right_knee"]
        RIGHT_ANKLE_IDX = kp_name_to_index["right_ankle"]
        LEFT_SHOULDER_IDX = kp_name_to_index["left_shoulder"]
        LEFT_ELBOW_IDX = kp_name_to_index["left_elbow"]
        LEFT_WRIST_IDX = kp_name_to_index["left_wrist"]
        LEFT_HIP_IDX = kp_name_to_index["left_hip"]
        LEFT_KNEE_IDX = kp_name_to_index["left_knee"]
        LEFT_ANKLE_IDX = kp_name_to_index["left_ankle"]
        MIDDLE_CLUB_IDX = kp_name_to_index["middle_club"]
        HEAD_CLUB_IDX = kp_name_to_index["head_club"]

        # 骨架连线定义
        SKELETON = [
            (kp_name_to_index["left_ankle"], kp_name_to_index["left_knee"], "left_ankle_to_left_knee"),
            (kp_name_to_index["left_knee"], kp_name_to_index["left_hip"], "left_knee_to_left_hip"),
            (kp_name_to_index["right_ankle"], kp_name_to_index["right_knee"], "right_ankle_to_right_knee"),
            (kp_name_to_index["right_knee"], kp_name_to_index["right_hip"], "right_knee_to_right_hip"),
            (kp_name_to_index["left_hip"], kp_name_to_index["right_hip"], "left_hip_to_right_hip"),
            (kp_name_to_index["left_shoulder"], kp_name_to_index["left_hip"], "left_shoulder_to_left_hip"),
            (kp_name_to_index["right_shoulder"], kp_name_to_index["right_hip"], "right_shoulder_to_right_hip"),
            (kp_name_to_index["left_shoulder"], kp_name_to_index["right_shoulder"], "left_shoulder_to_right_shoulder"),
            (kp_name_to_index["left_shoulder"], kp_name_to_index["left_elbow"], "left_shoulder_to_left_elbow"),
            (kp_name_to_index["right_shoulder"], kp_name_to_index["right_elbow"], "right_shoulder_to_right_elbow"),
            (kp_name_to_index["left_elbow"], kp_name_to_index["left_wrist"], "left_elbow_to_left_wrist"),
            (kp_name_to_index["right_elbow"], kp_name_to_index["right_wrist"], "right_elbow_to_right_wrist"),
            (kp_name_to_index["middle_club"], kp_name_to_index["head_club"], "middle_club_to_head_club"),
            (kp_name_to_index["left_wrist"], kp_name_to_index["middle_club"], "left_wrist_to_middle_club"),
            (kp_name_to_index["left_shoulder"], kp_name_to_index["left_ankle"], "left_shoulder_to_left_ankle"),
            (kp_name_to_index["right_shoulder"], kp_name_to_index["left_ankle"], "right_shoulder_to_left_ankle"),
        ]

        # 角度定义
        ANGLE_DEFINITIONS = {
            "right_hip_angle": (RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX, RIGHT_KNEE_IDX),
            "right_knee_angle": (RIGHT_HIP_IDX, RIGHT_KNEE_IDX, RIGHT_ANKLE_IDX),
            "right_elbow_angle": (RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX),
            "right_shoulder_angle": (RIGHT_ELBOW_IDX, RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX),
            "left_hip_angle": (LEFT_SHOULDER_IDX, LEFT_HIP_IDX, LEFT_KNEE_IDX),
            "left_knee_angle": (LEFT_HIP_IDX, LEFT_KNEE_IDX, LEFT_ANKLE_IDX),
            "left_elbow_angle": (LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, LEFT_WRIST_IDX),
            "left_shoulder_angle": (LEFT_ELBOW_IDX, LEFT_SHOULDER_IDX, LEFT_HIP_IDX),
            "club_angle": (LEFT_WRIST_IDX, MIDDLE_CLUB_IDX, HEAD_CLUB_IDX),
            "left_wrist_angle": (LEFT_ELBOW_IDX, LEFT_WRIST_IDX, MIDDLE_CLUB_IDX)
        }

        # 角度方向配置
        # 格式：角度名称: (参考向量, 旋转方向)
        ANGLE_DIRECTIONS = {
            "right_hip_angle": ('horizontal', 'ccw'), "right_knee_angle": ('vertical', 'ccw'), # 垂直参考，逆时针为正
            "right_elbow_angle": ('horizontal', 'ccw'), "right_shoulder_angle": ('horizontal', 'ccw'), # 水平参考，逆时针为正
            "left_hip_angle": ('horizontal', 'cw'), "left_knee_angle": ('vertical', 'cw'), # 水平参考，顺时针为正
            "left_elbow_angle": ('horizontal', 'cw'), "left_shoulder_angle": ('horizontal', 'cw'), # 水平参考，顺时针为正
            "club_angle": ('horizontal', 'ccw'), "left_wrist_angle": ('horizontal', 'ccw') # 水平参考，逆时针为正
        }

        # 初始化YOLO模型
        try:
            torch.serialization.add_safe_globals([PoseModel])
            model = YOLO(MODEL_PATH)
            print(f"Successfully loaded model: {MODEL_PATH}")
        except Exception as e:
            error_msg = f"Error loading YOLO model: {e}"
            print(error_msg)
            return {'error': error_msg}

        # 打开视频文件
        try:
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                error_msg = f"Error opening video file: {VIDEO_PATH}"
                print(error_msg)
                return {'error': error_msg}
        except Exception as e:
            error_msg = f"Error opening video file: {str(e)}"
            print(error_msg)
            return {'error': error_msg}

        # 获取视频信息
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
        fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数

        # 打印视频信息
        print(f"Video Info: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} Frames")

        # 检查帧尺寸是否有效
        if frame_width <= 0 or frame_height <= 0:
            error_msg = f"Invalid frame dimensions: {frame_width}x{frame_height}"
            print(error_msg)
            cap.release()
            return {'error': error_msg}

        # 尝试多种编解码器
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264 (优先使用浏览器兼容编码)
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
            cv2.VideoWriter_fourcc(*'X264'),  # H.264/AVC 
            cv2.VideoWriter_fourcc(*'H264'),  # 另一种H.264表示
        ]

        out = None
        used_codec = None
        for codec in fourcc_options:
            try:
                # 尝试创建视频写入器
                out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, codec, fps, (frame_width, frame_height))

                # 检查是否成功打开
                if out.isOpened():
                    used_codec = codec
                    print(f"Successfully initialized VideoWriter with codec: {codec}")
                    break
                else:
                    # 关闭失败的写入器
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                print(f"Error with codec {codec}: {str(e)}")
                if out:
                    out.release()
                out = None

        # 如果所有编解码器都失败
        if out is None or not out.isOpened():
            error_msg = f"Failed to initialize VideoWriter for path: {OUTPUT_VIDEO_PATH}\n"
            error_msg += f"Tried codecs: {fourcc_options}\n"
            error_msg += f"Frame size: {frame_width}x{frame_height}, FPS: {fps}"
            print(error_msg)
            cap.release()
            return {'error': error_msg}

        # 打印输出路径信息
        print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")

        # 存储每帧结果的列表
        all_frame_data = []

        # 绘图颜色配置
        POINT_COLOR = GolfAnalysisConfig.POINT_COLOR
        LINE_COLOR = GolfAnalysisConfig.LINE_COLOR
        OFFSET_PX = GolfAnalysisConfig.OFFSET_PX

        # 新版本两阶段检测模式
        # 第一阶段：收集所有帧数据
        preparation_list = []  # 准备动作帧编号list
        top_list = []          # 上杆顶点动作帧编号list
        impact_list = []       # 击球动作帧编号list
        finish_list = []       # 完成动作帧编号list
        state = 'Preparation'  # 当前动作状态
        detected_states = []
        key_frames = {
            "Preparation": None, "Top of Backswing": None, "Impact": None, "Finish": None
        }
        
        # 前置判断标志
        preparation_detected = False

        # 位移阈值和窗口大小
        DISPLACEMENT_THRESHOLD = GolfAnalysisConfig.DISPLACEMENT_THRESHOLD
        WINDOW_SIZE = GolfAnalysisConfig.WINDOW_SIZE

        # 存储历史位置（用于球杆移动检测）
        head_club_history = deque(maxlen=WINDOW_SIZE)  # 球杆头位置历史
        middle_club_history = deque(maxlen=WINDOW_SIZE)  # 球杆中点位置历史

        # 文本样式配置
        TEXT_FONT = GolfAnalysisConfig.TEXT_FONT
        TEXT_SCALE = GolfAnalysisConfig.TEXT_SCALE
        TEXT_THICKNESS = GolfAnalysisConfig.TEXT_THICKNESS
        TEXT_COLOR = GolfAnalysisConfig.TEXT_COLOR
        TEXT_BG_COLOR = GolfAnalysisConfig.TEXT_BG_COLOR

        # 存储差异总结
        differences_summary = []  # 用于存储姿势判定差异结果

        # 函数定义部分

        def calculate_angle_between_points(p1, p2, p3, angle_name=None):
            """计算三个点构成的角度（带方向）"""
            # 计算向量
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)

            # 计算向量模长
            magnitude_v1 = np.linalg.norm(v1)
            magnitude_v2 = np.linalg.norm(v2)

            # 检查模长是否为零
            if magnitude_v1 == 0 or magnitude_v2 == 0:
                return None  # 无法计算角度

            # 计算点积和叉积
            dot_product = np.dot(v1, v2)
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]

            # 计算无方向角度（0-180度）
            cos_angle = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)

            # 如果没有指定角度名称，返回无方向角度
            if angle_name is None:
                return angle_deg

            # 获取该角度的正方向配置
            reference_vector, rotation_direction = ANGLE_DIRECTIONS.get(angle_name, ('horizontal', 'ccw'))

            # 确定参考向量
            if reference_vector == 'horizontal':
                ref_vec = np.array([1, 0])  # 水平向右
            else:  # vertical
                ref_vec = np.array([0, -1])  # 垂直向上（图像坐标系中y轴向下为正）

            # 计算v1相对于参考向量的角度
            dot_ref = np.dot(v1, ref_vec)
            cross_ref = v1[0] * ref_vec[1] - v1[1] * ref_vec[0]
            angle_from_ref = np.degrees(np.arctan2(cross_ref, dot_ref))
            angle_from_ref = angle_from_ref % 360  # 确保在0-360度范围内

            # 根据旋转方向确定最终角度
            if rotation_direction == 'ccw':  # 逆时针为正
                if cross_product < 0:  # 叉积为负表示v2在v1的逆时针方向
                    final_angle = (angle_from_ref + angle_deg) % 360
                else:
                    final_angle = (angle_from_ref - angle_deg) % 360
            else:  # cw，顺时针为正
                if cross_product > 0:  # 叉积为正表示v2在v1的顺时针方向
                    final_angle = (angle_from_ref + angle_deg) % 360
                else:
                    final_angle = (angle_from_ref - angle_deg) % 360

            # 四舍五入到小数点后一位
            if final_angle is not None:
                final_angle = round(final_angle, 1)

            return final_angle

        def calculate_vector_angle(v1, v2):
            """计算两个向量之间的角度（0-360度）"""
            # 计算点积和行列式
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            # 计算角度（弧度）
            angle_rad = np.arctan2(det, dot)
            # 转换为角度
            angle_deg = np.degrees(angle_rad)
            # 确保角度在0-360度范围内
            return angle_deg if angle_deg >= 0 else angle_deg + 360

        def draw_angle_arc(image, p1, p2, p3, angle_value, color, radius=20):
            """在图像上绘制角度圆弧"""
            if angle_value is None:
                return  # 无角度值则不绘制

            # 计算向量
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)

            # 计算向量相对于x轴的角度
            angle1 = calculate_vector_angle([1, 0], v1)
            angle2 = calculate_vector_angle([1, 0], v2)

            # 确定起始和结束角度
            start_angle = min(angle1, angle2)
            end_angle = max(angle1, angle2)

            # 如果角度差大于180度，绘制较小的圆弧
            if end_angle - start_angle > 180:
                start_angle, end_angle = end_angle, start_angle + 360

            # 绘制圆弧
            center = (int(p2[0]), int(p2[1]))  # 圆心位置（顶点）
            cv2.ellipse(image, center, (radius, radius), 0, start_angle, end_angle, color, 2)

            # 在圆弧中点显示角度值（只显示整数部分）
            mid_angle = (start_angle + end_angle) / 2  # 计算中点角度
            # 计算文本位置
            text_x = int(p2[0] + radius * np.cos(np.radians(mid_angle)))
            text_y = int(p2[1] + radius * np.sin(np.radians(mid_angle)))
            # 绘制角度值文本
            cv2.putText(image, f"{int(angle_value)}°", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        def draw_pose_on_image(image, keypoints, confidences, angles):
            """在图像上绘制关键点、骨架和角度"""
            # 绘制骨架连线
            for p1_idx, p2_idx, connection_name in SKELETON:
                # 检查是否应该显示该连线
                if not SHOW_SKELETON.get(connection_name, True):
                    continue  # 跳过不显示的连线

                # 检查关键点置信度
                if confidences[p1_idx] > CONFIDENCE_THRESHOLD and confidences[p2_idx] > CONFIDENCE_THRESHOLD:
                    p1 = tuple(map(int, keypoints[p1_idx]))  # 转换为整数坐标
                    p2 = tuple(map(int, keypoints[p2_idx]))
                    # 确保坐标在图像范围内
                    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                        cv2.line(image, p1, p2, LINE_COLOR, 2)  # 绘制连线

            # 绘制关键点
            for i, kp in enumerate(keypoints):
                kp_name = KEYPOINT_NAMES[i]
                # 检查是否应该显示该关键点
                if not SHOW_KEYPOINTS.get(kp_name, True):
                    continue  # 跳过不显示的关键点

                # 检查置信度
                if confidences[i] > CONFIDENCE_THRESHOLD:
                    center = tuple(map(int, kp))  # 转换为整数坐标
                    # 确保坐标在图像范围内
                    if center[0] > 0 and center[1] > 0:
                        cv2.circle(image, center, 5, POINT_COLOR, -1)  # 绘制实心圆

            # 绘制所有角度和圆弧
            for angle_name, (idx1, idx2, idx3) in ANGLE_DEFINITIONS.items():
                # 检查是否应该显示该角度
                if not SHOW_ANGLES.get(angle_name, True):
                    continue  # 跳过不显示的角度

                angle_value = angles.get(angle_name)  # 获取角度值
                if angle_value is not None:
                    # 获取三个点
                    p1 = keypoints[idx1]
                    p2 = keypoints[idx2]  # 顶点
                    p3 = keypoints[idx3]

                    # 检查置信度
                    if (confidences[idx1] > CONFIDENCE_THRESHOLD and
                            confidences[idx2] > CONFIDENCE_THRESHOLD and
                            confidences[idx3] > CONFIDENCE_THRESHOLD):
                        # 获取该角度的颜色
                        color = ANGLE_COLORS.get(angle_name, (0, 255, 255))  # 默认为黄色
                        # 绘制角度圆弧
                        draw_angle_arc(image, p1, p2, p3, angle_value, color)
                        # 使用缩写名称以减少重叠
                        short_name = ANGLE_SHORT_NAMES.get(angle_name, angle_name.split('_')[0])
                        # 计算偏移位置（沿y轴向下偏移）
                        offset_x = 0
                        offset_y = OFFSET_PX
                        # 绘制角度名称
                        cv2.putText(image, short_name,
                                    (int(p2[0]) + offset_x, int(p2[1]) + offset_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        def calculate_distance(p1, p2):
            """计算两点之间的欧氏距离"""
            return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        # 新版本严格条件筛选函数
        def strict_preparation_metric(kps, _=None):
            """Preparation严格条件：左手腕y>髋关节平均y，且左手腕角度、左手肘角度、球杆角度都在250~280度"""
            try:
                lw_y = kps[LEFT_WRIST_IDX][1]
                lh_y = kps[LEFT_HIP_IDX][1]
                rh_y = kps[RIGHT_HIP_IDX][1]
                hip_avg_y = (lh_y + rh_y) / 2
                
                lw_angle = calculate_angle_between_points(
                    kps[LEFT_ELBOW_IDX], kps[LEFT_WRIST_IDX], kps[MIDDLE_CLUB_IDX], 'left_wrist_angle')
                le_angle = calculate_angle_between_points(
                    kps[LEFT_SHOULDER_IDX], kps[LEFT_ELBOW_IDX], kps[LEFT_WRIST_IDX], 'left_elbow_angle')
                club_angle = calculate_angle_between_points(
                    kps[LEFT_WRIST_IDX], kps[MIDDLE_CLUB_IDX], kps[HEAD_CLUB_IDX], 'club_angle')
                
                if lw_angle is None or le_angle is None or club_angle is None:
                    return False, 0
                
                avg_angle = (lw_angle + le_angle + club_angle) / 3
                is_strict = (lw_y > hip_avg_y and
                             250 <= lw_angle <= 280 and
                             250 <= le_angle <= 280 and
                             250 <= club_angle <= 280)
                return is_strict, avg_angle
            except:
                return False, 0

        def strict_top_metric(kps, _=None):
            """Top严格条件：左手腕角度<50，球杆与水平面夹角<20"""
            try:
                lw_angle = calculate_angle_between_points(
                    kps[LEFT_ELBOW_IDX], kps[LEFT_WRIST_IDX], kps[MIDDLE_CLUB_IDX], 'left_wrist_angle')
                club_hori_angle = calculate_club_horizontal_angle(
                    kps[MIDDLE_CLUB_IDX], kps[HEAD_CLUB_IDX])
                
                if lw_angle is None:
                    return False, 0
                
                club_hori_angle_abs = club_hori_angle if club_hori_angle < 90 else abs(180-club_hori_angle)
                is_strict = (lw_angle < 50 and club_hori_angle_abs < 20)
                return is_strict, lw_angle
            except:
                return False, 0

        def strict_impact_metric(kps, center_value=90):
            """Impact严格条件：球杆中点y>髋关节平均y，左髋-左脚踝连线与水平线夹角接近90度"""
            try:
                mc_y = kps[MIDDLE_CLUB_IDX][1]
                lh_y = kps[LEFT_HIP_IDX][1]
                rh_y = kps[RIGHT_HIP_IDX][1]
                hip_avg_y = (lh_y + rh_y) / 2
                
                left_hip = kps[LEFT_HIP_IDX]
                left_ankle = kps[LEFT_ANKLE_IDX]
                hori_angle = calculate_line_horizontal_angle(left_hip, left_ankle)
                
                is_strict = (mc_y > hip_avg_y and abs(hori_angle - center_value) <= 10)
                lw_x = kps[LEFT_WRIST_IDX][0]
                la_x = kps[LEFT_ANKLE_IDX][0]
                diff_x = abs(lw_x - la_x)
                return is_strict, diff_x
            except:
                return False, 0

        def strict_finish_metric_v2(kps, _=None):
            """Finish严格条件：球杆x<右肩x且左肘x<左肩x时为严格，否则为非严格"""
            try:
                mc_x = kps[MIDDLE_CLUB_IDX][0]
                mc_y = kps[MIDDLE_CLUB_IDX][1]
                rs_x = kps[RIGHT_SHOULDER_IDX][0]
                le_x = kps[LEFT_ELBOW_IDX][0]
                ls_x = kps[LEFT_SHOULDER_IDX][0]
                
                club1_angle = calculate_angle_between_points(
                    kps[LEFT_WRIST_IDX], kps[MIDDLE_CLUB_IDX], kps[HEAD_CLUB_IDX], 'club_angle')
                
                if mc_x < rs_x and le_x < ls_x:
                    return True, club1_angle if club1_angle is not None else 0
                else:
                    return False, mc_y
            except:
                return False, 0

        def select_best_and_judge(frame_list, keypoints_per_frame, strict_func, center_value=None, mode=None):
            """通用严格条件筛选和判定函数"""
            strict_list = []  # 满足严格条件的帧 (idx, metric)
            loose_list = []   # 不满足严格条件的帧 (idx, metric)
            
            for idx in frame_list:
                if idx >= len(keypoints_per_frame) or keypoints_per_frame[idx] is None:
                    continue
                kps = keypoints_per_frame[idx]
                is_strict, metric = strict_func(kps)
                if is_strict:
                    strict_list.append((idx, metric))
                else:
                    loose_list.append((idx, metric))
            
            if mode == 'finish':
                if strict_list:
                    # 有严格帧，选CLUB1角度最大的帧
                    best_idx, _ = max(strict_list, key=lambda x: x[1])
                    return best_idx, 'GOOD'
                elif loose_list:
                    # 没有严格帧，选middle_club.y最小的帧
                    best_idx, _ = min(loose_list, key=lambda x: x[1])
                    return best_idx, 'BAD'
                else:
                    return None, 'BAD'
            else:
                if strict_list:
                    # 有多个满足严格条件的帧，选择最接近目标值的帧
                    if center_value is not None:
                        best_idx, _ = min(strict_list, key=lambda x: abs(x[1] - center_value))
                    else:
                        best_idx, _ = strict_list[0]
                    return best_idx, 'GOOD'
                elif loose_list and center_value is not None:
                    best_idx, _ = min(loose_list, key=lambda x: abs(x[1] - center_value))
                    return best_idx, 'BAD'
                elif loose_list:
                    best_idx, _ = loose_list[0]
                    return best_idx, 'BAD'
                else:
                    return None, 'BAD'

        def calculate_club_horizontal_angle(mid_club, head_club):
            """计算球杆与水平面的夹角 (0-180度)"""
            dx = head_club[0] - mid_club[0]  # x方向差值
            dy = head_club[1] - mid_club[1]  # y方向差值

            # 计算与水平线的夹角 (0-90度)
            angle_rad = math.atan2(abs(dy), abs(dx))
            angle_deg = math.degrees(angle_rad)

            # 根据方向确定最终角度
            if dy < 0:  # 球杆向上
                return angle_deg
            else:  # 球杆向下
                return 180 - angle_deg

        def calculate_line_horizontal_angle(p1, p2):
            """计算两点连线与水平面的夹角"""
            dx = p2[0] - p1[0]  # x方向差值
            dy = p2[1] - p1[1]  # y方向差值
            return math.degrees(math.atan2(abs(dy), abs(dx)))  # 计算角度

        # 保存关键帧并进行姿势判定
        def save_and_evaluate_keyframe(frame_idx, frame, keypoints, confidences, angles, action):
            """
            保存关键帧图像并进行姿势判定
            
            frame_idx (int): 帧索引
            frame (numpy.ndarray): 原始帧图像
            keypoints (list): 关键点坐标列表
            confidences (list): 关键点置信度列表
            angles (dict): 角度值字典
            action (str): 动作名称
            """
            # 创建带标注的帧副本
            annotated_frame = frame.copy()
            # 在图像上绘制姿态和角度
            draw_pose_on_image(annotated_frame, keypoints, confidences, angles)

            # 在顶部显示动作名称
            text = action  # 动作名称
            # 计算文本尺寸
            text_size = cv2.getTextSize(text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
            text_x = int((frame_width - text_size[0]) / 2)  # 水平居中
            text_y = int(text_size[1] * 1.5)  # 垂直位置

            # 绘制背景矩形
            cv2.rectangle(annotated_frame,
                          (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10),
                          TEXT_BG_COLOR, -1)  # -1表示填充矩形

            # 绘制文本
            cv2.putText(annotated_frame, text, (text_x, text_y),
                        TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

            # 在下方显示判定信息
            y_offset = text_y + 50  # 起始y位置
            line_height = 40  # 行高

            # 存储当前动作的判定结果
            action_results = []

            # 根据不同的动作进行不同的姿势判定
            if action == "Preparation":  # 预备动作
                # 条件1: 左手肘、左手腕、球杆中点是否在一条直线上
                if (confidences[LEFT_ELBOW_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[LEFT_WRIST_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD):
                    # 获取关键点坐标
                    elbow = keypoints[LEFT_ELBOW_IDX]
                    wrist = keypoints[LEFT_WRIST_IDX]
                    club_mid = keypoints[MIDDLE_CLUB_IDX]

                    # 计算向量
                    vec1 = [elbow[0] - wrist[0], elbow[1] - wrist[1]]  # 手腕到肘部向量
                    vec2 = [club_mid[0] - wrist[0], club_mid[1] - wrist[1]]  # 手腕到球杆中点向量

                    # 计算夹角
                    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]  # 点积
                    magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)  # 向量1模长
                    magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)  # 向量2模长

                    if magnitude1 > 0 and magnitude2 > 0:
                        # 计算余弦值
                        cos_theta = dot_product / (magnitude1 * magnitude2)
                        # 计算角度（确保余弦值在有效范围内）
                        angle = math.degrees(math.acos(max(min(cos_theta, 1), -1)))

                        # 判定是否接近180度（允许5度偏差）
                        deviation = abs(angle - 180)  # 计算偏差
                        is_correct = deviation <= 5  # 是否满足条件
                        color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

                        # 生成状态文本
                        status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
                        text_line = f"Elbow-Wrist-Club Angle: {angle:.1f}° ({status})"
                        # 在图像上绘制文本
                        cv2.putText(annotated_frame, text_line, (50, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        y_offset += line_height  # 移动到下一行位置

                        # 存储结果
                        # action_results.append(("Elbow-Wrist-Club Alignment", is_correct, deviation))
                        # 改为字典存储
                        action_results.append({
                            "condition": "Elbow-Wrist-Club Alignment",
                            "is_correct": is_correct,  # 布尔值
                            "deviation": deviation  # 数字
                        })

                    else:
                        # 向量模长为零的情况
                        text_line = "Elbow-Wrist-Club: Not enough data"
                        cv2.putText(annotated_frame, text_line, (50, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        y_offset += line_height
                        action_results.append(("Elbow-Wrist-Club Alignment", False, None))
                else:
                    # 关键点未检测到的情况
                    text_line = "Elbow-Wrist-Club: Keypoints not detected"
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += line_height
                    action_results.append(("Elbow-Wrist-Club Alignment", False, None))

                # 条件2: 左肩和左脚踝连线是否垂直
                if (confidences[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
                    # 获取关键点坐标
                    shoulder = keypoints[LEFT_SHOULDER_IDX]
                    ankle = keypoints[LEFT_ANKLE_IDX]

                    # 计算连线与水平面的夹角
                    angle = calculate_line_horizontal_angle(shoulder, ankle)
                    deviation = abs(angle - 90)  # 计算与90度的偏差
                    is_correct = deviation <= 5  # 是否满足条件
                    color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

                    # 生成状态文本
                    status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
                    text_line = f"Shoulder-Ankle Vertical: {angle:.1f}° ({status})"
                    # 在图像上绘制文本
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += line_height

                    # 存储结果
                    # action_results.append(("Shoulder-Ankle Vertical", is_correct, deviation))
                    # 同样修改其他条件的存储方式
                    action_results.append({
                        "condition": "Shoulder-Ankle Vertical",
                        "is_correct": is_correct,
                        "deviation": deviation
                    })
                else:
                    # 关键点未检测到的情况
                    text_line = "Shoulder-Ankle: Keypoints not detected"
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += line_height
                    action_results.append(("Shoulder-Ankle Vertical", False, None))

            # 上杆顶点动作的判定
            elif action == "Top of Backswing":
                # 条件1: 球杆是否平行于水平面
                if (confidences[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[HEAD_CLUB_IDX] > CONFIDENCE_THRESHOLD):
                    # 获取关键点坐标
                    mid_club = keypoints[MIDDLE_CLUB_IDX]
                    head_club = keypoints[HEAD_CLUB_IDX]

                    # 计算球杆与水平面的夹角
                    club_angle = calculate_club_horizontal_angle(mid_club, head_club)
                    # 计算与水平线（0度或180度）的最小偏差
                    deviation = min(abs(club_angle - 0), abs(club_angle - 180))
                    is_correct = deviation <= 10  # 允许10度偏差
                    color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

                    # 生成状态文本
                    status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
                    text_line = f"Club Horizontal Angle: {club_angle:.1f}° ({status})"
                    # 在图像上绘制文本
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += line_height

                    # 存储结果
                    # action_results.append(("Club Horizontal", is_correct, deviation))
                    # 同样修改其他条件的存储方式
                    action_results.append({
                        "condition": "Club Horizontal",
                        "is_correct": is_correct,
                        "deviation": deviation
                    })
                else:
                    # 关键点未检测到的情况
                    text_line = "Club Horizontal: Keypoints not detected"
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += line_height
                    action_results.append(("Club Horizontal", False, None))

                # 条件2: 左肩x坐标是否在脚踝中间
                if (confidences[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[RIGHT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
                    # 获取关键点x坐标
                    shoulder_x = keypoints[LEFT_SHOULDER_IDX][0]
                    left_ankle_x = keypoints[LEFT_ANKLE_IDX][0]
                    right_ankle_x = keypoints[RIGHT_ANKLE_IDX][0]

                    # 计算脚踝中心点x坐标
                    ankle_center = (left_ankle_x + right_ankle_x) / 2
                    # 计算阈值（脚踝间距的10%）
                    threshold = abs(right_ankle_x - left_ankle_x) * 0.1
                    # 计算与中心的偏差
                    deviation = abs(shoulder_x - ankle_center)
                    is_correct = deviation <= threshold  # 是否满足条件
                    color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

                    # 生成状态文本
                    status = "Correct" if is_correct else f"Deviation: {deviation:.1f}px"
                    text_line = f"Shoulder X Position: {shoulder_x:.1f}, Center: {ankle_center:.1f} ({status})"
                    # 在图像上绘制文本
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += line_height

                    # 存储结果
                    # action_results.append(("Shoulder X Alignment", is_correct, deviation))
                    # 同样修改其他条件的存储方式
                    action_results.append({
                        "condition": "Shoulder X Alignment",
                        "is_correct": is_correct,
                        "deviation": deviation
                    })
                else:
                    # 关键点未检测到的情况
                    text_line = "Shoulder X: Keypoints not detected"
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += line_height
                    action_results.append(("Shoulder X Alignment", False, None))

            # 击球动作的判定
            elif action == "Impact":
                # 条件: 左肩和左脚踝连线是否垂直
                if (confidences[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
                    # 获取关键点坐标
                    shoulder = keypoints[LEFT_SHOULDER_IDX]
                    ankle = keypoints[LEFT_ANKLE_IDX]

                    # 计算连线与水平面的夹角
                    angle = calculate_line_horizontal_angle(shoulder, ankle)
                    deviation = abs(angle - 90)  # 计算与90度的偏差
                    is_correct = deviation <= 5  # 是否满足条件
                    color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

                    # 生成状态文本
                    status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
                    text_line = f"Left_shoulder-Ankle Vertical: {angle:.1f}° ({status})"
                    # 在图像上绘制文本
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += line_height

                    # 存储结果
                    # action_results.append(("Left_shoulder-Ankle Vertical", is_correct, deviation))
                    # 同样修改其他条件的存储方式
                    action_results.append({
                        "condition": "Left_shoulder-Ankle Vertical",
                        "is_correct": is_correct,
                        "deviation": deviation
                    })
                else:
                    # 关键点未检测到的情况
                    text_line = "Left_shoulder-Ankle: Keypoints not detected"
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += line_height
                    action_results.append(("Left_shoulder-Ankle Vertical", False, None))

            # 收杆动作的判定
            elif action == "Finish":
                # 条件: 右肩和左脚踝连线是否垂直
                if (confidences[RIGHT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                        confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
                    # 获取关键点坐标
                    shoulder = keypoints[RIGHT_SHOULDER_IDX]
                    ankle = keypoints[LEFT_ANKLE_IDX]

                    # 计算连线与水平面的夹角
                    angle = calculate_line_horizontal_angle(shoulder, ankle)
                    deviation = abs(angle - 90)  # 计算与90度的偏差
                    is_correct = deviation <= 5  # 是否满足条件
                    color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

                    # 生成状态文本
                    status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
                    text_line = f"Right_shoulder-Ankle Vertical: {angle:.1f}° ({status})"
                    # 在图像上绘制文本
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += line_height

                    # 存储结果
                    # action_results.append(("Right_shoulder-Ankle Vertical", is_correct, deviation))
                    # 同样修改其他条件的存储方式
                    action_results.append({
                        "condition": "Right_shoulder-Ankle Vertical",
                        "is_correct": is_correct,
                        "deviation": deviation
                    })
                else:
                    # 关键点未检测到的情况
                    text_line = "Right_shoulder-Ankle: Keypoints not detected"
                    cv2.putText(annotated_frame, text_line, (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += line_height
                    action_results.append(("Right_shoulder-Ankle Vertical", False, None))

            # 保存关键帧图像
            output_path = os.path.join(keyframe_dir, f"{action.replace(' ', '_')}_{frame_idx}.jpg")
            cv2.imwrite(output_path, annotated_frame)  # 写入图像文件
            print(f"Saved {action} key frame to {output_path}")  # 打印保存信息

            # 收集图片信息用于批量处理
            img_filename = f"{action.replace(' ', '_')}_{frame_idx}.jpg"
            # 将图片信息添加到全局列表，稍后批量处理
            if not hasattr(save_and_evaluate_keyframe, 'pending_analyses'):
                save_and_evaluate_keyframe.pending_analyses = []
            # 确保所有检测到的关键帧都会被添加到AI分析列表中，不管判定结果如何
            # 将动作名称转换为AI分析期望的格式
            ai_action = action.replace(' ', '_')
            save_and_evaluate_keyframe.pending_analyses.append((ai_action, img_filename))
            print(f"Added {action} key frame to AI analysis queue: {img_filename} (AI action: {ai_action})")

            # 存储差异信息
            differences_summary.append({
                "action": action,  # 动作名称
                "frame": frame_idx,  # 帧索引
                "results": action_results  # 判定结果（现在是字典列表）
            })

        # --- 主程序：逐帧处理视频 ---
        frame_count = 0  # 当前帧计数器
        while True:
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:  # 检查是否成功读取帧
                break  # 视频结束或读取错误，退出循环

            frame_count += 1  # 帧计数器递增
            # 每30帧打印一次处理进度
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames}")
            # 阶段1：动作检测进度
            if progress_callback and total_frames > 0:
                percent = int(frame_count / total_frames * 80)  # 检测阶段最多80%
                progress_callback({'stage': 'detect', 'percent': percent})

            # 使用YOLO模型进行姿态估计
            results = model(frame, verbose=False)  # verbose=False减少控制台输出

            # 创建带标注的帧副本
            annotated_frame = frame.copy()
            # 当前帧数据字典
            frame_data = {
                "frame_index": frame_count,  # 帧索引
                "persons": []  # 检测到的人物列表
            }

            # 初始化当前帧的关键点数据
            current_kps = None  # 关键点坐标
            current_confs = None  # 关键点置信度
            current_angles = {}  # 角度值
            head_club_pos = None  # 球杆头位置
            middle_club_pos = None  # 球杆中点位置

            # 检查是否有有效的检测结果
            if results and results[0].keypoints and results[0].keypoints.data.shape[0] > 0 and results[
                0].keypoints.xy is not None and results[0].keypoints.conf is not None:
                # 获取关键点数据（转换为numpy数组）
                keypoints_data = results[0].keypoints.xy.cpu().numpy()
                confidences = results[0].keypoints.conf.cpu().numpy()

                # 遍历检测到的每个人
                for person_idx in range(keypoints_data.shape[0]):
                    person_keypoints = keypoints_data[person_idx]  # 当前人物的关键点
                    person_confs = confidences[person_idx]  # 当前人物的关键点置信度

                    # 计算所有角度
                    person_angles = {}
                    for angle_name, (idx1, idx2, idx3) in ANGLE_DEFINITIONS.items():
                        # 检查关键点置信度是否满足阈值
                        if (person_confs[idx1] > CONFIDENCE_THRESHOLD and
                                person_confs[idx2] > CONFIDENCE_THRESHOLD and
                                person_confs[idx3] > CONFIDENCE_THRESHOLD):
                            # 获取三个点的坐标
                            p1 = person_keypoints[idx1]
                            p2 = person_keypoints[idx2]
                            p3 = person_keypoints[idx3]
                            # 计算角度
                            angle = calculate_angle_between_points(p1, p2, p3, angle_name)
                            # 存储角度值（转换为整数）
                            person_angles[angle_name] = int(angle) if angle is not None else None

                    # 构建当前人物的数据字典
                    person_data = {
                        "person_id": person_idx,  # 人物ID
                        "keypoints": person_keypoints.tolist(),  # 关键点坐标（转换为列表）
                        "confidences": [round(conf, 4) for conf in person_confs.tolist()],  # 置信度（保留4位小数）
                        "angles": person_angles  # 角度值
                    }
                    # 添加到当前帧的人物列表
                    frame_data["persons"].append(person_data)

                    # 只处理第一个检测到的人（假设视频中只有一个高尔夫球员）
                    if person_idx == 0:
                        # 在当前帧上绘制姿态和角度
                        draw_pose_on_image(annotated_frame, person_keypoints, person_confs, person_angles)
                        # 存储当前帧数据（用于动作检测）
                        current_kps = person_keypoints
                        current_confs = person_confs
                        current_angles = person_angles

                        # 存储球杆位置（如果置信度足够）
                        if person_confs[HEAD_CLUB_IDX] > CONFIDENCE_THRESHOLD:
                            head_club_pos = tuple(person_keypoints[HEAD_CLUB_IDX])
                        if person_confs[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD:
                            middle_club_pos = tuple(person_keypoints[MIDDLE_CLUB_IDX])

            # 将当前帧的数据添加到总列表
            all_frame_data.append(frame_data)

            # 更新球杆位置历史
            head_club_history.append(head_club_pos)
            middle_club_history.append(middle_club_pos)

            # 新版本两阶段检测：第一阶段 - 收集帧数据到各动作list
            if current_kps is not None and current_confs is not None:
                try:
                    # 获取关键点坐标
                    lw_x = current_kps[LEFT_WRIST_IDX][0]   # 左手腕x
                    rw_x = current_kps[RIGHT_WRIST_IDX][0]  # 右手腕x
                    lw_y = current_kps[LEFT_WRIST_IDX][1]   # 左手腕y
                    rw_y = current_kps[RIGHT_WRIST_IDX][1]  # 右手腕y
                    la_x = current_kps[LEFT_ANKLE_IDX][0]   # 左脚踝x
                    ra_x = current_kps[RIGHT_ANKLE_IDX][0]  # 右脚踝x
                    mc_x = current_kps[MIDDLE_CLUB_IDX][0]  # 球杆手柄x
                    mc_y = current_kps[MIDDLE_CLUB_IDX][1]  # 球杆手柄y
                    ch_x = current_kps[HEAD_CLUB_IDX][0]    # 球杆头x
                except Exception:
                    continue  # 关键点缺失，跳过

                # 状态切换逻辑
                if state == 'Preparation':
                    # 前置判断：检查是否满足准备动作条件
                    if not preparation_detected:
                        # 检查手腕是否在髋部之间且下方
                        lh_x = current_kps[LEFT_HIP_IDX][0]     # 左髋x
                        rh_x = current_kps[RIGHT_HIP_IDX][0]    # 右髋x
                        lh_y = current_kps[LEFT_HIP_IDX][1]     # 左髋y
                        rh_y = current_kps[RIGHT_HIP_IDX][1]    # 右髋y
                        
                        min_hip_x = min(lh_x, rh_x)
                        max_hip_x = max(lh_x, rh_x)
                        max_hip_y = max(lh_y, rh_y)
                        
                        wrists_in_hip_x = (min_hip_x < lw_x < max_hip_x) and (min_hip_x < rw_x < max_hip_x)
                        wrists_below_hip_y = (lw_y > max_hip_y) and (rw_y > max_hip_y)
                        
                        if wrists_in_hip_x and wrists_below_hip_y:
                            preparation_detected = True
                            print(f"第{frame_count}帧检测到准备动作")
                    
                    if lw_x < ra_x:  # 满足切换到Top的条件
                        state = 'Top'
                        print(f"第{frame_count}帧进入Top动作状态")
                        top_list.append(frame_count - 1)  # 当前帧归入上杆顶点list
                    else:
                        preparation_list.append(frame_count - 1)  # 当前帧归入准备动作list
                elif state == 'Top':
                    if ra_x < lw_x < la_x and mc_y > lw_y:  # 满足切换到Impact的条件
                        state = 'Impact'
                        print(f"第{frame_count}帧进入Impact动作状态")
                        impact_list.append(frame_count - 1)  # 当前帧归入击球动作list
                    else:
                        top_list.append(frame_count - 1)  # 当前帧归入上杆顶点list
                elif state == 'Impact':
                    if la_x < mc_x or la_x < ch_x or la_x < rw_x or la_x < lw_x:  # 满足切换到Finish的条件
                        state = 'Finish'
                        print(f"第{frame_count}帧进入Finish动作状态")
                        finish_list.append(frame_count - 1)  # 当前帧归入完成动作list
                    else:
                        impact_list.append(frame_count - 1)  # 当前帧归入击球动作list
                elif state == 'Finish':
                    finish_list.append(frame_count - 1)  # 当前帧归入完成动作list

                # 记录检测到的状态用于显示
                detected_states.append((state, frame_count))

            # 在视频顶部显示当前动作提示
            current_action_text = ""  # 初始化当前动作文本
            # 遍历所有检测到的状态
            for action, frame_idx in detected_states:
                # 只显示当前帧或之前的状态
                if frame_idx <= frame_count:
                    current_action_text = action  # 更新当前动作文本

            # 如果有当前动作文本，则绘制在视频顶部
            if current_action_text:
                # 计算文本尺寸
                text_size = cv2.getTextSize(current_action_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
                text_x = int((frame_width - text_size[0]) / 2)  # 水平居中
                text_y = int(text_size[1] * 1.5)  # 垂直位置

                # 绘制背景矩形
                cv2.rectangle(annotated_frame,
                              (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10),
                              TEXT_BG_COLOR, -1)  # -1表示填充矩形

                # 绘制文本
                cv2.putText(annotated_frame, current_action_text,
                            (text_x, text_y),
                            TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

            # 将带标注的帧写入输出视频
            out.write(annotated_frame)

        # 释放资源
        cap.release()
        out.release()

        # 新版本第二阶段：严格条件筛选和关键帧选择
        if progress_callback:
            progress_callback({'stage': 'analyze', 'percent': 80})
        
        try:
            # 准备关键点数据列表用于筛选
            keypoints_per_frame = []
            for frame_data in all_frame_data:
                if frame_data['persons']:
                    person0 = frame_data['persons'][0]
                    keypoints = np.array(person0['keypoints'])
                    keypoints_per_frame.append(keypoints)
                else:
                    keypoints_per_frame.append(None)
            
            # 前置判断：检查第一帧是否满足准备动作的初始条件
            if len(keypoints_per_frame) > 0 and keypoints_per_frame[0] is not None:
                kps0 = keypoints_per_frame[0]
                try:
                    lw_x = kps0[LEFT_WRIST_IDX][0]   # 左手腕x
                    rw_x = kps0[RIGHT_WRIST_IDX][0]  # 右手腕x
                    lh_x = kps0[LEFT_HIP_IDX][0]     # 左髋x
                    rh_x = kps0[RIGHT_HIP_IDX][0]    # 右髋x
                    lw_y = kps0[LEFT_WRIST_IDX][1]   # 左手腕y
                    rw_y = kps0[RIGHT_WRIST_IDX][1]  # 右手腕y
                    lh_y = kps0[LEFT_HIP_IDX][1]     # 左髋y
                    rh_y = kps0[RIGHT_HIP_IDX][1]    # 右髋y
                    
                    # 判断x坐标是否在左右髋之间
                    min_hip_x = min(lh_x, rh_x)
                    max_hip_x = max(lh_x, rh_x)
                    wrists_in_hip_x = (min_hip_x < lw_x < max_hip_x) and (min_hip_x < rw_x < max_hip_x)
                    # 判断y坐标是否在左右髋下面
                    max_hip_y = max(lh_y, rh_y)
                    wrists_below_hip_y = (lw_y > max_hip_y) and (rw_y > max_hip_y)
                    
                    if not (wrists_in_hip_x and wrists_below_hip_y):
                        print("未检测到准备动作：第一帧手腕不在髋部之间且下方")
                        return {'error': '未检测到准备动作：第一帧手腕不在髋部之间且下方'}
                except Exception:
                    print("未检测到准备动作：第一帧关键点缺失")
                    return {'error': '未检测到准备动作：第一帧关键点缺失'}
            
            # 第二阶段：对每个动作list进行严格条件筛选和判定
            print("开始第二阶段：严格条件筛选...")
            
            # Preparation严格条件筛选和判定，center_value为265度
            preparation_best_idx, preparation_result = select_best_and_judge(
                preparation_list, keypoints_per_frame, strict_preparation_metric, center_value=265)
            
            # Top严格条件筛选和判定，center_value为25度
            top_best_idx, top_result = select_best_and_judge(
                top_list, keypoints_per_frame, strict_top_metric, center_value=0)
            
            # Impact严格条件筛选和判定，center_value为90度
            impact_best_idx, impact_result = select_best_and_judge(
                impact_list, keypoints_per_frame, strict_impact_metric, center_value=0)
            
            # Finish严格条件筛选和判定，mode='finish'
            finish_best_idx, finish_result = select_best_and_judge(
                finish_list, keypoints_per_frame, strict_finish_metric_v2, mode='finish')
            
            # 更新关键帧信息
            if preparation_best_idx is not None:
                key_frames["Preparation"] = preparation_best_idx + 1
                print(f"Preparation最佳帧: {preparation_best_idx + 1}, 判定: {preparation_result}")
            
            if top_best_idx is not None:
                key_frames["Top of Backswing"] = top_best_idx + 1
                print(f"Top最佳帧: {top_best_idx + 1}, 判定: {top_result}")
            
            if impact_best_idx is not None:
                key_frames["Impact"] = impact_best_idx + 1
                print(f"Impact最佳帧: {impact_best_idx + 1}, 判定: {impact_result}")
            
            if finish_best_idx is not None:
                key_frames["Finish"] = finish_best_idx + 1
                print(f"Finish最佳帧: {finish_best_idx + 1}, 判定: {finish_result}")
            
            # 处理未识别的动作
            from config import GOLF_ACTIONS
            unrecognized_msg = 'GolfSwingsAssistant懵了\n这球杆去哪儿了？没识别到完整的挥杆动作哦。\n请您帮忙检查视频是否拍全了，学学\'参考视频\' 的范儿，再上传一次呗？'
            
            # 检查哪些动作未被识别
            for action in GOLF_ACTIONS:
                if action not in key_frames or key_frames[action] is None:
                    # 生成未识别动作的JSON文件
                    subdir = os.path.basename(output_dir)
                    cache_filename = f"desc_{subdir}_Unrecognized_{action}.json"
                    cache_path = os.path.join(output_dir, 'result_video', cache_filename)
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    
                    cache_data = {
                        'image_path': 'uploads/standard/key_frames/Unrecognized.jpg',
                        'subdir': subdir,
                        'filename': 'Unrecognized.jpg',
                        'description': unrecognized_msg,
                        'prompt': f'{action}动作',
                        'action': action,
                        'frame_idx': None,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"Generated unrecognized action JSON for {action}")
            
            # 保存关键帧图片和进行AI分析
            if any(key_frames.values()):
                cap = cv2.VideoCapture(VIDEO_PATH)
                total_keyframes = sum(1 for v in key_frames.values() if v is not None)
                done = 0
                
                # 确保pending_analyses列表被正确初始化
                if not hasattr(save_and_evaluate_keyframe, 'pending_analyses'):
                    save_and_evaluate_keyframe.pending_analyses = []
                
                for action, frame_idx in key_frames.items():
                    if frame_idx is None:
                        continue
                    
                    # 定位到关键帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                    ret, frame = cap.read()  # 读取关键帧
                    if not ret:
                        continue  # 读取失败则跳过

                    # 获取对应帧的数据
                    data_index = frame_idx - 1
                    if data_index >= len(all_frame_data):
                        continue  # 索引超出范围则跳过

                    # 获取帧数据
                    frame_data = all_frame_data[data_index]
                    if not frame_data['persons']:
                        continue  # 没有检测到人物则跳过

                    # 获取第一个人的数据
                    person0 = frame_data['persons'][0]
                    keypoints = np.array(person0['keypoints'])  # 关键点坐标
                    confidences = np.array(person0['confidences'])  # 关键点置信度
                    angles_dict = person0['angles']  # 角度值

                    # 保存并评估关键帧
                    save_and_evaluate_keyframe(frame_idx, frame, keypoints, confidences, angles_dict, action)
                    done += 1
                    
                    if progress_callback and total_keyframes > 0:
                        percent = 80 + int(done / total_keyframes * 20)
                        progress_callback({'stage': 'analyze', 'percent': percent})
                        
                cap.release()
                
                # 批量处理AI分析请求
                if hasattr(save_and_evaluate_keyframe, 'pending_analyses') and save_and_evaluate_keyframe.pending_analyses:
                    try:
                        print(f"开始批量处理AI分析请求... 共{len(save_and_evaluate_keyframe.pending_analyses)}个关键帧")
                        subdir = os.path.basename(output_dir)
                        
                        # 调用批量分析函数
                        batch_results = batch_assistant_analysis(save_and_evaluate_keyframe.pending_analyses, subdir)
                        
                        # 将结果转换为以动作为键的字典
                        batch_results_dict = {}
                        for result in batch_results:
                            if result['success']:
                                batch_results_dict[result['action']] = result['result']
                            else:
                                print(f"AI analysis failed for {result['action']}: {result['result']}")
                        
                        # 保存分析结果到缓存文件
                        cache_dir = os.path.join(output_dir, 'result_video')
                        os.makedirs(cache_dir, exist_ok=True)
                        
                        for action, img_filename in save_and_evaluate_keyframe.pending_analyses:
                            print(f"Processing AI analysis for {action}: {img_filename}")
                            if action in batch_results_dict:
                                description = batch_results_dict[action]
                                prompt = f"{action}动作"
                                
                                # 提取帧索引
                                frame_idx = int(img_filename.split('_')[-1].replace('.jpg', ''))
                                
                                cache_filename = f"desc_{subdir}_{img_filename.replace('.jpg', '.json')}"
                                cache_path = os.path.join(cache_dir, cache_filename)
                                
                                cache_data = {
                                    'image_path': f'uploads/{subdir}/key_frames/{img_filename}',
                                    'subdir': subdir,
                                    'filename': img_filename,
                                    'description': description,
                                    'prompt': prompt,
                                    'action': action,
                                    'frame_idx': frame_idx,
                                    'timestamp': datetime.datetime.now().isoformat()
                                }
                                
                                with open(cache_path, 'w', encoding='utf-8') as f:
                                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                                
                                print(f"Generated and cached description for {action} key frame")
                            else:
                                print(f"No AI analysis result for {action}")
                        
                        print("批量AI分析处理完成")
                    except Exception as e:
                        print(f"批量AI分析处理失败: {e}")
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            error_msg = f"Error processing key frames: {str(e)}"
            print(error_msg)
            return {'error': error_msg}
        # 处理结束后，确保进度100
        if progress_callback:
            progress_callback({'stage': 'analyze', 'percent': 100})
        
        # 等待文件写入完成
        import time
        time.sleep(1)

        # 打印差异总结
        print("\nSummary of differences from ideal pose:")
        # 遍历所有差异项
        for item in differences_summary:
            print(f"\nAction: {item['action']} (Frame {item['frame']})")
            # 遍历该动作的所有判定结果
            for (cond_name, cond_ok, deviation) in item['results']:
                if cond_ok:
                    print(f"  {cond_name}: OK")  # 条件满足
                else:
                    if deviation is not None:
                        print(f"  {cond_name}: Not OK, Deviation: {deviation:.1f}")  # 条件不满足且有偏差值
                    else:
                        print(f"  {cond_name}: Keypoints not available")  # 关键点不可用

        # 打印处理完成信息
        print(f"Processing finished. Annotated video saved to {OUTPUT_VIDEO_PATH}")
        print("Detected key frames:")
        # 打印所有检测到的关键帧
        for action, frame_idx in detected_states:
            print(f"  {action} at frame {frame_idx}")
        
        # 打印新版本严格条件判定结果
        print("\n新版本严格条件判定结果:")
        if 'preparation_result' in locals():
            print(f"  Preparation: {preparation_result}")
        if 'top_result' in locals():
            print(f"  Top of Backswing: {top_result}")
        if 'impact_result' in locals():
            print(f"  Impact: {impact_result}")
        if 'finish_result' in locals():
            print(f"  Finish: {finish_result}")

        # 返回分析结果
        result = {
            'output_video': OUTPUT_VIDEO_PATH,
            'key_frames': key_frames,
            'differences_summary': differences_summary,
            'detected_states': detected_states,
            # 新版本判定结果
            'strict_results': {
                'Preparation': preparation_result if 'preparation_result' in locals() else None,
                'Top of Backswing': top_result if 'top_result' in locals() else None,
                'Impact': impact_result if 'impact_result' in locals() else None,
                'Finish': finish_result if 'finish_result' in locals() else None
            },
            'best_frames': {
                'Preparation': preparation_best_idx + 1 if 'preparation_best_idx' in locals() and preparation_best_idx is not None else None,
                'Top of Backswing': top_best_idx + 1 if 'top_best_idx' in locals() and top_best_idx is not None else None,
                'Impact': impact_best_idx + 1 if 'impact_best_idx' in locals() and impact_best_idx is not None else None,
                'Finish': finish_best_idx + 1 if 'finish_best_idx' in locals() and finish_best_idx is not None else None
            }
        }

        return result

    except Exception as e:
        import traceback
        error_msg = f"Unexpected error during analysis: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}
    finally:
        # 确保释放资源
        if cap is not None and cap.isOpened():
            cap.release()
        if out is not None and out.isOpened():
            out.release()