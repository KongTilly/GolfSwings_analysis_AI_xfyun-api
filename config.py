# -*- coding: utf-8 -*-
"""
配置管理模块
集中管理所有配置常量和设置
"""

import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Flask应用配置
class FlaskConfig:
    """Flask应用配置类"""
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB限制
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    MODEL_PATH = os.getenv('MODEL_PATH', '.')

# 高尔夫分析配置
class GolfAnalysisConfig:
    """高尔夫分析配置类"""
    CONFIDENCE_THRESHOLD = 0.5
    DISPLACEMENT_THRESHOLD = 20
    WINDOW_SIZE = 20
    OFFSET_MM = 5
    OFFSET_PX = 20  # 5mm * 4px
    
    # 显示配置
    TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 2
    TEXT_THICKNESS = 5
    TEXT_COLOR = (0, 0, 0)
    TEXT_BG_COLOR = (255, 255, 255)
    POINT_COLOR = (0, 0, 255)
    LINE_COLOR = (0, 255, 0)
    
    # 关键点显示配置
    SHOW_KEYPOINTS = {
        "nose": False, "left_eye": False, "right_eye": False, "left_ear": False, "right_ear": False,
        "left_shoulder": True, "right_shoulder": True, "left_elbow": True, "right_elbow": True,
        "left_wrist": True, "right_wrist": True, "middle_club": True, "head_club": True,
        "left_hip": True, "right_hip": True, "left_knee": True, "right_knee": True, "left_ankle": True,
        "right_ankle": True
    }
    
    # 骨架显示配置
    SHOW_SKELETON = {
        "left_ankle_to_left_knee": True, "left_knee_to_left_hip": True, "right_ankle_to_right_knee": True,
        "right_knee_to_right_hip": True, "left_hip_to_right_hip": True, "left_shoulder_to_left_hip": True,
        "right_shoulder_to_right_hip": True, "left_shoulder_to_right_shoulder": True,
        "left_shoulder_to_left_elbow": True, "right_shoulder_to_right_elbow": True,
        "left_elbow_to_left_wrist": True, "right_elbow_to_right_wrist": True,
        "middle_club_to_head_club": True, "left_wrist_to_middle_club": True,
        "left_shoulder_to_left_ankle": True, "right_shoulder_to_left_ankle": True,
    }
    
    # 角度显示配置
    SHOW_ANGLES = {
        "right_hip_angle": False, "right_knee_angle": False, "right_elbow_angle": False, "right_shoulder_angle": False,
        "left_hip_angle": False, "left_knee_angle": False, "left_elbow_angle": True, "left_shoulder_angle": False,
        "club_angle": True, "left_wrist_angle": True
    }
    
    # 角度颜色配置
    ANGLE_COLORS = {
        "right_hip_angle": (0, 255, 255), "right_knee_angle": (255, 255, 0), "right_elbow_angle": (255, 0, 255),
        "right_shoulder_angle": (0, 165, 255), "left_hip_angle": (0, 255, 0), "left_knee_angle": (255, 0, 0),
        "left_elbow_angle": (128, 0, 128), "left_shoulder_angle": (0, 128, 128), "club_angle": (255, 192, 203),
        "left_wrist_angle": (255, 100, 0)
    }
    
    # 角度简称配置
    ANGLE_SHORT_NAMES = {
        "right_hip_angle": "RH", "right_knee_angle": "RK", "right_elbow_angle": "RE", "right_shoulder_angle": "RS",
        "left_hip_angle": "LH", "left_knee_angle": "LK", "left_elbow_angle": "LE", "left_shoulder_angle": "LS",
        "club_angle": "CLUB", "left_wrist_angle": "LW"
    }

# 星火API配置
class SparkAPIConfig:
    """星火API配置类"""
    APPID = "XXXXXXXX"
    API_SECRET = "XXXXXXXXXXXXXXXXXXXXXXXX"
    API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXX"
    IMAGE_UNDERSTANDING_URL = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"

# 高尔夫动作类型
GOLF_ACTIONS = ['Preparation', 'Top_of_Backswing', 'Impact', 'Finish']

# 演示数据配置
class DemoConfig:
    """演示数据配置类"""
    DEMO_SUBDIR = 'dome_show'
    DEMO_FILENAME = 'demo_show.mp4'
    
    # 生成演示关键帧数据
    def _create_demo_keyframe(action, frame_num):
        """创建演示关键帧数据
        
        Args:
            action (str): 动作类型
            frame_num (int): 帧号
            
        Returns:
            dict: 关键帧数据
        """
        return {
            'action': action,
            'user_image': f'dome_show/key_frames/{action}_{frame_num}.jpg',
            'standard_image': f'uploads/standard/key_frames/standard_{action}.jpg',
            'user_json': f'/static/dome_show/result_video/desc_dome_show_{action}_{frame_num}.json'
        }
    
    def _create_demo_ai_result(action, frame_num):
        """创建演示AI结果数据
        Args:
            action (str): 动作类型
            frame_num (int): 帧号
        """
        return {
            'action': action,
            'description': '正在加载分析结果...',
            'timestamp': '025-03-04T00:00:00',
            'image_path': f'dome_show/key_frames/{action}_{frame_num}.jpg',
            'user_json': f'/static/dome_show/result_video/desc_dome_show_{action}_{frame_num}.json'
        }
    
    DEMO_KEY_FRAMES = [
        _create_demo_keyframe('Preparation', 46),
        _create_demo_keyframe('Top_of_Backswing', 154),
        _create_demo_keyframe('Impact', 187),
        _create_demo_keyframe('Finish', 248)
    ]
    
    DEMO_AI_RESULTS = [
        _create_demo_ai_result('Preparation', 46),
        _create_demo_ai_result('Top_of_Backswing', 154),
        _create_demo_ai_result('Impact', 187),
        _create_demo_ai_result('Finish', 248)
    ]

# 错误消息配置
class ErrorMessages:
    """错误消息配置类"""
    UNRECOGNIZED_MSG = 'GolfSwingsAssistant懵了\n这球杆去哪儿了？没识别到完整的挥杆动作哦。\n请您帮忙检查视频是否拍全了，学学\'参考视频\' 的范儿，再上传一次呗？'
    ANALYSIS_FAILED = 'AI分析失败'
    MISSING_PARAMS = '缺少必要参数'
    PATH_FORMAT_ERROR = '图片路径格式错误'
    ACTION_NOT_RECOGNIZED = '无法识别动作类型'

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in FlaskConfig.ALLOWED_EXTENSIONS 