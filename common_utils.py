# -*- coding: utf-8 -*-
"""
通用工具模块
合并和优化重复的工具函数
"""

import os
import json
import glob
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from config import GOLF_ACTIONS, ErrorMessages

def ensure_directories(*dirs):
    """确保目录存在
    
    Args:
        *dirs: 要创建的目录路径
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def get_upload_subdir(filename: str) -> str:
    """生成唯一子目录名：日期时间+视频名（不含扩展名）
    
    Args:
        filename (str): 文件名
        
    Returns:
        str: 生成的子目录名
    """
    date_time_str = datetime.now().strftime('%Y%m%d_%H%M')
    name, _ = os.path.splitext(filename)
    return f"{date_time_str}_{name}"

def construct_image_path(subdir: str, filename: str) -> str:
    """构造图片路径
    
    Args:
        subdir (str): 子目录名
        filename (str): 文件名
        
    Returns:
        str: 构造的图片路径
    """
    return f'uploads/{subdir}/key_frames/{filename}'

def get_keyframe_dir(upload_folder: str, subdir: str) -> str:
    """获取关键帧目录路径
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        
    Returns:
        str: 关键帧目录路径
    """
    return os.path.join(upload_folder, subdir, 'key_frames')

def get_result_video_dir(upload_folder: str, subdir: str) -> str:
    """获取结果视频目录路径
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        
    Returns:
        str: 结果视频目录路径
    """
    return os.path.join(upload_folder, subdir, 'result_video')

def get_upload_subdirs(upload_folder: str, subdir: str) -> tuple:
    """获取上传相关的所有子目录
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        
    Returns:
        tuple: 包含主目录、关键帧目录、结果视频目录的元组
    """
    return (
        os.path.join(upload_folder, subdir),
        get_keyframe_dir(upload_folder, subdir),
        get_result_video_dir(upload_folder, subdir)
    )

def get_standard_image_path(action: str) -> str:
    """获取标准图片路径
    
    Args:
        action (str): 动作类型
        
    Returns:
        str: 标准图片路径
    """
    return f'uploads/standard/key_frames/standard_{action}.jpg'

def get_unrecognized_image_path() -> str:
    """获取未识别图片路径
    
    Returns:
        str: 未识别图片路径
    """
    return 'uploads/standard/key_frames/Unrecognized.jpg'

def ensure_upload_directories(upload_folder: str, subdir: str) -> None:
    """确保上传相关目录存在
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
    """
    for dir_path in get_upload_subdirs(upload_folder, subdir):
        os.makedirs(dir_path, exist_ok=True)

def extract_path_info(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """从图片路径中提取子目录和文件名
    
    Args:
        image_path (str): 图片路径
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (子目录名, 文件名)
    """
    path_parts = image_path.split('/')
    try:
        uploads_index = path_parts.index('uploads')
        if uploads_index + 2 < len(path_parts):
            subdir = path_parts[uploads_index + 1]
            filename = path_parts[-1]
            return subdir, filename
    except ValueError:
        pass
    return None, None

def write_progress(upload_folder: str, subdir: str, filename: str, progress: Dict[str, Any]) -> None:
    """写入进度文件
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        filename (str): 文件名
        progress (Dict[str, Any]): 进度信息
    """
    progress_path = os.path.join(upload_folder, subdir, f'progress_{filename}.txt')
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, 'w') as f:
        f.write(json.dumps(progress))

def read_progress(upload_folder: str, subdir: str, filename: str) -> Dict[str, Any]:
    """读取进度文件
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        filename (str): 文件名
        
    Returns:
        Dict[str, Any]: 进度信息
    """
    progress_path = os.path.join(upload_folder, subdir, f'progress_{filename}.txt')
    if not os.path.exists(progress_path):
        return {'stage': 'detect', 'percent': 0}
    with open(progress_path, 'r') as f:
        try:
            return json.loads(f.read())
        except:
            return {'stage': 'detect', 'percent': 0}

def get_keyframe_images(upload_folder: str, subdir: str) -> Dict[str, Optional[str]]:
    """获取关键帧图片信息
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        
    Returns:
        Dict[str, Optional[str]]: 动作到图片文件名的映射
    """
    keyframe_dir = get_keyframe_dir(upload_folder, subdir)
    action_images = {}
    
    for action in GOLF_ACTIONS:
        img_pattern = os.path.join(keyframe_dir, f'{action}_*.jpg')
        img_matches = glob.glob(img_pattern)
        
        if img_matches:
            img_matches.sort(key=os.path.getmtime, reverse=True)
            img_filename = os.path.basename(img_matches[0])
            action_images[action] = img_filename
        else:
            action_images[action] = None
    
    return action_images

def check_keyframe_files(upload_folder: str, subdir: str) -> Dict[str, bool]:
    """检查关键帧文件是否已生成
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        
    Returns:
        Dict[str, bool]: 动作到文件存在状态的映射
    """
    keyframe_dir = get_keyframe_dir(upload_folder, subdir)
    file_status = {}
    
    for action in GOLF_ACTIONS:
        pattern = os.path.join(keyframe_dir, f'{action}_*.jpg')
        matches = glob.glob(pattern)
        file_status[action] = len(matches) > 0
    
    return file_status

def get_action_from_filename(filename: str) -> Optional[str]:
    """从文件名中提取动作类型
    
    Args:
        filename (str): 文件名
        
    Returns:
        Optional[str]: 动作类型，如果未找到则返回None
    """
    for action in GOLF_ACTIONS:
        if f'_{action}_' in filename:
            return action
    return None

def format_keyframe_data(upload_folder: str, subdir: str) -> List[Dict[str, str]]:
    """格式化关键帧数据
    
    Args:
        upload_folder (str): 上传文件夹路径
        subdir (str): 子目录名
        
    Returns:
        List[Dict[str, str]]: 格式化的关键帧数据列表
    """
    key_frames = []
    keyframe_dir = get_keyframe_dir(upload_folder, subdir)
    
    for action in GOLF_ACTIONS:
        pattern = os.path.join(keyframe_dir, f'{action}_*.jpg')
        matches = glob.glob(pattern)
        user_img = ''
        
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            user_img = f'uploads/{subdir}/key_frames/' + os.path.basename(matches[0])
        
        std_img = get_standard_image_path(action)
        key_frames.append({
            'action': action.replace('_', ' '), 
            'user_image': user_img, 
            'standard_image': std_img
        })
    
    return key_frames

def parse_analysis_time_from_dirname(item: str) -> str:
    """从目录名解析分析时间
    
    Args:
        item (str): 目录名
        
    Returns:
        str: 格式化的时间字符串
    """
    try:
        if '_' in item:
            date_part = item.split('_')[0]
            if len(date_part) == 8:  # YYYYMMDD格式
                date_obj = datetime.strptime(date_part, '%Y%m%d')
                return date_obj.strftime('%Y年%m月%d日')
            elif len(date_part) == 13:  # YYYYMMDD_HHMM格式
                date_obj = datetime.strptime(item.split('_')[0] + '_' + item.split('_')[1], '%Y%m%d_%H%M')
                return date_obj.strftime('%Y年%m月%d日 %H:%M')
        return item
    except:
        return item

def get_history_analysis_list(upload_folder: str) -> List[Dict[str, str]]:
    """获取历史分析列表
    
    Args:
        upload_folder (str): 上传文件夹路径
        
    Returns:
        List[Dict[str, str]]: 历史分析记录列表
    """
    history_list = []
    
    for item in os.listdir(upload_folder):
        item_path = os.path.join(upload_folder, item)
        if os.path.isdir(item_path) and item != 'standard':
            result_video_dir = os.path.join(item_path, 'result_video')
            key_frames_dir = os.path.join(item_path, 'key_frames')
            
            if os.path.exists(result_video_dir) and os.path.exists(key_frames_dir):
                video_files = [f for f in os.listdir(item_path) if f.endswith(('.mp4', '.avi', '.mov'))]
                if video_files:
                    original_video = video_files[0]
                    analysis_time = parse_analysis_time_from_dirname(item)
                    
                    keyframe_count = 0
                    if os.path.exists(key_frames_dir):
                        keyframe_files = [f for f in os.listdir(key_frames_dir) if f.endswith('.jpg')]
                        keyframe_count = len(keyframe_files)
                    
                    history_list.append({
                        'subdir': item,
                        'original_video': original_video,
                        'analysis_time': analysis_time,
                        'keyframe_count': keyframe_count,
                        'video_path': f'uploads/{item}/{original_video}'
                    })
    
    history_list.sort(key=lambda x: x['analysis_time'], reverse=True)
    return history_list 