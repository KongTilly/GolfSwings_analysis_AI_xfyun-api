# -*- coding: utf-8 -*-
"""
缓存管理模块
统一处理所有缓存相关的逻辑
"""

import os
import json
import glob
import logging
from datetime import datetime
from typing import Optional, Dict, Any

class CacheManager:
    """缓存管理器类"""
    
    def __init__(self, upload_folder: str):
        """初始化缓存管理器
        
        Args:
            upload_folder (str): 上传文件夹路径
        """
        self.upload_folder = upload_folder
    
    def get_cache_path(self, subdir: str, action: str) -> Optional[str]:
        """获取缓存文件路径
        
        Args:
            subdir (str): 子目录名
            action (str): 动作类型
            
        Returns:
            Optional[str]: 缓存文件路径，如果未找到则返回None
        """
        cache_dir = os.path.join(self.upload_folder, subdir, 'result_video')
        
        # 首先尝试查找正常的动作JSON文件
        cache_filename = f"desc_{subdir}_{action}_*.json"
        pattern = os.path.join(cache_dir, cache_filename)
        matches = glob.glob(pattern)
        
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
        
        # 如果没有找到正常文件，尝试查找未识别动作的JSON文件
        unrecognized_filename = f"desc_{subdir}_Unrecognized_{action}.json"
        unrecognized_pattern = os.path.join(cache_dir, unrecognized_filename)
        if os.path.exists(unrecognized_pattern):
            return unrecognized_pattern
        
        return None
    
    def load_cache_data(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """加载缓存数据
        
        Args:
            cache_path (str): 缓存文件路径
            
        Returns:
            Optional[Dict[str, Any]]: 缓存数据，如果加载失败则返回None
        """
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"读取缓存文件失败: {e}")
            return None
    
    def save_cache_data(self, subdir: str, filename: str, data: Dict[str, Any]) -> bool:
        """保存缓存数据
        
        Args:
            subdir (str): 子目录名
            filename (str): 文件名
            data (Dict[str, Any]): 要保存的数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            cache_dir = os.path.join(self.upload_folder, subdir, 'result_video')
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_filename = f"desc_{subdir}_{filename.replace('.jpg', '.json')}"
            cache_path = os.path.join(cache_dir, cache_filename)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"已保存缓存数据: {cache_path}")
            return True
        except Exception as e:
            logging.error(f"保存缓存文件失败: {e}")
            return False
    
    def get_cached_analysis(self, subdir: str, action: str) -> Optional[Dict[str, Any]]:
        """获取缓存的分析结果
        
        Args:
            subdir (str): 子目录名
            action (str): 动作类型
            
        Returns:
            Optional[Dict[str, Any]]: 缓存的分析结果，如果未找到则返回None
        """
        cache_path = self.get_cache_path(subdir, action)
        if cache_path:
            return self.load_cache_data(cache_path)
        return None
    
    def save_analysis_result(self, subdir: str, action: str, img_filename: str, 
                           analysis_result: str, image_path: str) -> bool:
        """保存分析结果到缓存
        
        Args:
            subdir (str): 子目录名
            action (str): 动作类型
            img_filename (str): 图片文件名
            analysis_result (str): 分析结果
            image_path (str): 图片路径
            
        Returns:
            bool: 保存是否成功
        """
        cache_data = {
            'image_path': image_path,
            'subdir': subdir,
            'filename': img_filename,
            'description': analysis_result,
            'prompt': f"{action}动作",
            'action': action,
            'frame_idx': None,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.save_cache_data(subdir, img_filename, cache_data)
    
    def clear_cache(self, subdir: str, filename: str) -> bool:
        """清除指定文件的缓存
        
        Args:
            subdir (str): 子目录名
            filename (str): 文件名
            
        Returns:
            bool: 清除是否成功
        """
        try:
            progress_path = os.path.join(self.upload_folder, subdir, f'progress_{filename}.txt')
            if os.path.exists(progress_path):
                os.remove(progress_path)
                logging.info(f"已清除进度缓存: {progress_path}")
                return True
        except Exception as e:
            logging.error(f"清除缓存失败: {e}")
            return False
        return False 