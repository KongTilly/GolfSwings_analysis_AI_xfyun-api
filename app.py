import os
import threading
import time
import json
import glob
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from golf_analysis import analyze_golf_swing
from golfswingsAssistant import assistant_answer
from config import FlaskConfig, allowed_file, GOLF_ACTIONS, DemoConfig, ErrorMessages
from common_utils import (
    ensure_directories, write_progress, read_progress, get_keyframe_images,
    check_keyframe_files, get_action_from_filename, format_keyframe_data,
    get_upload_subdir, construct_image_path, get_unrecognized_image_path, 
    ensure_upload_directories, extract_path_info, get_history_analysis_list
)
from cache_manager import CacheManager


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FlaskConfig.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = FlaskConfig.MAX_CONTENT_LENGTH
app.config['ALLOWED_EXTENSIONS'] = FlaskConfig.ALLOWED_EXTENSIONS

# 初始化
ensure_directories(app.config['UPLOAD_FOLDER'])
os.environ['MODEL_PATH'] = FlaskConfig.MODEL_PATH
cache_manager = CacheManager(app.config['UPLOAD_FOLDER'])

def analyze_golf_swing_with_progress(video_path, output_dir, filename, subdir):
    """带进度跟踪的高尔夫挥杆分析
    
    Args:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出目录路径
        filename (str): 文件名
        subdir (str): 子目录名
        
    Returns:
        dict: 分析结果
    """
    def progress_callback(progress):
        write_progress(app.config['UPLOAD_FOLDER'], subdir, filename, progress)
    
    write_progress(app.config['UPLOAD_FOLDER'], subdir, filename, {'stage': 'detect', 'percent': 0})
    result = analyze_golf_swing(video_path, output_dir, progress_callback)
    
    # 等待关键帧生成完成
    keyframe_dir = os.path.join(output_dir, 'key_frames')
    max_wait_time = 30
    wait_count = 0
    while wait_count < max_wait_time:
        all_images_exist = all(
            glob.glob(os.path.join(keyframe_dir, f'{action}_*.jpg'))
            for action in GOLF_ACTIONS
        )
        if all_images_exist:
            break
        time.sleep(1)
        wait_count += 1

    write_progress(app.config['UPLOAD_FOLDER'], subdir, filename, {'stage': 'analyze', 'percent': 100})
    return result

def process_ai_analysis(subdir, action_images):
    """处理AI分析结果
    
    Args:
        subdir (str): 子目录名
        action_images (dict): 动作图片映射
        
    Returns:
        list: AI分析结果列表
    """
    ai_analysis_results = []
    
    for action in GOLF_ACTIONS:
        img_filename = action_images.get(action)
        
        if not img_filename or 'Unrecognized.jpg' in img_filename:
            ai_analysis_results.append({
                'action': action,
                'description': ErrorMessages.UNRECOGNIZED_MSG,
                'timestamp': '',
                'image_path': get_unrecognized_image_path()
            })
            continue
        
        # 检查缓存
        cached_data = cache_manager.get_cached_analysis(subdir, action)
        if cached_data and cached_data.get('description'):
            ai_analysis_results.append({
                'action': action,
                'description': cached_data['description'],
                'timestamp': cached_data.get('timestamp', ''),
                'image_path': cached_data.get('image_path', construct_image_path(subdir, img_filename))
            })
            continue
        
        # 新的AI分析
        try:
            analysis_result = assistant_answer(action, img_filename, subdir)
            image_path = construct_image_path(subdir, img_filename)
            
            ai_analysis_results.append({
                'action': action,
                'description': analysis_result,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            })
            
            cache_manager.save_analysis_result(subdir, action, img_filename, analysis_result, image_path)
        except Exception as e:
            ai_analysis_results.append({
                'action': action,
                'description': ErrorMessages.ANALYSIS_FAILED,
                'timestamp': '',
                'image_path': construct_image_path(subdir, img_filename) if img_filename else get_unrecognized_image_path()
            })
    
    return ai_analysis_results

@app.route('/', methods=['GET'])
def index():
    """显示主页"""
    return render_template('index.html')

@app.route('/history')
def show_history():
    """显示历史分析页面"""
    history_list = get_history_analysis_list(app.config['UPLOAD_FOLDER'])
    return render_template('history.html', history_list=history_list)

@app.route('/progress/<subdir>/<filename>')
def get_progress(subdir, filename):
    """获取分析进度
    
    Args:
        subdir (str): 子目录名
        filename (str): 文件名
        
    Returns:
        json: 进度信息
    """
    progress = read_progress(app.config['UPLOAD_FOLDER'], subdir, filename)
    return jsonify(progress)

@app.route('/check_files/<subdir>/<filename>')
def check_files(subdir, filename):
    """检查关键帧文件是否已生成
    
    Args:
        subdir (str): 子目录名
        filename (str): 文件名
        
    Returns:
        json: 文件状态信息
    """
    file_status = check_keyframe_files(app.config['UPLOAD_FOLDER'], subdir)
    return jsonify(file_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理视频文件上传并启动分析
    
    Returns:
        json: 任务信息或重定向
    """
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    subdir = get_upload_subdir(filename)
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], subdir)
    ensure_upload_directories(app.config['UPLOAD_FOLDER'], subdir)
    
    video_path = os.path.join(upload_dir, filename)
    file.save(video_path)
    
    # 启动分析线程
    def run_analysis():
        analyze_golf_swing_with_progress(video_path, upload_dir, filename, subdir)
    threading.Thread(target=run_analysis).start()
    
    return jsonify({'task_id': filename, 'subdir': subdir})

@app.route('/guide')
def show_guide():
    """显示指导页面"""
    result_data = {
        'key_frames': request.args.getlist('key_frames'),
        'differences_summary': request.args.getlist('differences_summary'),
        'detected_states': request.args.getlist('detected_states')
    }
    return render_template('guide.html', result=result_data)

@app.route('/ai_analysis/<subdir>/<filename>')
def show_ai_analysis(subdir, filename):
    """显示AI分析结果页面
    
    Args:
        subdir (str): 子目录名
        filename (str): 文件名
        
    Returns:
        str: 渲染的HTML页面
    """
    try:
        if subdir == DemoConfig.DEMO_SUBDIR:
            return render_template('ai_analysis.html', 
                                 ai_results=DemoConfig.DEMO_AI_RESULTS,
                                 subdir=subdir, filename=filename)
        
        action_images = get_keyframe_images(app.config['UPLOAD_FOLDER'], subdir)
        ai_analysis_results = process_ai_analysis(subdir, action_images)
        
        return render_template('ai_analysis.html', 
                             ai_results=ai_analysis_results,
                             subdir=subdir, filename=filename)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/results/<subdir>/<filename>')
def show_results(subdir, filename):
    """显示分析结果页面
    
    Args:
        subdir (str): 子目录名
        filename (str): 文件名
        
    Returns:
        str: 渲染的HTML页面
    """
    cache_manager.clear_cache(subdir, filename)
    key_frames = format_keyframe_data(app.config['UPLOAD_FOLDER'], subdir)
    
    result_data = {
        'input_video': f'{subdir}/{filename}',
        'output_video': f'{subdir}/result_video/analyzed_{os.path.splitext(filename)[0]}.mp4',
        'key_frames': key_frames,
        'differences_summary': [],
        'detected_states': []
    }
    return render_template('results.html', result=result_data)

@app.route('/demo_results')
def show_demo_results():
    """显示演示结果页面
    
    Returns:
        str: 渲染的HTML页面
    """
    result_data = {
        'input_video': f'{DemoConfig.DEMO_SUBDIR}/{DemoConfig.DEMO_FILENAME}',
        'output_video': f'{DemoConfig.DEMO_SUBDIR}/analyzed_{DemoConfig.DEMO_FILENAME}',
        'key_frames': DemoConfig.DEMO_KEY_FRAMES,
        'differences_summary': [],
        'detected_states': []
    }
    return render_template('results.html', result=result_data)

@app.route('/download/<subdir>/<filename>')
def download_file(subdir, filename):
    """下载分析结果视频文件
    
    Args:
        subdir (str): 子目录名
        filename (str): 文件名
        
    Returns:
        file: 文件下载响应
    """
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], subdir, 'result_video'), 
        filename, as_attachment=True
    )

@app.route('/demo_download/<filename>')
def demo_download_file(filename):
    """下载演示视频文件
    
    Args:
        filename (str): 文件名
        
    Returns:
        file: 文件下载响应
    """
    return send_from_directory('static/dome_show/result_video', filename, as_attachment=True)

@app.route('/api/describe_image', methods=['POST'])
def api_describe_image():
    """API接口：获取图片描述
    
    Returns:
        json: 图片描述结果
    """
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'success': False, 'error': ErrorMessages.MISSING_PARAMS})
        
        image_path = data['image_path']
        subdir, filename = extract_path_info(image_path)
        
        if not subdir or not filename:
            return jsonify({'success': False, 'error': ErrorMessages.PATH_FORMAT_ERROR})
        
        action = get_action_from_filename(filename)
        if not action:
            return jsonify({'success': False, 'error': ErrorMessages.ACTION_NOT_RECOGNIZED})
        
        cached_data = cache_manager.get_cached_analysis(subdir, action)
        if cached_data:
            return jsonify({
                'success': True, 
                'description': cached_data.get('description', ''),
                'prompt': cached_data.get('prompt', ''),
                'cached': True
            })
        
        result = assistant_answer(action, filename, subdir)
        description = "GolfSwingsAssistant: " + result
        
        cache_data = {
            'image_path': image_path,
            'subdir': subdir,
            'filename': filename,
            'description': description,
            'prompt': "",
            'timestamp': datetime.now().isoformat()
        }
        cache_manager.save_cache_data(subdir, filename, cache_data)
        
        return jsonify({
            'success': True,
            'description': description,
            'prompt': "",
            'cached': False
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)