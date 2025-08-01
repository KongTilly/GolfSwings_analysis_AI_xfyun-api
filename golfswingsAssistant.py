import _thread as thread
import logging
import base64
import datetime
import threading
import json
import websocket
import hashlib
import hmac
from urllib.parse import urlparse, urlencode
import ssl
from datetime import datetime
from time import mktime, sleep
from wsgiref.handlers import format_date_time
import os

# 日志配置已在config.py中统一配置

class Ws_Param(object):
    """星火API WebSocket参数配置类"""
    
    def __init__(self, APPID, APIKey, APISecret, imageunderstanding_url):
        """
        初始化WebSocket参数
        
        Args:
            APPID (str): 星火API的应用ID
            APIKey (str): 星火API的密钥
            APISecret (str): 星火API的密钥
            imageunderstanding_url (str): 图像理解API的WebSocket URL
        """
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(imageunderstanding_url).netloc
        self.path = urlparse(imageunderstanding_url).path
        self.ImageUnderstanding_url = imageunderstanding_url

    def create_url(self):
        """
        生成带鉴权参数的WebSocket URL
        
        Returns:
            str: 包含鉴权参数的完整WebSocket URL
        """
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'),
                                 signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.ImageUnderstanding_url + '?' + urlencode(v)
        return url

class SparkWebSocketClient:
    """星火API WebSocket客户端通用类"""
    
    def __init__(self, appid, api_secret, api_key, url):
        """
        初始化WebSocket客户端
        
        Args:
            appid (str): 星火API的应用ID
            api_secret (str): 星火API的密钥
            api_key (str): 星火API的密钥
            url (str): WebSocket连接URL
        """
        self.appid = appid
        self.api_secret = api_secret
        self.api_key = api_key
        self.url = url
        self.ws_param = Ws_Param(appid, api_key, api_secret, url)
        self.result = {"answer": ""}
        self.done_event = threading.Event()
    
    def gen_params(self, question):
        """
        生成请求参数
        
        Args:
            question: 要发送的问题或内容
            
        Returns:
            dict: 包含请求参数的字典
        """
        data = {
            "header": {
                "app_id": self.appid,
                "uid": "12345"
            },
            "parameter": {
                "chat": {
                    "domain": "general",
                    "temperature": 0.5,
                    "max_tokens": 2048
                }
            },
            "payload": {
                "message": {
                    "text": question
                }
            }
        }
        return data
    
    def on_message(self, ws, message):
        """
        处理接收到的消息
        
        Args:
            ws: WebSocket连接对象
            message (str): 接收到的消息
        """
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            logging.error(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            self.result["answer"] += content
            if status == 2:
                self.done_event.set()
                ws.close()
    
    def on_error(self, ws, error):
        """
        处理WebSocket错误
        
        Args:
            ws: WebSocket连接对象
            error: 错误信息
        """
        logging.error(f"WebSocket错误: {error}")
        self.done_event.set()
    
    def on_close(self, ws, *args):
        """
        处理WebSocket连接关闭
        
        Args:
            ws: WebSocket连接对象
            *args: 其他参数
        """
        logging.info("WebSocket连接已关闭")
    
    def on_open(self, ws):
        """
        处理WebSocket连接打开
        
        Args:
            ws: WebSocket连接对象
        """
        logging.info("WebSocket连接已建立")
        
        def run(*args):
            data = self.gen_params(self.question)
            ws.send(json.dumps(data))
        thread.start_new_thread(run, ())
    
    def send_request(self, question):
        """
        发送请求并返回结果
        
        Args:
            question: 要发送的问题或内容
            
        Returns:
            str: API返回的答案
        """
        self.question = question
        self.result = {"answer": ""}
        self.done_event.clear()
        
        websocket.enableTrace(False)
        ws_url = self.ws_param.create_url()
        ws = websocket.WebSocketApp(ws_url,
                                   on_message=self.on_message,
                                   on_error=self.on_error,
                                   on_close=self.on_close,
                                   on_open=self.on_open)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        
        # 等待结果
        self.done_event.wait(timeout=30)
        return self.result["answer"]


def assistant_answer(action, img_path, subdir=None):
    """
    调用星火API，返回针对特定动作的分析结果字符串。异常时返回错误信息字符串。
    action: 动作类型，如'Preparation', 'Top_of_Backswing', 'Impact', 'Finish'
    img_path: 只需传入图片文件名，subdir为static/uploads下的子目录
    """
    from config import SparkAPIConfig
    
    appid = SparkAPIConfig.APPID
    api_secret = SparkAPIConfig.API_SECRET
    api_key = SparkAPIConfig.API_KEY
    imageunderstanding_url = SparkAPIConfig.IMAGE_UNDERSTANDING_URL

    # 构造图片绝对路径
    if subdir:
        img_path_full = f'static/uploads/{subdir}/key_frames/{img_path}'
    else:
        img_path_full = img_path
    
    try:
        with open(img_path_full, 'rb') as f:
            imagedata = f.read()
    except Exception as e:
        logging.error(f"图片读取失败: {e}")
        return f"图片读取失败: {e}"

    # 构造针对特定动作的prompt
    action_prompts = {
        'Preparation': "请分析我的这张高尔夫球准备动作图片，重点关注站位、握杆姿势、身体平衡和准备阶段的要点。请用简洁明了的语言描述，控制在150字以内。",
        'Top_of_Backswing': "请分析我的这张高尔夫球上杆顶点图片，重点关注上杆幅度、身体旋转、手臂位置和上杆顶点的技术要点。请用简洁明了的语言描述，控制在150字以内。",
        'Impact': "请分析我的这张高尔夫球击球瞬间图片，重点关注击球姿势、身体角度、手臂位置和击球瞬间的技术要点。请用简洁明了的语言描述，控制在150字以内。",
        'Finish': "请分析我的这张高尔夫球收杆图片，重点关注收杆姿势、身体平衡、手臂位置和收杆阶段的技术要点。请用简洁明了的语言描述，控制在150字以内。"
    }
    
    prompt = action_prompts.get(action, "请分析这张高尔夫球挥杆动作图片的技术要点。请用简洁明了的语言描述，控制在150字以内。")
    
    text = [
        {"role": "user", "content": str(base64.b64encode(imagedata), 'utf-8'), "content_type": "image"},
        {"role": "user", "content": prompt}
    ]

    # 使用通用WebSocket客户端
    client = SparkWebSocketClient(appid, api_secret, api_key, imageunderstanding_url)
    result = client.send_request(text)
    
    return result

def batch_assistant_analysis(actions_and_images, subdir=None):
    """
    批量分析多个动作图片
    
    Args:
        actions_and_images (list): 动作和图片的列表，格式为[(action, img_path), ...]
        subdir (str, optional): 子目录名
        
    Returns:
        list: 分析结果列表
    """
    results = []
    
    def process_batch(batch, batch_index):
        """处理批次请求"""
        batch_results = []
        for action, img_path in batch:
            try:
                def process_single_request(action, img_path):
                    """处理单个请求"""
                    result = assistant_answer(action, img_path, subdir)
                    return {"action": action, "result": result, "success": True}
                
                batch_results.append(process_single_request(action, img_path))
            except Exception as e:
                batch_results.append({"action": action, "result": str(e), "success": False})
        return batch_results
    
    # 分批处理，每批最多3个请求
    batch_size = 2
    for i in range(0, len(actions_and_images), batch_size):
        batch = actions_and_images[i:i + batch_size]
        batch_results = process_batch(batch, i // batch_size)
        results.extend(batch_results)
    
    return results



