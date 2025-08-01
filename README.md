# 高尔夫挥杆分析系统 (Golf Swing Analysis System)

## 项目简介
基于计算机视觉和AI技术的高尔夫挥杆分析系统，能够自动识别挥杆动作关键帧并提供智能分析建议。系统采用YOLO目标检测和星火大模型API，为高尔夫爱好者提供专业的动作分析服务。

## 🏌️ 项目简介

## 功能特性

### 🎯 智能动作识别

- **自动识别挥杆阶段**：准备、上杆顶点、击球、收杆
  - Preparation (预备动作)
  - Top of Backswing (上杆顶点)
  - Impact (击球瞬间)
  - Finish (收杆动作)
- **高精度检测**: 基于YOLO模型的人体关键点检测
- **关键帧提取**：从视频中提取每个阶段的关键帧
- **AI智能分析**：基于星火大模型提供专业的动作分析
- **可视化展示**：直观展示分析结果和改进建议
- **历史记录管理**：保存和分析历史数据


### 📊 可视化界面
- **Web友好界面**: 基于Flask的现代化Web界面
- **关键帧对比**: 可视化显示标准动作与用户动作对比
- **历史记录**: 完整的历史分析记录管理
- **结果下载**: 支持分析结果视频下载

### 🎯 核心功能
- **视频上传与分析**：支持MP4、AVI、MOV格式视频上传
- **动作识别**：自动识别高尔夫挥杆的四个关键阶段
- **关键帧提取**：从每个阶段提取最具代表性的帧
- **AI分析**：基于星火大模型提供专业的动作分析
- **结果可视化**：生成分析报告和可视化结果

### 📊 分析维度
- **姿态检测**：人体关键点检测和骨架分析
- **角度计算**：计算身体各部位的角度数据
- **动作对比**：与标准动作进行对比分析
- **改进建议**：提供个性化的改进建议

### 🎨 用户界面
- **响应式设计**：支持桌面和移动设备
- **实时进度**：显示分析进度和状态
- **历史管理**：查看和管理历史分析记录
- **演示功能**：提供示例视频和结果展示

## 项目结构

```
golf_analysis_kong_api-new_way/
├── app.py                 # Flask应用主文件
├── golf_analysis.py       # 高尔夫挥杆分析核心模块
├── golfswingsAssistant.py # AI助手分析模块
├── config.py             # 配置管理
├── cache_manager.py      # 缓存管理
├── common_utils.py       # 通用工具函数
├── ImageUnderstanding.py # 图像理解模块
├── best.pt              # YOLO模型文件
├── requirements.txt     # 依赖包列表
├── static/             # 静态资源
│   ├── css/           # 样式文件
│   ├── uploads/       # 上传文件存储
│   └── dome_show/     # 演示数据
└── templates/         # HTML模板
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA支持（可选，用于GPU加速）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd golf_analysis_kong_api-new_way
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置星火API**
编辑 `config.py` 文件，填入您的星火API配置：
```python
class SparkAPIConfig:
    APPID = "your_app_id"
    API_SECRET = "your_api_secret"
    API_KEY = "your_api_key"
```

5. **运行应用**
```bash
python app.py
```

6. **访问应用**
打开浏览器访问: http://localhost:5000

## 📖 使用指南

### 1. 上传视频
- 点击上传区域
- 支持格式：MP4、AVI、MOV
- 文件大小限制：100MB

### 2. 等待分析
- 系统会自动开始分析过程
- 实时显示分析进度
- 分析时间取决于视频长度

### 3. 查看结果
- 查看关键帧提取结果
- 阅读AI分析报告
- 下载分析结果

### 4. 历史记录
- 在历史页面查看所有分析记录
- 重新查看或下载历史结果

## 🔧 配置说明

### 主要配置项

#### Flask配置 (`config.py`)
```python
class FlaskConfig:
    UPLOAD_FOLDER = 'static/uploads'  # 上传文件夹
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 文件大小限制
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}  # 支持格式
```

#### 分析配置
```python
class GolfAnalysisConfig:
    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
    DISPLACEMENT_THRESHOLD = 20  # 位移阈值
    WINDOW_SIZE = 20  # 窗口大小
```

#### 星火API配置
```python
class SparkAPIConfig:
    APPID = "your_app_id"
    API_SECRET = "your_api_secret"
    API_KEY = "your_api_key"
    IMAGE_UNDERSTANDING_URL = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"
```

## 📁 项目结构详解

### 核心文件说明

- **`app.py`**：Flask应用主文件，包含所有路由和请求处理
- **`golf_analysis.py`**：高尔夫挥杆分析核心模块，包含姿态检测和动作识别
- **`golfswingsAssistant.py`**：AI助手模块，负责与星火API交互
- **`config.py`**：配置管理模块，集中管理所有配置项
- **`cache_manager.py`**：缓存管理模块，提高分析效率
- **`common_utils.py`**：通用工具函数模块

### 静态资源结构
```
static/
├── css/style.css          # 主样式文件
├── uploads/               # 用户上传文件
│   ├── standard/         # 标准动作参考
│   └── [user_uploads]/   # 用户上传文件
└── dome_show/            # 演示数据
    ├── demo_show.mp4     # 演示视频
    ├── key_frames/       # 关键帧
    └── result_video/     # 分析结果
```

## 🔍 API接口

### 主要接口

#### 1. 文件上传
```
POST /upload
Content-Type: multipart/form-data
```

#### 2. 进度查询
```
GET /progress/<subdir>/<filename>
```

#### 3. 结果查看
```
GET /results/<subdir>/<filename>
```

#### 4. AI分析
```
GET /ai_analysis/<subdir>/<filename>
```

#### 5. 文件下载
```
GET /download/<subdir>/<filename>
```

## 🎯 高尔夫动作识别

系统能够识别以下四个关键阶段：

1. **准备阶段 (Preparation)**
   - 识别挥杆准备动作
   - 分析站姿和握杆姿势

2. **上杆顶点 (Top of Backswing)**
   - 检测上杆最高点
   - 分析上杆角度和位置

3. **击球瞬间 (Impact)**
   - 识别击球时刻
   - 分析击球姿势和角度

4. **收杆完成 (Finish)**
   - 检测收杆动作
   - 分析收杆姿势和平衡

## 🤖 AI分析功能

### 分析内容
- **动作规范性**：与标准动作对比
- **姿势分析**：身体各部位角度分析
- **改进建议**：个性化的改进建议
- **技术要点**：关键技术要点说明

### 分析维度
- **身体角度**：肩部、肘部、腕部等角度
- **球杆角度**：球杆与地面角度
- **身体平衡**：重心分布和平衡性
- **动作流畅性**：动作连贯性和节奏

## 🔧 开发指南

### 添加新的分析维度
1. 在 `golf_analysis.py` 中添加新的计算函数
2. 在 `config.py` 中配置显示参数
3. 更新前端模板以显示新数据

### 自定义AI分析
1. 修改 `golfswingsAssistant.py` 中的分析逻辑
2. 调整星火API的请求参数
3. 更新分析结果的展示格式

### 扩展支持格式
1. 在 `config.py` 中添加新格式
2. 确保OpenCV支持该格式
3. 测试文件上传和处理

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 确保 `best.pt` 文件存在
   - 检查CUDA环境配置

2. **API调用失败**
   - 验证星火API配置
   - 检查网络连接

3. **文件上传失败**
   - 检查文件格式和大小
   - 确保上传目录有写权限

4. **分析结果异常**
   - 检查视频质量
   - 确保视频包含完整挥杆动作

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 基础挥杆分析功能
- AI智能分析集成
- 响应式Web界面

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 贡献流程
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue


---

**注意**：使用本系统前，请确保您拥有星火API的有效账号和密钥。 