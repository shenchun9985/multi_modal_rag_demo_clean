# src/utils/logger.py
import logging
import os
from datetime import datetime



# 1. 确定日志目录（项目根目录下的 logs 文件夹）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 回到项目根目录
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 2. 日志文件名：按日期命名，如 logs/app_2025-03-30.log
LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y-%m-%d')}.log")

# 3. 定义日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 4. 创建 root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # 总开关，接收所有级别

# 5. 文件处理器：记录所有级别日志到文件
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# 6. 控制台处理器：只输出 INFO 及以上级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# 7. 添加处理器
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)




# 限制第三方库的日志级别为 WARNING，避免 DEBUG 信息刷屏
third_party_libs = [
    'PIL',                 # 图片处理库
    'faiss',               # 向量检索库
    'urllib3',             # HTTP 连接池
    'asyncio',             # 异步IO库
    'httpcore',            # HTTP 核心库
    'httpx',               # HTTP 客户端
    'sentence_transformers', # 句子向量模型
    'python_multipart',    # 表单解析库
    'matplotlib',          # 如果使用过
    'torch',               # PyTorch
]

for lib in third_party_libs:
    logging.getLogger(lib).setLevel(logging.WARNING)





# 8. 提供便捷函数获取 logger
def get_logger(name: str):
    return logging.getLogger(name)