import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 必须配置（每台电脑不同）----------
# 手动下载的 embedding 模型路径（绝对路径）
EMBEDDING_MODEL_PATH = r"D:/my_embedding_model/sentence-transformersparaphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_NAME = r"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# API Key（建议用环境变量，也可直接填字符串）
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")

# ---------- 可选配置（按需修改）----------
# 知识库文件夹（默认项目根目录下的 knowledge）
KNOWLEDGE_FOLDER = os.path.join(BASE_DIR, "knowledge")

# 模型名称
MODEL_NAME = "doubao-seed-2-0-lite-260215"

#请求url
URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# RAG 参数
TOP_K = 2
RAG_MIN_SCORE = 0.6
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_KNOWLEDGE_LENGTH = 1000

# 对话历史轮数
MAX_HISTORY_TURNS = 20


# 会话持久化数据库路径
DB_PATH = os.path.join(BASE_DIR, "conversations.db")