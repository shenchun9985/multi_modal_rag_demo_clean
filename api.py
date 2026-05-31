from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from src.core.multi_model_chat_bot import MultiModelChatBot
import os
from src.core.simple_rag import ModernRAG

# 自动获取项目根目录下的 knowledge 文件夹
base_dir = os.path.dirname(os.path.abspath(__file__))
knowledge_path = os.path.join(base_dir, "knowledge")

app = FastAPI(title="我的多模态对话API",description='支持图文+RAG')

rag = ModernRAG(knowledge_path)

bot = MultiModelChatBot(rag=rag)


#定义请求体结构
class ChatRequest(BaseModel):
    image_path: Optional[str] = None
    question: str

#定义post接口
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    full_answer = ""
    for chunk in bot.chat(req.image_path,req.question):
        full_answer += chunk

    return {"answer": full_answer}