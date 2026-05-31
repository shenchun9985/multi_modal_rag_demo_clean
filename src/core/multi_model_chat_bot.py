

from src.utils.logger import get_logger

logger = get_logger(__name__)

import time
import requests
import json

from src.utils.image_utils import encode_image_to_base64
import os

from .prompt_builder import PromptBuilder
# 导入配置
from config import DOUBAO_API_KEY, MODEL_NAME, MAX_HISTORY_TURNS,TOP_K, RAG_MIN_SCORE,URL

class MultiModelChatBot:
    def __init__ (self,api_key=None,rag=None,rag_top_k=TOP_K,rag_min_score=RAG_MIN_SCORE,prompt_builder=None,
                  request_timeout=30,max_retries=1):
        self.api_key = api_key or os.getenv("DOUBAO_API_KEY") or DOUBAO_API_KEY
        if not self.api_key:
            raise ValueError("请设置DOUBAO_API_KEY的环境变量或在初始化时传入api_key")
        self._chat_history=[]
        self.rag = rag
        self.rag_top_k = rag_top_k
        self.rag_min_score = rag_min_score
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.request_timeout = request_timeout
        self.max_retries = max_retries  #预留，暂未使用
        # 注意：MAX_HISTORY_TURNS 作为类属性，但允许实例覆盖
        self.MAX_HISTORY_TURNS = MAX_HISTORY_TURNS
    def _call_api(self,model,messages,stream=True):
        """流式调用API，生成器产出文本片段。出错时抛出异常（由上层捕获）"""
        url = URL
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 5000,
            "stream": stream,
            "thinking": {"type": "disabled"}
        }
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                # 流式请求
                response = requests.post(url, headers=headers, json=payload,
                                         stream=stream, timeout=self.request_timeout)  # response:封装了 HTTP 响应信息的对象
                response.encoding = "utf-8"

                # 检查HTTP状态码
                if response.status_code != 200:
                    try:
                        error_detail = response.json().get("error", {}).get("message", "未知错误")
                    except Exception:
                        error_detail = response.text[:200]
                    raise Exception(f"API返回错误{response.status_code}: {error_detail}")

                # 逐行读取SSE数据
                for line in response.iter_lines(decode_unicode=True):
                    # response.iter_lines() = 从 response 里拿出 raw 数据流
                    # = 包装成迭代器返回
                    # 它不取状态码、不取文本、不取响应头，只取「底层数据流」，然后做成迭代器给你一行行读。
                    if line and line.startswith("data: "):
                        data_str = line[5:].lstrip()  # 去掉“data: ”前缀
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

                return

            except Exception as e:
                last_exception = e
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(1)  # 重试前等待
                continue

                # 所有重试都失败
            raise last_exception

    @staticmethod
    def _build_user_message(image_path,question):      #return 结果, 错误
        content = []
        if image_path and image_path != "":
            data_url,err = encode_image_to_base64(image_path)
            if err:
                return None,f"图片处理失败: {err}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}        #"url": "数据:image类型;base64,编码后的图片文本"
            })

        content.append({"type": "text","text": question})
        return {"role": "user", "content": content},None

    def _prune_old_images(self):
        """只保留最近一张图片，其余替换为文本提示（适用于单图连续追问场景）"""
        last_img_idx = -1
        for idx,msg in enumerate(reversed(self._chat_history)):
            if msg.get("role") == "user" and isinstance(msg.get("content"),list):
                if any(item.get("type") == "image_url" for item in msg["content"]):
                    last_img_idx = len(self._chat_history)-1-idx
                    break

        if last_img_idx == -1:
            return

        for idx,msg in enumerate(self._chat_history):
            if msg.get("role") != "user" or not isinstance(msg.get("content"),list) or idx ==last_img_idx:
                continue

            new_content = [item for item in msg["content"] if item.get("type") !="image_url"]
            new_content.append({"type": "text", "text": "[此处图片已被省略以节省记忆]"})
            msg["content"] = new_content



    def _save_history(self,user_msg,ai_msg):
        self._chat_history.append(user_msg)
        self._chat_history.append(ai_msg)
        #修剪，保留最近一张图片
        self._prune_old_images()
        #保留最近MAX_HISTORY_TURNS 轮（2*MAX_HISTORY_TURNS 条消息）
        max_messages = 2 * self.MAX_HISTORY_TURNS   #原理：当实例本身没有MAX_HISTORY_TURNS这个属性时，python会自动去他的类及父类
                                                    # 里面去查找，故self.也可以拿到类属性的值
        if len(self._chat_history) > max_messages:
            self._chat_history = self._chat_history[-max_messages:]

    def get_history(self):
        return self._chat_history.copy()

    def clear_history(self):
        self._chat_history = []


    #主入口    对外只暴露这个方法
    #最终两个目的：
    #1.yield chunk（给用户看答案）
    #2.self._save_history(...)（让下次能记住上下文）
    def chat(self,image_path,question):
        if not question or not question.strip():
            yield "请输入问题："
            return

        # 拿历史
        messages = self.get_history()

        # 1. 获取原始用户消息（用于保存历史，不含任何RAG前缀）
        user_msg_for_history, err = MultiModelChatBot._build_user_message(image_path, question)
        if err:
            yield err
            return

        # 2. 获取RAG知识（纯文本，临时使用）
        # 2. 获取RAG知识（使用改进后的 get_knowledge 方法）
        knowledge_text = ""
        if self.rag:
            try:
                knowledge_text = self.rag.get_knowledge(
                    question,
                    top_k=self.rag_top_k,
                    min_score=self.rag_min_score
                )
                if knowledge_text:
                    knowledge_text = knowledge_text + "\n\n"
                logger.info(f"RAG检索结果长度: {len(knowledge_text)} 字符")  # 只记录长度，避免日志过大

                # 输出文件名和相似度（用于定位资料）
                raw_results = self.rag.retrieve(question, top_k=self.rag_top_k)
                if raw_results:
                    logger.info(f"详细资料来源（共{len(raw_results)}条）:")
                    for idx, (score, text, filename) in enumerate(raw_results, start=1):
                        if score >= self.rag_min_score:
                            logger.info(f"  [{idx}] 文件: {filename}, 相似度: {score:.4f}")


            except Exception as e:
                logger.error(f"rag检索异常：{e}",exc_info=True)


        # 3. 构造本次请求用的用户消息（带知识前缀，但不保存）
        final_question = f"{knowledge_text}\n\n问题：{question}" if knowledge_text else \
            f"（注意：没有提供任何参考资料。请直接根据你自身知识回答，不要编造引用标记。）\n\n问题：{question}"
        user_msg_for_request, _ = MultiModelChatBot._build_user_message(image_path, final_question)

        #添加系统提示
        system_prompt = self.prompt_builder.build_system_prompt()
        if not messages or messages[0].get("role") != "system":
            messages.insert(0,system_prompt)
        else:
            messages[0] = system_prompt

        messages.append(user_msg_for_request)

        #调用大模型
        model = MODEL_NAME

        full_answer = ""
        try:
            #流式调用，每次获得一个文本片段
            for chunk in self._call_api(model,messages,stream=True):
                full_answer += chunk
                yield chunk
        except Exception as e:
            logger.error(f"API调用失败：{e}")
            yield f"请求出错：{e}"
            return


        #完整回答接收完毕后保存到历史
        ai_msg = {"role": "assistant","content": [{"type": "text","text": full_answer}]}
        self._save_history(user_msg_for_history,ai_msg)
