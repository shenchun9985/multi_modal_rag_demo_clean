import gradio as gr
import re
import time
import json
import tempfile
from src.core.multi_modal_chat_bot import MultiModalChatBot
from src.core.simple_rag import ModernRAG
from src.utils.db_utils import SessionDB
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_ui_blocks():
    logger.info("正在启动RAG知识库...")
    rag = ModernRAG()
    bot = MultiModalChatBot(rag=rag, rag_top_k=3, rag_min_score=0.6)
    db = SessionDB()

    # 辅助函数
    def get_session_choices():
        sessions = db.get_all_sessions()
        return [(s["session_id"], s["title"]) for s in sessions]

    def refresh_session_dropdown():
        return gr.update(choices=get_session_choices())

    # 将 bot._chat_history 转换为 Gradio 消息格式（字典列表）
    def bot_history_to_gradio():
        gradio_messages = []
        for msg in bot.get_history():
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                # 提取文本
                text_parts = [item["text"] for item in content if item.get("type") == "text"]
                content = " ".join(text_parts) if text_parts else "[非文本内容]"
            gradio_messages.append({"role": role, "content": content})
        return gradio_messages

    # 加载会话（切换时调用）
    def load_session(session_id):
        if not session_id:
            return [], gr.update(), None
        history = db.load_history(session_id, max_turns=bot.MAX_HISTORY_TURNS)
        # 清空 bot 内存历史并加载新历史
        bot.clear_history()
        for msg in history:
            # 注意：content 已经是纯文本
            bot._chat_history.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})
        gradio_history = bot_history_to_gradio()
        return gradio_history, gr.update(value=session_id), session_id

    # 新建会话
    def new_session():
        new_id = db.create_session()
        bot.clear_history()
        new_choices = get_session_choices()
        return [], gr.update(choices=new_choices, value=new_id), new_id

    # 删除会话
    def delete_session(session_id):
        if not session_id:
            return [], gr.update(), None
        db.delete_session(session_id)
        sessions = db.get_all_sessions()
        if not sessions:
            bot.clear_history()
            return [], gr.update(choices=[], value=None), None
        first = sessions[0]["session_id"]
        # 重新加载第一个会话
        hist = db.load_history(first, max_turns=bot.MAX_HISTORY_TURNS)
        bot.clear_history()
        for msg in hist:
            bot._chat_history.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})
        gradio_history = bot_history_to_gradio()
        dropdown_update = gr.update(choices=get_session_choices(), value=first)
        return gradio_history, dropdown_update, first

    # 聊天响应（不再重复加载历史）
    def respond(message, image_path, history, session_id, save_to_db):
        if not session_id:
            session_id = db.create_session(title="默认会话")
        actual_session = session_id if save_to_db else f"temp_{int(time.time())}"

        # 保存用户消息到数据库（如果需要）
        if save_to_db:
            user_content = f"📷 {message}" if image_path else message
            db.save_message(actual_session, "user", user_content)

        # 添加用户消息到 UI 历史
        history.append({"role": "user", "content": message if not image_path else f"📷 {message}"})
        yield history, gr.update()

        # 直接调用 bot.chat（内部使用内存中的 bot._chat_history，不需要重新加载）
        # 添加 assistant 占位
        history.append({"role": "assistant", "content": ""})
        yield history, gr.update()

        full = ""
        first_chunk = True
        start_time = time.time()
        for chunk in bot.chat(image_path, message):
            if first_chunk:
                first_chunk = False
                logger.info(f"首字延迟: {(time.time()-start_time)*1000:.0f}ms")
            full += chunk
            display = re.sub(r'\n{3,}', '\n\n', full)
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = display
            yield history, gr.update()

        if save_to_db:
            db.save_message(actual_session, "assistant", full)
            # 刷新下拉框（可选，但不阻塞）
            yield history, gr.update(choices=get_session_choices())
        else:
            yield history, gr.update()

    # 导出会话
    def export_session(session_id):
        if not session_id:
            return None
        history = db.load_history(session_id, max_turns=1000)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            return f.name

    # UI 布局
    css = """
    .gradio-container {max-width: 1200px; margin: auto;}
    .sidebar {background-color: #f5f5f5; padding: 15px; border-radius: 10px;}
    .main-chat {padding: 10px;}
    """
    with gr.Blocks(title="多模态记忆机器人") as demo:
        gr.Markdown("# 📷 多模态记忆机器人（支持图片+文字+知识库）")
        with gr.Row():
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.Markdown("### 会话管理")
                initial_choices = get_session_choices()
                session_list = gr.Dropdown(
                    label="已有会话",
                    choices=initial_choices,
                    interactive=True,
                    value=initial_choices[0][0] if initial_choices else None
                )
                with gr.Row():
                    new_btn = gr.Button("➕ 新建会话")
                    del_btn = gr.Button("🗑️ 删除会话")
                refresh_btn = gr.Button("🔄 刷新列表")
                save_toggle = gr.Checkbox(label="保存到数据库", value=True, info="关闭后本次聊天不会保存")
                export_btn = gr.Button("📥 导出当前会话")
                output_file = gr.File(label="下载导出文件", visible=False)

            with gr.Column(scale=3, elem_classes="main-chat"):
                chatbot = gr.Chatbot(label="对话记录")
                with gr.Row():
                    image_input = gr.Image(type="filepath", label="上传图片（可选）")
                    question_input = gr.Textbox(label="你的问题", lines=2)
                submit_btn = gr.Button("发送")

        current_session = gr.State(initial_choices[0][0] if initial_choices else None)

        # 事件绑定
        session_list.change(load_session, inputs=[session_list], outputs=[chatbot, session_list, current_session])
        new_btn.click(new_session, outputs=[chatbot, session_list, current_session])
        del_btn.click(delete_session, inputs=[current_session], outputs=[chatbot, session_list, current_session])
        refresh_btn.click(refresh_session_dropdown, outputs=[session_list])
        export_btn.click(export_session, inputs=[current_session], outputs=[output_file]).then(
            lambda: gr.update(visible=True), outputs=[output_file]
        )
        submit_btn.click(
            respond,
            inputs=[question_input, image_input, chatbot, current_session, save_toggle],
            outputs=[chatbot, session_list]
        )
        question_input.submit(
            respond,
            inputs=[question_input, image_input, chatbot, current_session, save_toggle],
            outputs=[chatbot, session_list]
        )

    demo.launch(server_name="127.0.0.1", css=css)

if __name__ == "__main__":
    run_ui_blocks()