import pickle

from src.utils.logger import get_logger
logger = get_logger(__name__)

from src.utils.split_text_utils import split_text

import os


import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_PATH,EMBEDDING_MODEL_NAME, KNOWLEDGE_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, RAG_MIN_SCORE, MAX_KNOWLEDGE_LENGTH,INDEX_PATH,CHUNKS_PATH,TS_PATH
)

# ===================== 基础配置 =====================
# 镜像地址（加速下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'







# ===================== RAG 核心类 =====================
class ModernRAG:
    # 初始化：加载模型 + 构建索引
    def __init__(self, folder=None):
        self.folder = folder if folder is not None else KNOWLEDGE_FOLDER          # 知识库目录
        self.chunks = []              # 存放所有文本片段
        self.index = None             # FAISS 索引（检索用）

        logger.info("正在加载 Embedding 模型（本地）...")
        # 加载本地向量化模型
        self.model = SentenceTransformer(EMBEDDING_MODEL_PATH) if os.path.exists(EMBEDDING_MODEL_PATH) else SentenceTransformer(EMBEDDING_MODEL_NAME)

        # 一启动就自动：读取文档 → 切分 → 向量化 → 建索引
        self._build_index()

        # 初始化缓存（加载或构建索引）
        self._init_cache()


    # ==================== 缓存持久化相关 ====================
    def _get_knowledge_timestamp(self):
        """返回 knowledge 文件夹下所有文件的最新修改时间（字符串）"""
        if not os.path.exists(self.folder):
            return "0"
        latest = os.path.getmtime(self.folder)
        for root, dirs, files in os.walk(self.folder):
            for f in files:
                path = os.path.join(root, f)
                mtime = os.path.getmtime(path)
                if mtime > latest:
                    latest = mtime
        return str(latest)

    def _init_cache(self):
        index_path = INDEX_PATH
        chunks_path = CHUNKS_PATH
        ts_path = TS_PATH

        current_ts = self._get_knowledge_timestamp()

        # 检查缓存是否有效
        cache_ok = (os.path.exists(index_path) and
                    os.path.exists(chunks_path) and
                    os.path.exists(ts_path))
        if cache_ok:
            with open(ts_path, 'r') as f:
                saved_ts = f.read().strip()
            if saved_ts == current_ts:
                try:
                    self.index = faiss.read_index(index_path)
                    with open(chunks_path, 'rb') as f:
                        self.chunks = pickle.load(f)
                    logger.info(f"加载缓存成功，共 {len(self.chunks)} 个片段")
                    return
                except Exception as e:
                    logger.warning(f"加载缓存失败：{e}")

        # 缓存无效，重新构建
        logger.info("构建索引...")
        self._build_index()
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(ts_path, 'w') as f:
            f.write(current_ts)
        logger.info("索引已保存")



    # ==========================================
    # 第一步：读取文档 + 文本切分（最基础）   返回句子列表
    # 作用：把 txt 文件读出来，切成一句一句话
    # ==========================================
    def _load_chunks(self):
        chunks = []  # 用来存所有句子

        # 检查文件夹是否存在
        if not os.path.exists(self.folder):
            raise FileNotFoundError(f"知识库文件夹不存在: {self.folder}")
        # 遍历 knowledge 文件夹里的所有文件
        for filename in os.listdir(self.folder):
            if not filename.endswith('.txt'):  # 只处理 txt 文件
                continue
            filepath = os.path.join(self.folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            text_chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for chunk in text_chunks:
                chunks.append((chunk, filename))  # 存储 (文本, 文件名)
        return chunks

    # ==========================================
    # 第二步：构建索引（核心）
    # 1. 读取并切分文本
    # 2. 把所有句子向量化并归一化
    # 3. 存入 FAISS 内积索引（用于余弦相似度检索）
    # ==========================================
    def _build_index(self):
        logger.info("加载文档并切分...")
        self.chunks = self._load_chunks()  # 调用上面的函数，拿到所有句子

        # print("=== 所有切分后的句子 ===")
        # for i, c in enumerate(self.chunks):
        #     print(f"{i}: {c}")

        if not self.chunks:
            raise ValueError("knowledge 文件夹无有效txt文件")

        logger.info(f"共 {len(self.chunks)} 个片段")

        logger.info("向量化中...")
        # 把所有文本句子 → 变成数字向量（计算机能看懂的形式）
        vectors = self.model.encode(self.chunks)

        # ---------- 改进1：使用余弦相似度 ----------
        # 对向量进行 L2 归一化，这样内积就等于余弦相似度
        faiss.normalize_L2(vectors)

        # 创建内积索引（IndexFlatIP 计算余弦相似度）        扁平（暴力）内积索引
        self.index = faiss.IndexFlatIP(vectors.shape[1])  #IP:InnerProduct内积
        # 把归一化后的向量存入索引
        self.index.add(vectors.astype('float32'))

        logger.info("索引构建完成（使用余弦相似度）")

    # ==========================================
    # 第三步：检索（最关键）
    # 输入：你的问题
    # 输出：最相似的知识库句子列表，每个元素为 (相似度, 句子)
    # ==========================================
    def retrieve(self, query, top_k=TOP_K):
        # 1. 把你的问题也向量化
        q_vec = self.model.encode([query])  #返回numpy.ndarray类实例
        # 2. 对查询向量做同样的归一化
        faiss.normalize_L2(q_vec)

        # 3. 在 FAISS 里搜索最相似的 top_k 个句子（返回的是余弦相似度）
        similarities, indices = self.index.search(q_vec.astype('float32'), top_k)
        '''
        similarities：相似度分数
        indices：对应的句子编号
        '''

        results = []
        # 4. 把相似度 + 原文打包返回
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1:
                # sim 已经是余弦相似度，范围 [-1, 1]，越大越相关
                text,filename = self.chunks[idx]
                results.append((sim, text, filename))

        return results

    def get_knowledge(self, query, top_k=TOP_K, min_score=RAG_MIN_SCORE,
                      max_length=MAX_KNOWLEDGE_LENGTH):
        """返回纯知识文本（含编号和相似度），供大模型参考"""
        results = self.retrieve(query, top_k)
        if not results:
            return "",[]
        formatted = []
        details = []
        for idx, (score, text, filename) in enumerate(results, start=1):
            if score < min_score:
                continue
            if len(text) > max_length:
                text = text[:max_length] + "..."
            formatted.append(f"[{idx}].(相似度{score:.4f}) {text}")
            details.append((idx,filename,score))
        if not formatted:
            return "",[]
        knowledge_text = "参考资料:\n" + "\n".join(formatted)
        return knowledge_text, details


    # ==========================================
    # 第四步：回答（整理结果输出 + 智能过滤）
    # 功能：
    #   - 如果只有一条结果，直接返回。
    #   - 如果有两条及以上，判断第二条是否明显差于第一条。
    #     若差太多（比例<阈值 或 绝对差>阈值），则只返回第一条并提示忽略低质量结果。
    #     否则全部返回。
    # ==========================================
    def answer(self, query):
        # 调用检索，拿到相关片段
        results = self.retrieve(query)

        if not results:
            return "资料中没有相关信息。"

        # 如果只有一个结果，直接返回
        if len(results) == 1:
            score, chunk = results[0]
            return f"【最相关片段】(相似度: {score:.3f})\n{chunk}"

        # 有两个及以上结果的情况
        first_score, first_chunk = results[0]
        second_score, second_chunk = results[1]

        # ---------- 改进2：智能过滤 ----------
        # 判断第二个结果是否明显差于第一个
        is_second_bad = (second_score < first_score * 0.5) or \
                        ((first_score - second_score) > 0.2)

        if is_second_bad:
            # 只返回第一个，并给出提示
            return (f"【最相关片段】(相似度: {first_score:.3f})\n{first_chunk}\n\n"
                    f"(另一个结果相似度 {second_score:.3f} 过低，已忽略)")
        else:
            # 两个结果都比较相关，全部返回
            out = "根据资料找到以下相关内容：\n"
            for i, (score, chunk) in enumerate(results, 1):
                out += f"\n【片段{i}】(相似度: {score:.3f})\n{chunk}\n" + "-" * 40 + "\n"
            return out


# ===================== 3. 运行入口 =====================
if __name__ == "__main__":
    rag = ModernRAG(KNOWLEDGE_FOLDER)
    logger.info("RAG 已启动（纯本地，余弦相似度 + 智能过滤）")

    # 循环聊天
    while True:
        q = input("\n你: ").strip()
        if q.lower() == "q":
            break
        if not q:
            continue
        print("助手:", rag.answer(q))
