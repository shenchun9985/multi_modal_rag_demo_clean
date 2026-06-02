from config import CHUNK_SIZE,CHUNK_OVERLAP


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    chunks = []

    def _split(part, sep_idx=0):
        if len(part) <= chunk_size:
            if part.strip():
                chunks.append(part.strip())
            return
        # 当前层的分隔符
        sep = separators[sep_idx] if sep_idx < len(separators) else ""
        if not sep:
            # 最后按固定长度切（有overlap）
            for i in range(0, len(part), chunk_size - overlap):
                piece = part[i:i + chunk_size]
                if piece.strip():
                    chunks.append(piece.strip())
            return
        # 按当前分隔符切
        segments = part.split(sep)
        current = ""
        for seg in segments:
            if len(current) + len(seg) + len(sep) <= chunk_size:
                current += (sep + seg) if current else seg
            else:
                if current:
                    _split(current, sep_idx + 1)
                current = seg
        if current:
            _split(current, sep_idx + 1)

    _split(text)
    return chunks