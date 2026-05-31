from src.utils.logger import get_logger
logger = get_logger(__name__)


from PIL import Image
import base64
import io

def encode_image_to_base64(image_path,max_size=(512,512),quality=75):
    """

    :param image_path: 图片文件路径
    :param max_size: 缩略图最大尺寸（宽，高）
    :param quality: JPEG压缩质量（1-100）
    :return:
        成功时：（data_url,None),data_url 形如 “data:image/jpeg;base64,ⅹⅹⅹⅹ”
    """
    try:
        with Image.open(image_path) as img:
            #压缩分辨率
            img.thumbnail(max_size,Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            #判断是否需要保留透明
            if img.format =="PNG" and img.mode == "RGBA":
                #保留PNG透明，无损压缩
                img.save(buffer, format="PNG",optimize=True,compress_level=6)
                mime = 'image/png'
            else:
                #其他情况统一转 JPEG
                if img.mode in('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(buffer, format="JPEG",quality=quality)
                mime = 'image/jpeg'
            b64 = base64.b64encode(buffer.getvalue()).decode()
            data_url = f"data:{mime};base64,{b64}"
            return data_url,None
    except Exception as e:
        logger.warning(f"图片处理失败: {image_path}, 错误: {e}")
        return None,str(e)














