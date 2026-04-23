
from pathlib import Path


def get_file_context(saved_path: Path) -> str:
    """统一处理上传文件的读取逻辑。"""
    if saved_path.suffix.lower() in {".docx", ".doc"}:
        # 按需导入，避免环境里没装 python-docx 时连整个应用入口都无法导入。
        from docx import Document

        document = Document(saved_path)
        return "\n".join([para.text for para in document.paragraphs])

    with open(saved_path, "r", encoding="utf-8") as file:
        return file.read()
