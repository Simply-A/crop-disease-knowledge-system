"""双编码器封装：文本走 bge-m3，图像走 CLIP。

为什么不用同一个模型编码两种模态：
    纯 CLIP 的文本编码器对中文专业术语（如"稀释浓度""安全间隔期"）语义表达很弱，
    直接用它检索防治文本会明显掉点；而 bge 系列又完全不理解图像。
    因此采用"文本库 + 图像库"双链路，各自用最强编码器，在检索层做融合。

模型均为全局单例（首次调用时加载，之后复用），避免 FastAPI 每次请求重复加载。
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

import config


# ============================================================ 文本编码器
class TextEmbedder:
    """基于 sentence-transformers 的中文文本编码器。"""

    def __init__(self, model_name: str = config.TEXT_MODEL, device: str | None = config.DEVICE):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim: int = self.model.get_sentence_embedding_dimension()
        # bge-m3 不需要查询指令前缀；bge-zh-v1.5 系列建议加，以贴近官方用法。
        self._use_instruction = ("bge" in model_name.lower()) and ("m3" not in model_name.lower())

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """批量编码，返回 L2 归一化后的 float32 向量。"""
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vecs, dtype="float32")

    def encode_query(self, text: str) -> np.ndarray:
        """单条查询编码（文档检索场景下查询与文档使用不对称前缀）。"""
        query = f"为这个句子生成表示以用于检索相关文章：{text}" if self._use_instruction else text
        return self.encode([query])[0]


# ============================================================ 图像编码器
class ImageEmbedder:
    """基于 transformers CLIP 的图像编码器，同时保留文本编码能力备用。"""

    def __init__(self, model_name: str = config.IMAGE_MODEL, device: str | None = config.DEVICE):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim: int = self.model.config.projection_dim

    def encode_images(self, images: Sequence, batch_size: int = 16) -> np.ndarray:
        """编码 PIL.Image 列表（RGB），返回归一化向量。"""
        torch = self.torch
        out: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = list(images[i : i + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0)

    def encode_image(self, image) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_image_paths(self, paths: Iterable[str], batch_size: int = 16) -> np.ndarray:
        from PIL import Image

        return self.encode_images(
            [Image.open(p).convert("RGB") for p in paths], batch_size=batch_size
        )

    def encode_texts(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """CLIP 文本编码，仅用于跨模态对比实验（以文搜图）。

        注意：中文查询效果有限，正式检索请使用 TextEmbedder。
        """
        torch = self.torch
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0)


# ============================================================ 单例管理
_text_embedder: TextEmbedder | None = None
_image_embedder: ImageEmbedder | None = None


def get_text_embedder() -> TextEmbedder:
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = TextEmbedder()
    return _text_embedder


def get_image_embedder() -> ImageEmbedder:
    global _image_embedder
    if _image_embedder is None:
        _image_embedder = ImageEmbedder()
    return _image_embedder


def release() -> None:
    """释放模型，回收显存（用于实验脚本连续跑多组配置）。"""
    global _text_embedder, _image_embedder
    _text_embedder = None
    _image_embedder = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
