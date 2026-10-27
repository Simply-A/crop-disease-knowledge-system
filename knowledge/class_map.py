"""病害类别映射表。

作用：把 PlantVillage 的目录名 / 检测模型的类别名统一成系统内部的元数据三元组
(crop_type, disease_name, disease_id)，作为两个向量库共用的过滤字段。

关键约定：disease_name 必须与改进 YOLOv8 检测模型输出的类别名完全一致，
否则"检测结果 → 知识库检索"这一步的类别硬过滤会全部落空。
"""
from __future__ import annotations

import re

# disease_id 用英文短标识，便于做数据库主键与前端路由；
# disease_name 用中文全称，与检测模型类别表保持一致。
DISEASE_CLASSES: dict[str, dict] = {
    # PlantVillage 目录名 : 元数据
    "Apple___Apple_scab": {
        "disease_id": "apple_scab",
        "crop_type": "苹果",
        "disease_name": "苹果黑星病",
        "alias": ["apple scab", "黑星病", "疮痂病"],
        "pathogen": "Venturia inaequalis",
    },
    "Apple___Black_rot": {
        "disease_id": "apple_black_rot",
        "crop_type": "苹果",
        "disease_name": "苹果黑腐病",
        "alias": ["apple black rot", "黑腐病"],
        "pathogen": "Botryosphaeria obtusa",
    },
    "Cherry___Powdery_mildew": {
        "disease_id": "cherry_powdery_mildew",
        "crop_type": "樱桃",
        "disease_name": "樱桃白粉病",
        "alias": ["cherry powdery mildew", "白粉病"],
        "pathogen": "Podosphaera clandestina",
    },
    "Corn_(maize)___Common_rust_": {
        "disease_id": "corn_common_rust",
        "crop_type": "玉米",
        "disease_name": "玉米普通锈病",
        "alias": ["common rust", "普通锈病", "锈病"],
        "pathogen": "Puccinia sorghi",
    },
    "Grape___Black_rot": {
        "disease_id": "grape_black_rot",
        "crop_type": "葡萄",
        "disease_name": "葡萄黑腐病",
        "alias": ["grape black rot", "黑腐病"],
        "pathogen": "Guignardia bidwellii",
    },
    "Peach___Bacterial_spot": {
        "disease_id": "peach_bacterial_spot",
        "crop_type": "桃",
        "disease_name": "桃细菌性斑点病",
        "alias": ["bacterial spot", "细菌性斑点病", "细菌性穿孔病"],
        "pathogen": "Xanthomonas arboricola pv. pruni",
    },
    "Potato___Early_blight": {
        "disease_id": "potato_early_blight",
        "crop_type": "马铃薯",
        "disease_name": "马铃薯早疫病",
        "alias": ["potato early blight", "早疫病", "轮纹病"],
        "pathogen": "Alternaria solani",
    },
    "Potato___Late_blight": {
        "disease_id": "potato_late_blight",
        "crop_type": "马铃薯",
        "disease_name": "马铃薯晚疫病",
        "alias": ["potato late blight", "晚疫病", "马铃薯瘟"],
        "pathogen": "Phytophthora infestans",
    },
    "Tomato___Early_blight": {
        "disease_id": "tomato_early_blight",
        "crop_type": "番茄",
        "disease_name": "番茄早疫病",
        "alias": ["tomato early blight", "早疫病", "轮纹病"],
        "pathogen": "Alternaria solani",
    },
    "Tomato___Late_blight": {
        "disease_id": "tomato_late_blight",
        "crop_type": "番茄",
        "disease_name": "番茄晚疫病",
        "alias": ["tomato late blight", "晚疫病", "番茄疫病"],
        "pathogen": "Phytophthora infestans",
    },
}

# 反向索引：中文病名 / 别名 / disease_id → 标准条目键的集合。
# 用集合而非单值，是因为别名存在跨类别冲突（"早疫病"同时是番茄与马铃薯的别名），
# 遇到冲突时必须报歧义而不是猜一个——猜错会让检索过滤到错误病害的知识。
_NAME_INDEX: dict[str, set[str]] = {}


def _register(name: str | None, key: str) -> None:
    if not name:
        return
    _NAME_INDEX.setdefault(name.strip().lower(), set()).add(key)


for _key, _meta in DISEASE_CLASSES.items():
    _register(_meta["disease_name"], _key)
    _register(_meta["disease_id"], _key)
    for _a in _meta["alias"]:
        _register(_a, _key)


def _norm(dir_name: str) -> str:
    """归一化目录名：去尾部下划线、统一大小写，容忍 PlantVillage 的命名差异。"""
    return re.sub(r"_+$", "", dir_name.strip()).lower()


_NORM_INDEX = {_norm(k): k for k in DISEASE_CLASSES}


def _meta_of(key: str) -> dict:
    meta = DISEASE_CLASSES[key]
    return {
        "disease_id": meta["disease_id"],
        "crop_type": meta["crop_type"],
        "disease_name": meta["disease_name"],
    }


def resolve_dir(dir_name: str) -> dict | None:
    """按图像库目录名解析元数据；无法识别时返回 None。"""
    key = _NORM_INDEX.get(_norm(dir_name))
    return _meta_of(key) if key else None


def resolve_name(name: str) -> dict | None:
    """按中文病名 / 英文别名 / disease_id 解析元数据。

    名称未登记或存在歧义时返回 None，由调用方决定是报错还是原样保留。
    """
    keys = _NAME_INDEX.get(str(name or "").strip().lower())
    if not keys or len(keys) > 1:
        return None
    return _meta_of(next(iter(keys)))


def ambiguous_names() -> dict[str, list[str]]:
    """列出存在歧义的名称及其候选类别，便于排查映射问题。"""
    out: dict[str, list[str]] = {}
    for name, keys in _NAME_INDEX.items():
        if len(keys) > 1:
            out[name] = sorted(DISEASE_CLASSES[k]["disease_name"] for k in keys)
    return out


def all_names() -> list[str]:
    """返回全部标准病名，用于检测模型类别表与前端下拉框。"""
    return [m["disease_name"] for m in DISEASE_CLASSES.values()]


if __name__ == "__main__":
    # 本模块是库，不是脚本。直接运行时把映射表打出来：既证明文件本身可用，
    # 也能一眼看到跨作物的别名歧义——这是"检测结果 → 知识库检索"最易踩的坑。
    print(__doc__.strip())
    print()
    print("=" * 90)
    print("当前文件定位：类别映射库，无命令行入口。已登记类别如下。")
    print("=" * 90)
    print("  %-26s %-6s %-14s %s" % ("PlantVillage 目录名", "作物", "标准病名", "disease_id"))
    print("  " + "-" * 86)
    for _dir, _m in DISEASE_CLASSES.items():
        print("  %-26s %-6s %-14s %s" % (_dir, _m["crop_type"], _m["disease_name"], _m["disease_id"]))
    print()
    _amb = ambiguous_names()
    print("别名歧义（%d 条）—— 这些简称不能直接用于类别过滤，必须带作物前缀：" % len(_amb))
    if _amb:
        for _name, _owners in sorted(_amb.items()):
            print("  %-10s -> %s" % (_name, " / ".join(_owners)))
    else:
        print("  （无）")
    print()
    print("共 %d 个标准病名，可直接用作检测模型的类别表与前端下拉选项。" % len(all_names()))
    print()
    print("注：disease_name 必须与检测模型输出的类别名完全一致，否则类别硬过滤会全部落空且不报错。")
