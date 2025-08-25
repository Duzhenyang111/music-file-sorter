"""
在代码中设置参数：对文件夹内所有音频进行性别预测，并将文件移动到“男性/女性”子文件夹。
不依赖命令行参数，直接运行此脚本或在 Python 中 import 调用 main()。
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from sort_song_gender.predict_folder import predict_folder_df





def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_move(src: str, dst_dir: str) -> str:
    """将文件移动到目标目录，若重名则自动追加序号后缀，返回最终目标路径。"""
    ensure_dir(dst_dir)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dst = os.path.join(dst_dir, base)
    if not os.path.exists(dst):
        try:
            shutil.move(src, dst)
        except PermissionError as e:
            print(f"拒绝访问，跳过文件移动：{src} -> {dst}，原因：{e}")
            raise
        return dst
    # 处理重名
    idx = 1
    while True:
        cand = os.path.join(dst_dir, f"{name}_{idx}{ext}")
        if not os.path.exists(cand):
            try:
                shutil.move(src, cand)
            except PermissionError as e:
                print(f"拒绝访问，跳过文件移动：{src} -> {cand}，原因：{e}")
                raise
            return cand
        idx += 1


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"未找到模型文件: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_DIR):
        print(f"未找到输入目录: {INPUT_DIR}")
        return

    df = predict_folder_df(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        sr=SAMPLE_RATE,
        duration=DURATION_SEC,
    )
    if df.empty:
        print("未得到有效预测结果（可能所有文件解码失败或不支持的格式）。")
        return

    male_dir = os.path.join(TARGET_DIR, "男性")
    female_dir = os.path.join(TARGET_DIR, "女性")
    ensure_dir(male_dir)
    ensure_dir(female_dir)

    moved = 0
    for path, gender, _conf in df.itertuples(index=False, name=None):
        try:
            dst_dir = male_dir if gender == "男性" else female_dir
            final_dst = safe_move(path, dst_dir)
            moved += 1
        except Exception as e:
            print(f"移动失败：{path} -> {dst_dir}，原因：{e}")
    print(f"已移动 {moved} 个文件到：{male_dir} / {female_dir}")


if __name__ == "__main__":
    # ===== 在这里配置参数 =====
    MODEL_PATH: str = "gender_model.joblib"
    INPUT_DIR: str = r"\\10.254.97.43\cifs\user-fs\chenzihao\wangyi_music\未整理\dzy\其他\Charlie Lim"  # 待预测的根目录
    TARGET_DIR: str = r"C:\Users\duzhenyang\Desktop\music_store11\欧美男歌手"  # 输出根目录
    SAMPLE_RATE: int = 22050
    DURATION_SEC: float = 30.0
    # ========================
    main()


