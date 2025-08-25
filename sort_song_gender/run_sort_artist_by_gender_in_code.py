"""
对“歌手为子文件夹”的目录进行性别归类：
- 读取每个歌手文件夹内的音频，批量预测其性别
- 以多数投票决定该歌手的性别
- 将整个歌手文件夹移动到 目标根目录/男性 或 目标根目录/女性

无需命令行参数，直接在文件内配置即可运行。
"""

from __future__ import annotations

import os
import shutil
from typing import List

try:
    from sort_song_gender.predict_folder import predict_folder_df
except ModuleNotFoundError:
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(__file__))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from sort_song_gender.predict_folder import predict_folder_df
    print(111111111111111111111111111111111111111111111111111111111111111111)





def is_dir(path: str) -> bool:
    try:
        return os.path.isdir(path)
    except Exception:
        return False


def list_artist_dirs(root: str) -> List[str]:
    result: List[str] = []
    for entry in os.scandir(root):
        try:
            if entry.is_dir(follow_symlinks=False):
                result.append(entry.path)
        except PermissionError:
            continue
    return result


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def merge_or_move_folder(src: str, dst: str) -> None:
    """移动或合并整个文件夹到 dst。
    - 若 dst 已存在，则合并内容；对重名文件按 _1, _2 追加序号
    - 跨盘符/网络共享使用 shutil.move 以提高兼容性
    """
    if not is_dir(src):
        raise ValueError(f"源不是文件夹：{src}")

    dst_parent = os.path.dirname(dst) or "."
    os.makedirs(dst_parent, exist_ok=True)

    if not os.path.exists(dst):
        # 目标不存在，直接移动整个目录
        try:
            shutil.move(src, dst)
        except PermissionError as e:
            print(f"拒绝访问，跳过文件夹移动：{src} -> {dst}，原因：{e}")
        return

    # 合并目录内容
    for name in os.listdir(src):
        s_path = os.path.join(src, name)
        d_path = os.path.join(dst, name)
        if os.path.isdir(s_path):
            merge_or_move_folder(s_path, d_path)
        else:
            base, ext = os.path.splitext(name)
            final_path = d_path
            idx = 1
            while os.path.exists(final_path):
                candidate = f"{base}_{idx}{ext}" if ext else f"{base}_{idx}"
                final_path = os.path.join(dst, candidate)
                idx += 1
            try:
                shutil.move(s_path, final_path)
            except PermissionError as e:
                print(f"拒绝访问，跳过文件移动：{s_path} -> {final_path}，原因：{e}")
    # 尝试删除空的源目录
    try:
        os.rmdir(src)
    except OSError:
        pass


def decide_gender_for_artist(artist_dir: str) -> str | None:
    df = predict_folder_df(
        model_path=MODEL_PATH,
        input_dir=artist_dir,
        sr=SAMPLE_RATE,
        duration=DURATION_SEC,
    )
    print(f"df:{df}")
    if df.empty:
        return None
    male_votes = int((df["gender"] == "男性").sum())
    female_votes = int((df["gender"] == "女性").sum())
    instrumental_votes = int((df["gender"] == "纯音乐").sum())
    print(f"男性票数：{male_votes}, 女性票数：{female_votes}, 纯音乐票数：{instrumental_votes}")

    # 多数票决定。若纯音乐占多数，则归为纯音乐；否则在男女中比较。
    if instrumental_votes > max(male_votes, female_votes):
        return "纯音乐"
    if male_votes > female_votes:
        return "男性"
    if female_votes > male_votes:
        return "女性"
    return TIE_BREAK


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"未找到模型文件: {MODEL_PATH}")
        return
    if not os.path.exists(SOURCE_ROOT):
        print(f"未找到输入根目录: {SOURCE_ROOT}")
        return

    male_root = os.path.join(TARGET_ROOT, "欧美男歌手")
    female_root = os.path.join(TARGET_ROOT, "欧美女歌手")
    instrumental_root = os.path.join(TARGET_ROOT, "纯音乐")
    ensure_dir(male_root)
    ensure_dir(female_root)
    ensure_dir(instrumental_root)

    artist_dirs = list_artist_dirs(SOURCE_ROOT)
    print(f"发现歌手文件夹：{len(artist_dirs)} 个")

    moved = 0
    skipped = 0
    for artist_dir in artist_dirs:
        artist_name = os.path.basename(artist_dir)
        try:
            gender = decide_gender_for_artist(artist_dir)
            if gender is None:
                print(f"跳过（无可用音频）：{artist_name}")
                skipped += 1
                continue
            if gender == "男性":
                target_dir = os.path.join(male_root, artist_name)
            elif gender == "女性":
                target_dir = os.path.join(female_root, artist_name)
            else:
                target_dir = os.path.join(instrumental_root, artist_name)
            merge_or_move_folder(artist_dir, target_dir)
            moved += 1
            print(f"已归类 {artist_name} -> {gender}")
        except Exception as e:
            print(f"处理失败：{artist_name}，原因：{e}")
    print(f"完成。移动歌手文件夹 {moved} 个，跳过 {skipped} 个。")


if __name__ == "__main__":
    # ===== 在这里配置参数 =====
    MODEL_PATH: str = "gender_model_pro.joblib"
    SOURCE_ROOT: str = r"\\?\UNC\aistor.ztgame.com\cifs\user-fs\chenzihao\wangyi_music\0825\其他" # 包含“歌手”子文件夹的根目录
    TARGET_ROOT: str = r"\\?\UNC\aistor.ztgame.com\cifs\user-fs\chenzihao\wangyi_music\0825"   # 输出根目录
    SAMPLE_RATE: int = 22050
    DURATION_SEC: float = 10.0
    TIE_BREAK: str = "男性"  # 当男女票数相同如何处理："男性" 或 "女性"；若三者都相等，仍使用该值
    # ========================
    main()


