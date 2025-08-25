import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

try:
    from sort_song_gender.model import FeatureConfig, extract_features, SUPPORTED_EXTS
except ModuleNotFoundError:
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(__file__))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from sort_song_gender.model import FeatureConfig, extract_features, SUPPORTED_EXTS


def is_audio(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTS


def collect_audio_files(base_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _dirs, files in os.walk(base_dir):
        for name in files:
            if is_audio(name):
                paths.append(os.path.join(root, name))
    return paths


def predict_folder_df(model_path: str, input_dir: str, sr: int = 22050, duration: float = 30.0) -> pd.DataFrame:
    """在 Python 中直接调用，返回 DataFrame: columns=[path, gender, confidence]"""
    clf = load(model_path)
    cfg = FeatureConfig(sample_rate=sr, duration=duration)

    audio_paths = collect_audio_files(input_dir)
    if not audio_paths:
        return pd.DataFrame({"path": [], "gender": [], "confidence": []})  # 空结果

    features: List[np.ndarray] = []
    keep_index: List[int] = []
    for idx, p in enumerate(tqdm(audio_paths, desc="提取特征", ncols=80)):
        vec = extract_features(p, cfg)
        if vec is None:
            continue
        features.append(vec)
        keep_index.append(idx)

    if not features:
        return pd.DataFrame({"path": [], "gender": [], "confidence": []})  # 全部失败

    X = np.asarray(features, dtype=np.float32)
    has_proba = hasattr(clf[-1], "predict_proba")
    if has_proba:
        prob = clf.predict_proba(X)
        pred = np.argmax(prob, axis=1)
        conf = np.max(prob, axis=1)
    else:
        pred = clf.predict(X)
        conf = np.zeros_like(pred, dtype=np.float32)

    rows = []
    for i, label in enumerate(pred):
        path = audio_paths[keep_index[i]]
        gender = "男性" if int(label) == 1 else "女性"
        rows.append((path, gender, float(conf[i])))
    return pd.DataFrame(rows, columns=["path", "gender", "confidence"])  # type: ignore[arg-type]


def predict_folder(model_path: str, input_dir: str, sr: int, duration: float, output: str | None) -> None:
    df = predict_folder_df(model_path, input_dir, sr, duration)
    if df.empty:
        print("未得到有效预测结果。")
        return
    if output:
        df.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"结果已保存到: {output}（共 {len(df)} 条）")
    else:
        for path, gender, confidence in df.itertuples(index=False, name=None):
            print(f"{path}\t{gender}\t{confidence:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="对文件夹内所有音频进行性别预测")
    parser.add_argument("--model", type=str, default="gender_model.joblib", help="模型文件路径")
    parser.add_argument("--input_dir", type=str, required=True, help="音频目录（递归扫描）")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径，可选")
    parser.add_argument("--sr", type=int, default=22050, help="重采样采样率")
    parser.add_argument("--duration", type=float, default=30.0, help="使用的音频秒数")
    args = parser.parse_args()

    predict_folder(args.model, args.input_dir, args.sr, args.duration, args.output)


if __name__ == "__main__":
    main()