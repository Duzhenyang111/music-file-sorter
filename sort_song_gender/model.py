"""
大数据集音频性别分类：快速特征提取 + 并行训练 + 模型持久化

功能
- 并行提取音频特征（支持 mp3/wav/flac 等），对大数据集友好
- 采用 kaiser_fast 重采样，限定前 N 秒提升速度
- 使用聚合统计（mean/std）压缩时序，特征维数小且稳定
- Sklearn Pipeline（StandardScaler + 逻辑回归），一键保存/加载
- CLI：训练/预测

使用示例
训练：
  python -m sort_song_gender.model train \
    --male_dir "\\\\?\\UNC\\server\\share\\...\\欧美男歌手" \
    --female_dir "\\\\?\\UNC\\server\\share\\...\\欧美女歌手" \
    --model_out "gender_model.joblib" \
    --n_jobs 8 --duration 30
python -m sort_song_gender.model train --male_dir "\\\\?\\UNC\\10.254.97.43\\cifs\\user-fs\\chenzihao\\wangyi_music\\0722\\欧美男歌手" --female_dir "\\\\?\\UNC\\10.254.97.43\\cifs\\user-fs\\chenzihao\\wangyi_music\\0722\\欧美女歌手" --model_out "gender_model.joblib" --n_jobs 8 --duration 30
预测：
  python -m sort_song_gender.model predict \
    --model "gender_model.joblib" \
    --audio "C:\\path\\to\\song.mp3"
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator

import numpy as np
import librosa
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def _is_audio(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTS


def iter_files(base_path: str) -> Generator[str, None, None]:
    for root, _dirs, files in os.walk(base_path):
        for name in files:
            if _is_audio(name):
                yield os.path.join(root, name)


@dataclass
class FeatureConfig:
    sample_rate: int = 22050
    duration: Optional[float] = 30.0  # 仅取前 N 秒，可显著提速；None 表示全部
    n_mfcc: int = 20
    n_fft: int = 2048
    hop_length: int = 512
    fmin: float = 20.0
    fmax: Optional[float] = None  # None 使用 sr/2


def compute_feature_vector(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    # 基础谱特征
    S = np.abs(librosa.stft(y=y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        fmin=cfg.fmin, fmax=cfg.fmax, power=2.0
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=cfg.n_mfcc)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=cfg.hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=cfg.hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, hop_length=cfg.hop_length)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, hop_length=cfg.hop_length)

    # 统计聚合（mean/std），使得变长音频 -> 定长向量
    def agg(x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])

    feats: List[np.ndarray] = [
        agg(mfcc),
        agg(chroma),
        agg(zcr),
        agg(centroid),
        agg(bandwidth),
        agg(rolloff),
    ]
    return np.concatenate(feats, axis=0)


def extract_features(audio_path: str, cfg: FeatureConfig) -> Optional[np.ndarray]:
    try:
        # 兼容 Windows 长路径前缀，比如 \\?\UNC\server\share\...
        def _normalize_unc_path(p: str) -> str:
            if os.name == "nt":
                if p.startswith("\\\\?\\UNC\\"):
                    # \\?\UNC\server\share -> \\server\share
                    return "\\\\" + p[len("\\\\?\\UNC\\"):]
                if p.startswith("\\\\?\\"):
                    # \\?\C:\path -> C:\path
                    return p[len("\\\\?\\"):]
            return p

        audio_path = _normalize_unc_path(audio_path)

        def _load_audio(path: str) -> Tuple[np.ndarray, int]:
            # 尝试 1：重采样 + kaiser_fast（最快）
            try:
                y1, sr1 = librosa.load(
                    path,
                    sr=cfg.sample_rate,
                    mono=True,
                    res_type="kaiser_fast",
                    duration=cfg.duration,
                )
                if y1.size > 0:
                    return y1, int(sr1)
            except Exception:
                pass
            # 尝试 2：不重采样（sr=None），由后续特征计算自行处理
            try:
                y2, sr2 = librosa.load(
                    path,
                    sr=None,
                    mono=True,
                    duration=cfg.duration,
                )
                if y2.size > 0:
                    return y2, int(sr2)
            except Exception:
                pass
            # 失败
            return np.array([], dtype=np.float32), 0

        y, sr = _load_audio(audio_path)
        if y.size == 0:
            return None
        # 短片段补零，避免谱特征产生 NaN
        if y.size < cfg.n_fft:
            pad = cfg.n_fft - y.size
            y = np.pad(y, (0, pad), mode="constant")
        if cfg.fmax is None:
            cfg = FeatureConfig(
                sample_rate=cfg.sample_rate,
                duration=cfg.duration,
                n_mfcc=cfg.n_mfcc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                fmin=cfg.fmin,
                fmax=float(sr) / 2.0,
            )
        vec = compute_feature_vector(y, int(sr), cfg)
        # 替换非有限值，避免后续全部被丢弃
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(vec)):
            return None
        return vec
    except Exception:
        return None


def build_dataset(male_dir: str, female_dir: str, cfg: FeatureConfig, n_jobs: int = 8) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    male_paths = [p for p in iter_files(male_dir)] if male_dir else []
    female_paths = [p for p in iter_files(female_dir)] if female_dir else []

    paths = male_paths + female_paths
    labels = [1] * len(male_paths) + [0] * len(female_paths)  # 1=男性, 0=女性

    if len(paths) == 0:
        raise ValueError("未发现任何音频文件，请确认输入目录与后缀是否正确")

    # 并行提取
    feats = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(extract_features)(p, cfg) for p in tqdm(paths, desc="提取特征", ncols=80)
    )

    X: List[np.ndarray] = []
    y: List[int] = []
    kept: List[str] = []
    for f, label, path in zip(feats, labels, paths):
        if f is None or not np.all(np.isfinite(f)):
            continue
        X.append(f)
        y.append(label)
        kept.append(path)

    if len(X) == 0:
        raise RuntimeError("所有音频特征均提取失败，请检查依赖与文件格式")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), kept


def train_and_save(X: np.ndarray, y: np.ndarray, model_out: str) -> Pipeline:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                solver="saga", max_iter=2000, n_jobs=-1, class_weight="balanced"
            ),
        ),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"验证集准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["女性", "男性"]))

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    dump(clf, model_out)
    print(f"模型已保存: {model_out}")
    return clf


def predict_file(model_path: str, audio_path: str, cfg: FeatureConfig) -> Tuple[str, float]:
    clf: Pipeline = load(model_path)
    feat = extract_features(audio_path, cfg)
    if feat is None:
        raise RuntimeError("音频特征提取失败")
    feat = feat.reshape(1, -1)
    prob = None
    if hasattr(clf[-1], "predict_proba"):
        prob = clf.predict_proba(feat)
        pred = int(np.argmax(prob, axis=1)[0])
        conf = float(np.max(prob))
    else:
        pred = int(clf.predict(feat)[0])
        conf = 0.0
    gender = "男性" if pred == 1 else "女性"
    return gender, conf


def main() -> None:
    parser = argparse.ArgumentParser(description="快速音频性别分类：训练/预测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="训练并保存模型")
    p_train.add_argument("--male_dir", type=str, required=True, help="男性目录（递归）")
    p_train.add_argument("--female_dir", type=str, required=True, help="女性目录（递归）")
    p_train.add_argument("--model_out", type=str, default="gender_model.joblib")
    p_train.add_argument("--n_jobs", type=int, default=8, help="并行线程数")
    p_train.add_argument("--duration", type=float, default=30.0, help="每段音频使用的秒数")
    p_train.add_argument("--sr", type=int, default=22050, help="重采样采样率")
    p_train.add_argument("--cache_npz", type=str, default=None, help="可选：保存特征为 npz")

    p_pred = sub.add_parser("predict", help="加载模型并预测单文件")
    p_pred.add_argument("--model", type=str, required=True, help="模型文件 .joblib")
    p_pred.add_argument("--audio", type=str, required=True, help="待预测音频路径")
    p_pred.add_argument("--duration", type=float, default=30.0)
    p_pred.add_argument("--sr", type=int, default=22050)

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = FeatureConfig(sample_rate=args.sr, duration=args.duration)
        X, y, kept = build_dataset(args.male_dir, args.female_dir, cfg, n_jobs=args.n_jobs)
        if args.cache_npz:
            os.makedirs(os.path.dirname(args.cache_npz) or ".", exist_ok=True)
            np.savez_compressed(args.cache_npz, X=X, y=y, files=np.array(kept))
            print(f"已缓存特征到: {args.cache_npz}")
        train_and_save(X, y, args.model_out)
    elif args.cmd == "predict":
        cfg = FeatureConfig(sample_rate=args.sr, duration=args.duration)
        gender, conf = predict_file(args.model, args.audio, cfg)
        if conf > 0:
            print(f"预测结果: {gender} (置信度: {conf:.4f})")
        else:
            print(f"预测结果: {gender}")


if __name__ == "__main__":
    main()
