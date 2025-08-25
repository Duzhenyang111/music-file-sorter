"""
在代码中直接设置参数，加载模型并对文件夹内所有音频进行性别预测。
不依赖命令行参数，便于在 Python 环境或 IDE/Jupyter 中直接运行。
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from sort_song_gender.predict_folder import predict_folder_df





def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"未找到模型文件: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_DIR):
        print(f"未找到输入目录: {INPUT_DIR}")
        return

    df: pd.DataFrame = predict_folder_df(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        sr=SAMPLE_RATE,
        duration=DURATION_SEC,
    )

    if df.empty:
        print("未得到有效预测结果（可能所有文件解码失败或不支持的格式）。")
        return

    print(df.head())
    print(f"共 {len(df)} 条预测结果。")

    if OUTPUT_CSV:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"已保存到: {OUTPUT_CSV}")


if __name__ == "__main__":
    # ===== 在这里修改参数（支持三分类：男性/女性/纯音乐）=====
    MODEL_PATH: str = "gender_model_pro.joblib"  # 或二分类模型 "gender_model.joblib"
    INPUT_DIR: str = "."  # 待预测目录（递归）
    OUTPUT_CSV: Optional[str] = "predictions.csv"  # 设为 None 则不保存
    SAMPLE_RATE: int = 22050
    DURATION_SEC: float = 30.0
    # ====================================================
    main()


