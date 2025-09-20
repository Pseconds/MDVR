# 音频读写相关工具函数
import librosa  # 音频加载
import soundfile as sf  # 音频保存
import numpy as np
from typing import Tuple

def read_wav(path: str) -> Tuple[int, np.ndarray]:
    """
    读取wav音频文件。
    输入：path (str)，文件路径
    输出：sr (int)，采样率；y (np.ndarray), shape=[N,]，单通道波形
    """
    y, sr = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
    return sr, y

def write_wav(path: str, sr: int, y: np.ndarray):
    """
    保存音频到wav文件。
    输入：
        path (str)
        sr (int)
        y (np.ndarray), shape=[N,]
    输出：无
    """
    sf.write(path, y, sr)
