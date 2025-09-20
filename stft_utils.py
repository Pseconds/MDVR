# STFT/ISTFT 工具函数
import numpy as np  # 数值计算

def _hann(n: int) -> np.ndarray:
    """
    生成Hann窗。
    输入：n (int)，窗长
    输出：win (np.ndarray), shape=[n,]
    """
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)

def stft(y: np.ndarray, n_fft: int=1024, hop: int=256):
    """
    计算短时傅里叶变换（STFT）。
    输入：
        y (np.ndarray), shape=[N,]，时域信号
        n_fft (int)，窗长
        hop (int)，步长
    输出：
        S (np.ndarray), shape=[F, T]，复数频谱
        win (np.ndarray), shape=[n_fft,]，窗函数
        len(y) (int)，原始长度
        n_fft (int)
        hop (int)
    """
    y = np.asarray(y, dtype=np.float32)
    win = _hann(n_fft).astype(np.float32)
    pad = n_fft - hop  # 前后补零
    ypad = np.pad(y, (pad, pad), mode='constant')
    n_frames = 1 + (len(ypad) - n_fft) // hop  # 帧数
    S = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        seg = ypad[t * hop:t * hop + n_fft] * win  # 加窗
        fx = np.fft.rfft(seg, n=n_fft)  # FFT
        S[:, t] = fx
    return S, win, len(y), n_fft, hop

def istft(S: np.ndarray, win: np.ndarray, siglen: int, n_fft: int=1024, hop: int=256) -> np.ndarray:
    """
    逆短时傅里叶变换（ISTFT）。
    输入：
        S (np.ndarray), shape=[F, T]，频谱
        win (np.ndarray), shape=[n_fft,]，窗
        siglen (int)，原始信号长度
        n_fft (int)
        hop (int)
    输出：y (np.ndarray), shape=[siglen,]
    """
    n_frames = S.shape[1]
    ylen = hop * (n_frames - 1) + n_fft  # 重叠加后长度
    y = np.zeros(ylen, dtype=np.float32)
    wsum = np.zeros(ylen, dtype=np.float32)
    for t in range(n_frames):
        seg = np.fft.irfft(S[:, t], n=n_fft).astype(np.float32)  # 逆FFT
        y[t * hop:t * hop + n_fft] += seg * win  # 重叠加
        wsum[t * hop:t * hop + n_fft] += win ** 2
    nz = wsum > 1e-8
    y[nz] /= wsum[nz]  # 能量归一化
    pad = n_fft - hop
    y = y[pad:pad + siglen]  # 去除补零
    return y
