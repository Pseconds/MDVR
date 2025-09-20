# MVDR波束形成与仿真核心函数
import numpy as np  # 数值计算
from stft_utils import stft, istft, _hann  # STFT/ISTFT工具
from mic_geometry import unit_vec_from_positions  # 阵列几何工具

def steering_farfield(mics_xy: np.ndarray, u: np.ndarray, freqs: np.ndarray, c: float, ref: int=0) -> np.ndarray:
    """
    计算远场导向矢量（steering vector）。
    输入：
        mics_xy (np.ndarray), shape=[M,2]，麦克风坐标
        u (np.ndarray), shape=[2,]，单位方向向量
        freqs (np.ndarray), shape=[F,]，频率轴
        c (float)，声速
        ref (int)，参考麦克风编号
    输出：
        A (np.ndarray), shape=[M, F]，导向矢量
    """
    r = mics_xy - mics_xy[ref:ref+1]  # 相对参考麦克风的坐标差
    taus = - (r @ u.reshape(2,1)).squeeze(-1) / c  # 各麦克风的时延
    A = np.exp(-1j * 2*np.pi * freqs.reshape(1,-1) * taus.reshape(-1,1))  # 远场相移
    return A

def steering_nearfield(mics_xy: np.ndarray, src_xy: np.ndarray, freqs: np.ndarray, c: float, ref: int=0) -> np.ndarray:
    """
    计算近场导向矢量。
    输入：
        mics_xy (np.ndarray), shape=[M,2]
        src_xy (np.ndarray), shape=[2,]
        freqs (np.ndarray), shape=[F,]
        c (float)
        ref (int)
    输出：A (np.ndarray), shape=[M, F]
    """
    r = np.linalg.norm(mics_xy - src_xy.reshape(1,2), axis=1)  # 各麦克风到目标距离
    dr = r - r[ref]  # 相对参考麦克风的距离差
    phase = np.exp(-1j * 2*np.pi * freqs.reshape(1,-1) * (dr.reshape(-1,1)/c))  # 相移
    return phase

def simulate_array_signals(mics_xyz: np.ndarray, c: float, sr: int, target: np.ndarray, noise: np.ndarray,
                          target_pos: tuple, noise_pos: tuple, n_fft: int, hop: int, snr_db: float=0.0,
                          model: str='near', ref_mic: int=0):
    """
    仿真阵列接收信号（目标+噪声）。
    输入：
        mics_xyz (np.ndarray), shape=[M,2]，麦克风坐标
        c (float)，声速
        sr (int)，采样率
        target/noise (np.ndarray), shape=[N,]，目标/噪声波形
        target_pos/noise_pos (tuple)，目标/噪声位置
        n_fft (int)，STFT窗口
        hop (int)，STFT步长
        snr_db (float)，信噪比
        model (str)，'near'或'far'
        ref_mic (int)
    输出：
        X_mix_t, X_tgt_t, X_noi_t (np.ndarray), shape=[M, N]
        freqs (np.ndarray), shape=[F,]
    """
    freqs = (sr * np.arange(0, n_fft//2+1) / n_fft).astype(np.float32)  # 频率轴
    M = mics_xyz.shape[0]  # 麦克风数
    tgt_src = np.array([target_pos[0], target_pos[1]], dtype=np.float32)  # 目标位置
    noi_src = np.array([noise_pos[0], noise_pos[1]], dtype=np.float32)  # 噪声位置

    # 计算目标和噪声的导向矢量
    if model == 'far':
        u_t = unit_vec_from_positions(mics_xyz, tgt_src)
        A_t = steering_farfield(mics_xyz, u_t, freqs, c, ref_mic)
        u_n = unit_vec_from_positions(mics_xyz, noi_src)
        A_n = steering_farfield(mics_xyz, u_n, freqs, c, ref_mic)
    else:
        A_t = steering_nearfield(mics_xyz, tgt_src, freqs, c, ref_mic)
        A_n = steering_nearfield(mics_xyz, noi_src, freqs, c, ref_mic)

    def geom_tau_near(mic_xy, src_xy): return np.linalg.norm(mic_xy - src_xy) / c  # 近场时延
    def geom_tau_far(mic_xy, u, ref_xy): return - ((mic_xy - ref_xy) @ u) / c  # 远场时延
    ref_xy = mics_xyz[ref_mic]

    # 计算每个麦克风的相对时延
    if model == 'far':
        u_t = unit_vec_from_positions(mics_xyz, tgt_src)
        u_n = unit_vec_from_positions(mics_xyz, noi_src)
        taus_t_abs = np.array([geom_tau_far(mics_xyz[m], u_t, ref_xy) for m in range(M)], dtype=np.float32)
        taus_n_abs = np.array([geom_tau_far(mics_xyz[m], u_n, ref_xy) for m in range(M)], dtype=np.float32)
    else:
        taus_t_abs = np.array([geom_tau_near(mics_xyz[m], tgt_src) for m in range(M)], dtype=np.float32)
        taus_n_abs = np.array([geom_tau_near(mics_xyz[m], noi_src) for m in range(M)], dtype=np.float32)
    taus_t = taus_t_abs - taus_t_abs[ref_mic]  # 目标相对时延
    taus_n = taus_n_abs - taus_n_abs[ref_mic]  # 噪声相对时延

    # 对目标和噪声信号做STFT，shape=[F, T]
    S_t, _, L_t, _, _ = stft(target, n_fft, hop)
    S_n, _, L_n, _, _ = stft(noise, n_fft, hop)
    F, T = S_t.shape  # 频点数，帧数
    # 对齐噪声帧数
    if S_n.shape[1] < T:
        pad_T = T - S_n.shape[1]
        S_n = np.pad(S_n, ((0,0),(0,pad_T)), mode='wrap')
    elif S_n.shape[1] > T:
        S_n = S_n[:, :T]

    # 生成每通道的目标和噪声频谱，shape=[M, F, T]
    X_t = np.zeros((M, F, T), dtype=np.complex64)
    X_n = np.zeros((M, F, T), dtype=np.complex64)
    freqs = (sr * np.arange(0, n_fft//2+1) / n_fft).astype(np.float32)
    for m in range(M):
        phase_t = np.exp(-1j * 2*np.pi * freqs.reshape(-1,1) * taus_t[m])  # 目标相移
        phase_n = np.exp(-1j * 2*np.pi * freqs.reshape(-1,1) * taus_n[m])  # 噪声相移
        amp_t = np.abs(A_t[m:m+1, :])  # 目标幅度
        amp_n = np.abs(A_n[m:m+1, :])  # 噪声幅度
        X_t[m] = (S_t * phase_t) * (amp_t.T if amp_t.shape[1]==F else 1.0)
        X_n[m] = (S_n * phase_n) * (amp_n.T if amp_n.shape[1]==F else 1.0)

    # 按指定信噪比缩放噪声
    if np.isfinite(snr_db):
        Pt = np.mean(np.abs(X_t[0])**2) + 1e-12  # 目标功率
        Pn = np.mean(np.abs(X_n[0])**2) + 1e-12  # 噪声功率
        g = np.sqrt(Pt / (Pn * (10**(snr_db/10.0))))
        X_n *= g

    # 逆STFT得到时域信号，shape=[M, N]
    mix_mic, tgt_mic, noi_mic = [], [], []
    for m in range(M):
        yt = istft(X_t[m], _hann(n_fft), L_t, n_fft, hop)  # 目标信号
        yn = istft(X_n[m], _hann(n_fft), L_t, n_fft, hop)  # 噪声信号
        mix_mic.append(yt+yn)  # 混合信号
        tgt_mic.append(yt)
        noi_mic.append(yn)
    return np.array(mix_mic), np.array(tgt_mic), np.array(noi_mic), freqs

def estimate_cov_from_noise(X_n: np.ndarray) -> np.ndarray:
    """
    用噪声信号估计协方差矩阵。
    输入：X_n (np.ndarray), shape=[M, F, T]
    输出：Rnn (np.ndarray), shape=[F, M, M]
    """
    M, F, T = X_n.shape
    Rnn = np.zeros((F, M, M), dtype=np.complex64)
    for f in range(F):
        Nf = X_n[:, f, :]
        Rnn[f] = (Nf @ Nf.conj().T) / max(T,1)
    return Rnn

def mvdr_weights(Rnn: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    计算MVDR权重。
    输入：
        Rnn (np.ndarray), shape=[F, M, M]，噪声协方差
        a (np.ndarray), shape=[F, M]，导向矢量
    输出：w (np.ndarray), shape=[F, M]
    """
    F, M, _ = Rnn.shape
    w = np.zeros((F, M), dtype=np.complex64)
    for f in range(F):
        R = Rnn[f]
        lam = 1e-3 * (np.trace(R).real / max(M,1) + 1e-12)
        Rl = R + lam * np.eye(M, dtype=np.complex64)
        try:
            Rinv = np.linalg.inv(Rl)
        except np.linalg.LinAlgError:
            Rinv = np.linalg.pinv(Rl)
        af = a[f].reshape(M,1)
        denom = (af.conj().T @ Rinv @ af).item() + 1e-12
        w[f] = ((Rinv @ af) / denom).squeeze(1)
    return w

def build_steering_for_target(mics_xyz: np.ndarray, target_pos: tuple, freqs: np.ndarray, c: float, model: str, ref_mic: int) -> np.ndarray:
    """
    构造目标方向的导向矢量。
    输入：
        mics_xyz (np.ndarray), shape=[M,2]
        target_pos (tuple), (x, y)
        freqs (np.ndarray), shape=[F,]
        c (float)
        model (str)
        ref_mic (int)
    输出：A (np.ndarray), shape=[F, M]
    """
    tgt_src = np.array([target_pos[0], target_pos[1]], dtype=np.float32)
    if model == 'far':
        centroid = mics_xyz.mean(axis=0)
        u = (tgt_src - centroid); u = u / (np.linalg.norm(u)+1e-12)
        A = steering_farfield(mics_xyz, u, freqs, c, ref_mic)
    else:
        A = steering_nearfield(mics_xyz, tgt_src, freqs, c, ref_mic)
    return A.T  # [F,M]

def apply_mvdr(X_mix: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    应用MVDR权重，输出波束信号。
    输入：
        X_mix (np.ndarray), shape=[M, F, T]
        w (np.ndarray), shape=[F, M]
    输出：Y (np.ndarray), shape=[F, T]
    """
    F, M = w.shape
    _, F2, T = X_mix.shape
    Y = np.zeros((F, T), dtype=np.complex64)
    for f in range(F):
        Y[f] = w[f].conj().reshape(1,M) @ X_mix[:, f, :]
    return Y
