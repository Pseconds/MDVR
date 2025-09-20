import os
import numpy as np
import argparse
from audio_utils import read_wav, write_wav
from mic_geometry import parse_pos, load_mics, to_xy_array, compute_angles
from plot_utils import plot_angles
from mvdr_core import simulate_array_signals, estimate_cov_from_noise, build_steering_for_target, mvdr_weights, apply_mvdr
from stft_utils import stft, istft, _hann


def main():
    ap = argparse.ArgumentParser(description="MVDR")
    ap.add_argument('--mic_json', default='./mic_positions_example.json')  # 麦克风位置json文件
    ap.add_argument('--c', type=float, default=0.3430)  # 声速，单位m/s
    ap.add_argument('--target_wav', default='./data/clean/sp01.wav')  # 目标音频路径
    ap.add_argument('--target_pos', default='0,10')  # 目标位置，格式x,y
    ap.add_argument('--noise_wav', default='./data/noise/output_noise.wav')  # 噪声音频路径
    ap.add_argument('--noise_pos', default='0,5')  # 噪声位置，格式x,y
    ap.add_argument('--outdir', default='out')  # 输出文件夹
    ap.add_argument('--units', default='m')  # 单位显示
    ap.add_argument('--show', action='store_true')  # 是否弹窗显示图片
    ap.add_argument('--model', choices=['near','far'], default='near')  # 近场/远场模型
    ap.add_argument('--snr_db', type=float, default=0.0)  # 信噪比
    ap.add_argument('--ref_mic', type=int, default=0)  # 参考麦克风编号
    ap.add_argument('--n_fft', type=int, default=1024)  # STFT窗口长度
    ap.add_argument('--hop', type=int, default=256)  # STFT步长
    args = ap.parse_args()

    # 读取麦克风阵列信息，返回list，每个元素为dict，包含id,x,y
    mics = load_mics(args.mic_json)
    # 解析目标和噪声位置，返回(x, y)
    tgt = parse_pos(args.target_pos)
    noi = parse_pos(args.noise_pos)
    # 采样率强制16kHz
    sr = 16000
    # 采样率强制16kHz
    sr = 16000
    angles = compute_angles(mics, tgt, args.c)

    # 可视化阵列几何和角度
    os.makedirs(args.outdir, exist_ok=True)  # 创建输出目录
    plot_angles(mics, tgt, angles, args.units, outpng=os.path.join(args.outdir,'angles_target.png'), arrow_len=2, noise=noi)  # 只绘制角度箭头并显示噪声

    # 读取目标和噪声音频，返回采样率和波形（float32, shape=[样本数,]）
    sr_t, y_t = read_wav(args.target_wav)
    _, y_t = read_wav(args.target_wav)
    _, y_n = read_wav(args.noise_wav)
    L = max(len(y_t), len(y_n))
    if len(y_t) < L:
        y_t = np.pad(y_t, (0, L-len(y_t)))
    if len(y_n) < L:
        y_n = np.pad(y_n, (0, L-len(y_n)))

    # 计算时长
    tgt_dur = len(y_t) / sr
    noi_dur = len(y_n) / sr
    # 获取麦克风二维坐标数组，shape=[M,2]
    mics_xy = to_xy_array(mics)
    # 仿真阵列接收信号，输出：
    #   X_mix_t: shape=[M, N]，混合信号
    #   X_tgt_t: shape=[M, N]，目标信号
    #   X_noi_t: shape=[M, N]，噪声信号
    #   freqs: shape=[F,]，频率轴
    X_mix_t, X_tgt_t, X_noi_t, freqs = simulate_array_signals(
        mics_xy, args.c, sr, y_t, y_n, tgt, noi, args.n_fft, args.hop,
        snr_db=args.snr_db, model=args.model, ref_mic=args.ref_mic
    )

    # 对混合信号做STFT，S_mix shape=[M, F, T]
    M = X_mix_t.shape[0]  # 麦克风数
    S_mix = []
    for m in range(M):
        Sm, _, _, _, _ = stft(X_mix_t[m], args.n_fft, args.hop)  # Sm: [F, T]
        S_mix.append(Sm)
    S_mix = np.stack(S_mix, axis=0)

    # 估计噪声协方差矩阵 Rnn，shape=[F, M, M]
    S_n_list = []
    for m in range(M):
        Sn, _, _, _, _ = stft(X_noi_t[m], args.n_fft, args.hop)
        S_n_list.append(Sn)
    S_n = np.stack(S_n_list, axis=0)
    Rnn = estimate_cov_from_noise(S_n)

    # 构造目标方向的导向矢量 a，shape=[F, M]
    a = build_steering_for_target(mics_xy, tgt, freqs, args.c, args.model, args.ref_mic)
    # 计算MVDR权重，w shape=[F, M]
    w = mvdr_weights(Rnn, a)
    # 应用MVDR权重，输出Y shape=[F, T]
    Y = apply_mvdr(S_mix, w)

    # 逆STFT得到时域波束输出 y_bf，shape=[N,]
    y_bf = istft(Y, _hann(args.n_fft), X_mix_t.shape[1], args.n_fft, args.hop)
    bf_path = os.path.join(args.outdir, 'mvdr.wav')
    write_wav(bf_path, sr, y_bf)
    print("Saved:", bf_path)

    # 保存每个麦克风的混合、目标、噪声信号
    for m in range(M):
        write_wav(os.path.join(args.outdir, f'mix_mic{m}.wav'), sr, X_mix_t[m])
        write_wav(os.path.join(args.outdir, f'target_mic{m}.wav'), sr, X_tgt_t[m])
        write_wav(os.path.join(args.outdir, f'noise_mic{m}.wav'), sr, X_noi_t[m])
    print("Saved mic WAVs.")

if __name__ == "__main__":
    main()
