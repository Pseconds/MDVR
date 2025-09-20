# 阵列几何与角度可视化工具
import math  # 数学计算
import matplotlib.pyplot as plt  # 绘图库

def plot_angles(mics: list, target: tuple, angles: list, units: str, outpng: str, noise: tuple, arrow_len: float = 0.08):
    """
    绘制每个麦克风指向目标的方位角箭头，并显示噪声位置。
    输入：
        mics (list[dict])：每个dict含id,x,y
        target (tuple)：目标位置(x, y)
        angles (list[dict])：每个dict含azimuth_deg等
        units (str)：坐标单位
        outpng (str)：输出图片路径
        arrow_len (float)：箭头长度
        noise (tuple)：噪声位置(x, y)，可选
    输出：无
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = [m['x'] for m in mics]
    ys = [m['y'] for m in mics]
    ax.scatter(xs, ys, marker='o', label='Mics')
    ax.scatter([target[0]], [target[1]], marker='^', label='Target')
    ax.scatter([noise[0]], [noise[1]], marker='s', label='Noise', color='red')
    ax.annotate('Noise', (noise[0], noise[1]), textcoords="offset points", xytext=(5,5), color='red')
    for m, a in zip(mics, angles):
        mx, my = m['x'], m['y']  # 麦克风坐标
        theta = math.radians(a['azimuth_deg'])  # 方位角（弧度）
        ux, uy = math.cos(theta), math.sin(theta)  # 单位方向向量
        ax.arrow(mx, my, ux*arrow_len, uy*arrow_len, head_width=arrow_len*0.25, length_includes_head=True)  # 画箭头
        ax.annotate(f"{a['id']}: {a['azimuth_deg']:.1f}°", (mx, my), textcoords="offset points", xytext=(5,5))  # 标注角度
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(f'X ({units})')
    ax.set_ylabel(f'Y ({units})')
    ax.set_title('Incidence Angles to Target (2D)')
    ax.legend()
    fig.savefig(outpng, bbox_inches='tight', dpi=200)
