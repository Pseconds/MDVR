# 阵列几何与位置相关工具函数
import json  # 用于读取麦克风位置json
from typing import List, Tuple
import numpy as np  # 用于数值计算
import math  # 用于几何计算

def parse_pos(arg: str) -> Tuple[float, float]:
    """
    解析位置字符串，返回二维坐标元组。
    输入：arg (str)，格式为 'x,y'
    输出：(x, y) (float, float)
    """
    parts = [p.strip() for p in arg.split(',')]
    if len(parts) != 2:
        raise ValueError(f"Position must be 'x,y', got: {arg}")
    x = float(parts[0])  # x坐标
    y = float(parts[1])  # y坐标
    return x, y

def load_mics(path: str) -> List[dict]:
    """
    读取麦克风阵列配置文件，返回每个麦克风的字典列表。
    输入：path (str)，json文件路径
    输出：List[dict]，每个dict包含'id', 'x', 'y'
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'mics' in data:
        mics = data['mics']
    elif isinstance(data, list):
        mics = data
    else:
        raise ValueError("mic_json must be a list or a dict with key 'mics'")
    normed = []
    for i, m in enumerate(mics):
        mic_id = m.get('id', f"mic{i}")  # 麦克风编号
        x = float(m['x'])  # x坐标
        y = float(m['y'])  # y坐标
        normed.append({'id': mic_id, 'x': x, 'y': y})
    return normed

def to_xy_array(mics: List[dict]) -> np.ndarray:
    """
    将麦克风列表转为二维坐标数组。
    输入：mics (List[dict])，每个dict含x,y
    输出：xy (np.ndarray)，shape=[M,2]
    """
    M = len(mics)
    xy = np.zeros((M, 2), dtype=np.float32)
    for i, m in enumerate(mics):
        xy[i, 0] = m['x']  # 第i个麦克风x
        xy[i, 1] = m['y']  # 第i个麦克风y
    return xy

def unit_vec_from_positions(mics_xy: np.ndarray, src_xy: np.ndarray) -> np.ndarray:
    """
    计算从阵列中心指向目标的单位向量。
    输入：
        mics_xy (np.ndarray), shape=[M,2]，麦克风坐标
        src_xy (np.ndarray), shape=[2,]，目标坐标
    输出：
        u (np.ndarray), shape=[2,]，单位向量
    """
    centroid = mics_xy.mean(axis=0)  # 阵列中心
    v = src_xy - centroid  # 指向目标的向量
    n = np.linalg.norm(v) + 1e-12  # 防止除零
    return v / n

def compute_angles(mics: List[dict], src: Tuple[float, float], c: float) -> list:
    """
    计算每个麦克风指向目标的方位角、距离、时延。
    输入：
        mics (List[dict])，每个dict含x,y
        src (Tuple[float, float])，目标坐标
        c (float)，声速
    输出：
        angles (list[dict])，每个dict含id, azimuth_deg, range_m, tdoa_s, x, y
    """
    sx, sy = src
    out = []
    for m in mics:
        dx = sx - m['x']  # x方向距离
        dy = sy - m['y']  # y方向距离
        r = math.hypot(dx, dy)  # 欧氏距离
        az = math.degrees(math.atan2(dy, dx))  # 方位角（度）
        tdoa = r / c if c > 0 else float('nan')  # 时延
        out.append({'id': m['id'], 'azimuth_deg': az, 'range_m': r, 'tdoa_s': tdoa,
                    'x': m['x'], 'y': m['y']})
    return out