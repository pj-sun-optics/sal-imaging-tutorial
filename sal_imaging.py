"""
sal_imaging.py — 合成孔径激光雷达 (SAL) 成像最小可运行示例
================================================================

一份从零构建的、教学导向的 SAL 二维成像仿真。目的是把
"数据怎么采 -> 怎么算 -> 怎么变成一幅二维图像" 这条链路
用最少的代码讲清楚，每个函数对应成像流程中的一步。

成像流程：
    正演采集 raw echo  ->  距离向压缩(FFT)  ->  方位向压缩(后向投影 BP)  ->  取模出图

关键物理：
    距离向分辨率  δr = c / (2B)         由信号带宽 B 决定（激光载频高，B 易做大）
    方位向分辨率  δa = λR / (2L)        由合成孔径长度 L 决定（平台运动"攒"出来）
    两维用两种不同物理量分开：距离=回波时延，方位=相位历程/多普勒

注意：这是教学玩具，为看清原理牺牲了效率与保真度（逐像素BP、
忽略距离徙动、点目标近似）。理解原理足够，勿作精度基准。

依赖: numpy, matplotlib  (自聚焦演示额外用到 scipy)
"""

import numpy as np


# ----------------------------------------------------------------------
# 物理与系统参数
# ----------------------------------------------------------------------
class SALConfig:
    """SAL 系统与场景参数。默认值对应一个机载对地小场景。"""
    def __init__(self):
        self.c    = 3e8            # 光速 (m/s)
        self.lam  = 1.5e-6         # 波长 1.5 μm (光载频 ~200 THz)
        self.B    = 6e9            # 扫频带宽 (Hz) -> δr = c/2B = 25 mm
        self.T    = 1e-4           # 扫频周期 (s)
        self.Ns   = 400            # 每脉冲快时间采样数 (距离向)
        self.R0   = 1000.0         # 场景中心斜距 (m)
        self.Nx   = 600            # 航迹位置数 (方位向慢时间采样)
        self.dx   = 0.004          # 相邻航迹位置间隔 (m)

    @property
    def x_plane(self):
        """航迹上各发收位置的方位坐标 (m)，以场景中心为原点。"""
        return (np.arange(self.Nx) - self.Nx // 2) * self.dx

    @property
    def t_fast(self):
        """一个脉冲内部的快时间采样时刻 (s)。"""
        return np.linspace(0, self.T, self.Ns, endpoint=False)

    @property
    def L(self):
        """合成孔径长度 (m)。"""
        return self.Nx * self.dx

    def resolution(self):
        """返回 (距离向分辨率, 方位向分辨率)，单位 m。"""
        dr = self.c / (2 * self.B)
        da = self.lam * self.R0 / (2 * self.L)
        return dr, da


# ----------------------------------------------------------------------
# 1. 正演：由地面目标生成原始回波矩阵
# ----------------------------------------------------------------------
def forward(cfg, targets, jitter=None):
    """
    正演采集：模拟飞机沿航迹逐点发收，得到二维原始回波矩阵。

    参数
    ----
    cfg     : SALConfig
    targets : list of (az, dr, amp)
              az  目标方位坐标 (m)
              dr  目标距离偏移 (m, 相对场景中心 R0)
              amp 反射强度
    jitter  : None 或长度 Nx 的数组
              每个航迹位置的额外距离误差 (m)，用于模拟平台抖动/相位误差。
              注意：波长量级(~μm)的 jitter 就足以破坏成像 —— 这是运动补偿存在的理由。

    返回
    ----
    raw : (Nx, Ns) complex ndarray
          原始回波（去调频后的拍频信号）。直接看它是一片噪声，图藏在相位里。

    要点
    ----
    - 拍频只对"相对距离 dR = R - R0"编码，避免绝对斜距(~1000m)被 FFT 区间折叠。
    - 载频相位 exp(-j·4πR/λ) 里藏着方位向的相位历程，是合成孔径的信息来源。
    """
    Nx, Ns = cfg.Nx, cfg.Ns
    x_plane, t_fast = cfg.x_plane, cfg.t_fast
    R0, B, T, c, lam = cfg.R0, cfg.B, cfg.T, cfg.c, cfg.lam

    if jitter is None:
        jitter = np.zeros(Nx)

    raw = np.zeros((Nx, Ns), dtype=complex)
    for (az, dr, amp) in targets:
        # 各航迹位置到该目标的斜距 (Nx,)，叠加抖动误差
        R = np.sqrt((R0 + dr) ** 2 + (x_plane - az) ** 2) + jitter
        dR = R - R0
        beat    = np.exp(1j * 2 * np.pi * (B / T) * (2 * dR[:, None] / c) * t_fast[None, :])
        carrier = np.exp(-1j * 4 * np.pi * R[:, None] / lam)
        raw += amp * beat * carrier
    return raw


# ----------------------------------------------------------------------
# 2. 距离向压缩：每行 FFT，把拍频变成距离
# ----------------------------------------------------------------------
def range_compress(cfg, raw, window=True):
    """
    距离向压缩。对每一行(每个脉冲)做 FFT：不同拍频 -> 不同距离。

    返回
    ----
    RC        : (Nx, Ns) complex，距离压缩后的数据
    dR_of_bin : (Ns,) 每个频率 bin 对应的相对斜距 (m)
                由 f = (B/T)(2dR/c)  =>  dR = f·T·c/(2B) 标定。

    window : 是否加 Hamming 窗压距离向旁瓣（强烈建议开，否则强点旁瓣淹没弱点）。
    """
    Ns = cfg.Ns
    w = np.hamming(Ns) if window else np.ones(Ns)
    RC = np.fft.fft(raw * w[None, :], axis=1)
    dR_of_bin = np.fft.fftfreq(Ns, d=cfg.T / Ns) * cfg.T * cfg.c / (2 * cfg.B)
    return RC, dR_of_bin


# ----------------------------------------------------------------------
# 3. 方位向压缩：后向投影 (Back-Projection)
# ----------------------------------------------------------------------
def backprojection(cfg, RC, dR_of_bin, az_grid, r_grid, window=True):
    """
    方位向后向投影成像 —— 合成孔径真正发生的地方。

    对成像网格里每个像素 (az, dr)：
      1. 算出它到各航迹位置的斜距 R (即它专属的"相位签名")
      2. 在距离压缩剖面上插值取出对应复值
      3. 补偿载频相位后相干累加 —— 猜对位置能量同相叠加成亮点，
         猜错位置随机相位相互抵消成背景。

    参数
    ----
    az_grid, r_grid : 成像网格的方位/距离坐标数组
    window          : 方位向是否加 Hamming 窗

    返回
    ----
    img : (len(r_grid), len(az_grid)) 实数，像素亮度 = 该点反射强度
    """
    Nx, lam = cfg.Nx, cfg.lam
    x_plane = cfg.x_plane
    R0 = cfg.R0

    order = np.argsort(dR_of_bin)
    Rs = dR_of_bin[order]
    wa = np.hamming(Nx) if window else np.ones(Nx)

    img = np.zeros((len(r_grid), len(az_grid)))
    for ia, az in enumerate(az_grid):
        # 该方位列上所有距离像素到各航迹位置的斜距 (Nr, Nx)
        R = np.sqrt((R0 + r_grid[:, None]) ** 2 + (x_plane[None, :] - az) ** 2)
        dR = R - R0
        acc = np.zeros(len(r_grid), dtype=complex)
        for i in range(Nx):
            re = np.interp(dR[:, i], Rs, RC[i, order].real)
            im = np.interp(dR[:, i], Rs, RC[i, order].imag)
            acc += wa[i] * (re + 1j * im) * np.exp(1j * 4 * np.pi * R[:, i] / lam)
        img[:, ia] = np.abs(acc)
    return img


# ----------------------------------------------------------------------
# 便捷封装：一次跑完整条成像链
# ----------------------------------------------------------------------
def image_scene(cfg, targets, az_grid, r_grid, jitter=None):
    """正演 -> 距离压缩 -> 方位BP，返回 (img, raw, RC)。"""
    raw = forward(cfg, targets, jitter=jitter)
    RC, dR_of_bin = range_compress(cfg, raw)
    img = backprojection(cfg, RC, dR_of_bin, az_grid, r_grid)
    return img, raw, RC


# ----------------------------------------------------------------------
# 目标图案工具：把字母/点阵铺到地面当反射率分布
# ----------------------------------------------------------------------
def letters_SAL():
    """返回组成字母 'SAL' 的目标列表 [(az, dr, amp), ...]。"""
    scene = np.zeros((13, 26))
    def stroke(pts):
        for (r, col) in pts:
            scene[r, col] = 1.0
    # S
    stroke([(2,2),(2,3),(2,4),(2,5),(3,2),(4,2),(5,3),(5,4),(6,5),
            (7,6),(8,6),(9,2),(9,3),(9,4),(9,5),(6,4),(6,5)])
    # A
    stroke([(9,9),(8,9),(7,10),(6,10),(5,11),(4,11),(3,12),(2,13),(3,14),
            (4,15),(5,15),(6,16),(7,16),(8,17),(9,17),(6,12),(6,13),(6,14),(6,15)])
    # L
    stroke([(2,20),(3,20),(4,20),(5,20),(6,20),(7,20),(8,20),(9,20),(9,21),(9,22),(9,23)])
    rows, cols = np.where(scene > 0)
    az = (cols - 13) * 0.006
    dr = (6 - rows) * 0.04
    return [(a, d, 1.0) for a, d in zip(az, dr)]


# ----------------------------------------------------------------------
# 演示主程序
# ----------------------------------------------------------------------
def demo(save_path="sal_demo.png", with_jitter=False):
    """跑一遍完整成像并画三联图（原始/距离压缩/最终图像）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = SALConfig()
    targets = letters_SAL()
    dr_res, da_res = cfg.resolution()
    print(f"距离向分辨率 δr = {dr_res*1000:.2f} mm   方位向分辨率 δa = {da_res*1000:.2f} mm")
    print(f"合成孔径长度 L = {cfg.L:.2f} m,  目标点数 = {len(targets)}")

    jitter = None
    if with_jitter:
        rng = np.random.default_rng(0)
        jitter = rng.normal(0, 0.3e-6, cfg.Nx)   # 0.3 μm 抖动 -> 图像散焦
        print("已注入 0.3 μm 随机航迹抖动（演示相位误差如何破坏成像）")

    az_grid = np.linspace(-0.10, 0.10, 320)
    r_grid  = np.linspace(-0.30, 0.30, 320)
    img, raw, RC = image_scene(cfg, targets, az_grid, r_grid, jitter=jitter)
    img /= img.max()

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(np.abs(raw), aspect="auto", cmap="gray",
                 extent=[0, cfg.Ns, cfg.x_plane[-1], cfg.x_plane[0]])
    ax[0].set_title("(1) Raw echo (looks like noise)")
    ax[0].set_xlabel("range samples"); ax[0].set_ylabel("azimuth (m)")

    RCs = np.fft.fftshift(RC, axes=1)
    _, dRbin = range_compress(cfg, raw)
    Rb = np.fft.fftshift(dRbin)
    ax[1].imshow(np.abs(RCs), aspect="auto", cmap="gray",
                 extent=[Rb[0], Rb[-1], cfg.x_plane[-1], cfg.x_plane[0]])
    ax[1].set_xlim(-0.30, 0.30)
    ax[1].set_title("(2) Range compressed")
    ax[1].set_xlabel("range (m, rel)"); ax[1].set_ylabel("azimuth (m)")

    ax[2].imshow(img ** 0.5, aspect="auto", cmap="gray", origin="lower",
                 extent=[az_grid[0], az_grid[-1], r_grid[0], r_grid[-1]])
    title = "(3) Final SAL image" + ("  [with jitter → defocused]" if with_jitter else "")
    ax[2].set_title(title)
    ax[2].set_xlabel("azimuth = x (m)"); ax[2].set_ylabel("range = y (m)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"图已保存到 {save_path}")


if __name__ == "__main__":
    demo("sal_demo.png", with_jitter=False)
    demo("sal_demo_jitter.png", with_jitter=True)
