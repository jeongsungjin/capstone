import numpy as np

try:
    # SciPy가 없을 경우를 대비해 폴백을 제공한다.
    from scipy.interpolate import CubicSpline as CubicSpline1D  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    CubicSpline1D = None  # type: ignore
    _HAVE_SCIPY = False

class CubicSpline2D:
    def __init__(self, x, y, interval=0.2):
        self.interval = interval

        # 호길이 누적
        dx_input = np.diff(x)
        dy_input = np.diff(y)
        self.rs = np.concatenate([[0.0], np.cumsum(np.hypot(dx_input, dy_input))])

        # 샘플 파라미터 s
        if self.rs[-1] <= 0.0:
            ss = np.array([0.0])
        else:
            ss = np.arange(0.0, self.rs[-1], interval)

        if _HAVE_SCIPY and CubicSpline1D is not None:
            sx = CubicSpline1D(self.rs, x)
            sy = CubicSpline1D(self.rs, y)

            self.rx = sx(ss)
            self.ry = sy(ss)

            dx = sx(ss, 1)
            dy = sy(ss, 1)
            self.ryaw = np.arctan2(dy, dx)

            ddx = sx(ss, 2)
            ddy = sy(ss, 2)
        else:
            # SciPy가 없을 때: 선형보간 + 유한차분
            self.rx = np.interp(ss, self.rs, x)
            self.ry = np.interp(ss, self.rs, y)

            if len(ss) >= 2:
                dx = np.gradient(self.rx, interval, edge_order=2)
                dy = np.gradient(self.ry, interval, edge_order=2)
            else:
                dx = np.array([0.0])
                dy = np.array([0.0])
            self.ryaw = np.arctan2(dy, dx)

            if len(ss) >= 3:
                ddx = np.gradient(dx, interval, edge_order=2)
                ddy = np.gradient(dy, interval, edge_order=2)
            else:
                ddx = np.zeros_like(dx)
                ddy = np.zeros_like(dy)

        # 곡률
        denom = (dx ** 2 + dy ** 2) ** 1.5
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = np.abs(dx * ddy - dy * ddx) / denom
        self.rkappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)

def catesian_to_frenet(x: float, y: float, csp: CubicSpline2D):
    dist = np.hypot(csp.rx - x, csp.ry - y)
    
    idx = np.argmin(dist)

    dx = x - csp.rx[idx]
    dy = y - csp.ry[idx]

    cos = np.cos(csp.ryaw[idx])
    sin = np.sin(csp.ryaw[idx])

    s = idx * csp.interval
    d = np.copysign(dist[idx], cos * dy - sin * dx)

    return s, d