import numpy as np
from scipy.interpolate import CubicSpline as CubicSpline1D

class CubicSpline2D:
    def __init__(self, x, y, interval=0.2):
        self.interval = interval
        
        dx = np.diff(x)
        dy = np.diff(y)

        self.rs = np.concatenate([[0], np.cumsum(np.hypot(dx, dy))])
        sx = CubicSpline1D(self.rs, x)
        sy = CubicSpline1D(self.rs, y)
        
        ss = np.arange(0, self.rs[-1], interval)
        self.rx = sx(ss)
        self.ry = sy(ss)
        
        dx = sx(ss, 1)
        dy = sy(ss, 1)
        self.ryaw = np.arctan2(dy, dx)
        
        ddx = sx(ss, 2)
        ddy = sy(ss, 2)
        
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