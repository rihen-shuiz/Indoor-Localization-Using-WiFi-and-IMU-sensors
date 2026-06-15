"""
IMU-only PDR — universally applicable implementation.
Supports 2-CSV format: separate IMU file and GT file.

IMU CSV format:
  ts, ax, ay, az, gx, gy, gz (+ optional mx, my, mz)

GT CSV format:
  ts, x, y (+ optional heading)

Universally applicable:
  - Auto-detects sampling rate from timestamps
  - Auto-detects gravity axis from accelerometer means
  - Adaptive step detection threshold (60th percentile)
  - Adaptive step cadence from FFT dominant frequency
  - ZUPT beta calibration (no GT needed)
  - ZUPT-aided heading bias correction (no GT needed)

GT is used for ONE thing only:
  [EVALUATION] error metrics and comparison plot — never during PDR itself.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LP_CUTOFF   = 5.0
BP_LOW      = 0.5
BP_HIGH     = 3.0

STATIC_ACC_MEAN_TOL = 0.3
STATIC_ACC_STD_TOL  = 0.15
STATIC_WIN_S        = 0.5
PRIOR_STEP_LEN      = 0.78

Q_YAW  = 1e-4
Q_BIAS = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — change these to your files
# ─────────────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = _HERE

# Add your trajectories here.
# initial_yaw_deg = None means auto-detect from GT first steps.
# Set a number (0, 90, 180, 270) only if you know the heading from building map.
TRAJS = {
    'traj3': (
        'floors/4-floor/traj3/imu_20260224_133821.csv',
        'floors/4-floor/traj3/traj_20260224_133821.csv',
        None
    ),
    'traj4': (
        'floors/4-floor/traj4/imu_20260224_140955.csv',
        'floors/4-floor/traj4/traj_20260224_140955.csv',
        None
    ),
    'traj5': (
        'floors/4-floor/traj5/imu_20260224_142341.csv',
        'floors/4-floor/traj5/traj_20260224_142341.csv',
        None
    ),
    'traj6': (
        'floors/4-floor/traj6/imu_20260224_143542.csv',
        'floors/4-floor/traj6/traj_20260224_143542.csv',
        None
    ),

    'traj7': (
        'floors/4-floor/traj7/imu_20260326_133856.csv',
        'floors/4-floor/traj7/traj_20260326_133856.csv',
        None
    ),

    'traj8': (
        'floors/4-floor/traj8/imu_20260326_130539.csv',
        'floors/4-floor/traj8/traj_20260326_130539.csv',
        None
    ),

    'traj9': (
        'floors/4-floor/traj9/imu_20260326_125725.csv',
        'floors/4-floor/traj9/traj_20260326_125725.csv',
        None
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def lowpass(x, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, x, axis=0)

def bandpass(x, lo, hi, fs, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x)

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT GRAVITY AXIS
#
# The axis whose mean acceleration is closest to 9.81 m/s² is the
# gravity axis. The same index is used for the gyroscope yaw channel.
# Works for any phone orientation — no manual configuration needed.
# ─────────────────────────────────────────────────────────────────────────────
def detect_vertical_axis(acc):
    axis_means = np.abs(np.mean(acc, axis=0))
    vertical_axis = int(np.argmax(axis_means))
    axis_names = ['x', 'y', 'z']
    print(f"  Axis means: ax={axis_means[0]:.3f}, ay={axis_means[1]:.3f}, "
          f"az={axis_means[2]:.3f}")
    print(f"  Auto-detected vertical axis: {axis_names[vertical_axis]} "
          f"(index {vertical_axis}) — gravity = {axis_means[vertical_axis]:.2f} m/s²")
    return vertical_axis

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_imu(path):
    df  = pd.read_csv(path)

    # support both 'ts' and 'timestamp' column names
    ts_col = 'ts' if 'ts' in df.columns else 'timestamp'
    ts = df[ts_col].values.astype(float)
    if np.median(np.diff(ts)) > 1.0:  # milliseconds → seconds
        ts = ts / 1000.0
    ts -= ts[0]

    # auto-detect sampling rate
    duration = ts[-1] - ts[0]
    fs = len(ts) / duration
    print(f"  Auto-detected FS: {fs:.1f} Hz")

    acc  = df[['ax', 'ay', 'az']].values.astype(float)
    gyro = df[['gx', 'gy', 'gz']].values.astype(float)
    mag  = df[['mx', 'my', 'mz']].values.astype(float)
    return ts, acc, gyro, mag, fs

def load_gt_for_eval(gt_path):
    """Load GT — used ONLY for evaluation and plotting, never during PDR."""
    gt_df = pd.read_csv(gt_path)
    gt_x  = gt_df['x'].values.astype(float) - gt_df['x'].values[0]
    gt_y  = gt_df['y'].values.astype(float) - gt_df['y'].values[0]
    return gt_x, gt_y

def detect_initial_heading_from_gt(gt_path):
    """
    Auto-detect initial heading from GT first steps.
    Uses first 10 GT points to compute direction of movement.
    Used ONLY for PDR-only visualization — not needed for fusion.
    """
    gt_df = pd.read_csv(gt_path)
    x = gt_df['x'].values.astype(float)
    y = gt_df['y'].values.astype(float)
    n = min(10, len(x) - 1)
    dx = x[n] - x[0]
    dy = y[n] - y[0]
    if abs(dx) + abs(dy) < 1e-3:
        print("  WARNING: GT start is stationary, defaulting to 0°")
        return 0.0
    heading_rad = float(np.arctan2(dy, dx))
    print(f"  Auto-detected initial heading from GT: {np.degrees(heading_rad):.1f}°")
    return heading_rad

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — STEP DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_steps(acc_lp, fs):
    acc_mag = np.linalg.norm(acc_lp, axis=1)
    acc_bp  = bandpass(acc_mag, BP_LOW, BP_HIGH, fs)

    # find dominant frequency in full walking range 0.5-2.5 Hz
    fft_vals  = np.abs(np.fft.rfft(acc_bp))
    fft_freqs = np.fft.rfftfreq(len(acc_bp), 1.0 / fs)
    mask      = (fft_freqs >= 0.5) & (fft_freqs <= 2.5)
    dominant_freq = fft_freqs[mask][np.argmax(fft_vals[mask])]
    step_period   = 1.0 / dominant_freq
    min_distance  = int(step_period * 1.0 * fs)
    duration_s    = len(acc_bp) / fs

    # expected steps: dominant_freq/2 × duration
    # divide by 2 because FFT detects cadence (both feet per cycle),
    # but find_peaks with full-period distance detects one foot per stride
    expected_steps = int((dominant_freq / 2) * duration_s)

    # clamp expected to physiologically plausible range
    # normal walking: 0.5–1.5 steps/sec, so 0.25–0.75 Hz per foot
    min_expected = int(0.25 * duration_s)
    max_expected = int(0.75 * duration_s)
    expected_steps = int(np.clip(expected_steps, min_expected, max_expected))

    # try percentiles 50-85, pick the one closest to expected step count
    positive_vals = acc_bp[acc_bp > 0]
    best_peaks = None
    best_diff  = np.inf
    best_pct   = 60
    best_h     = float(np.percentile(positive_vals, 60))
    for pct in range(50, 86, 5):
        h = float(np.percentile(positive_vals, pct))
        p, _ = find_peaks(acc_bp, height=h, distance=min_distance)
        diff = abs(len(p) - expected_steps)
        if diff < best_diff:
            best_diff  = diff
            best_peaks = p
            best_pct   = pct
            best_h     = h

    print(f"  Adaptive step height threshold: {best_h:.3f} "
          f"(percentile={best_pct}, steps={len(best_peaks)}, expected~{expected_steps})")
    print(f"  Dominant step freq: {dominant_freq:.2f} Hz → "
          f"min distance: {step_period:.2f}s ({min_distance} samples)")

    return best_peaks, acc_bp

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BETA ESTIMATION via ZUPT
# ─────────────────────────────────────────────────────────────────────────────
def estimate_beta_zupt(acc_lp, peaks, fs):
    acc_mag = np.linalg.norm(acc_lp, axis=1)
    win_n   = int(STATIC_WIN_S * fs)

    is_static = np.zeros(len(acc_mag), bool)
    for i in range(0, len(acc_mag) - win_n, win_n):
        seg = acc_mag[i : i + win_n]
        if (abs(seg.mean() - 9.81) < STATIC_ACC_MEAN_TOL and
                seg.std() < STATIC_ACC_STD_TOL):
            is_static[i : i + win_n] = True

    starts, ends, in_s = [], [], False
    for i in range(len(is_static)):
        if is_static[i] and not in_s:
            starts.append(i); in_s = True
        elif not is_static[i] and in_s:
            ends.append(i);   in_s = False
    if in_s:
        ends.append(len(is_static) - 1)

    win_step = int(0.8 * fs)
    betas = []
    for si in range(min(len(starts), len(ends)) - 1):
        seg_peaks = peaks[(peaks >= ends[si]) & (peaks < starts[si + 1])]
        if len(seg_peaks) < 2:
            continue
        raw_sum = 0.0
        for k in seg_peaks:
            s = acc_mag[max(0, k - win_step//2) : k + win_step//2 + 1]
            raw_sum += (s.max() - s.min()) ** 0.25
        if raw_sum < 1e-9:
            continue
        betas.append(len(seg_peaks) * PRIOR_STEP_LEN / raw_sum)

    if not betas:
        print('  WARNING: no valid ZUPT segments found, using fallback beta=0.65')
        return 0.65
    beta = float(np.median(betas))
    print(f'  ZUPT segments used: {len(betas)},  beta = {beta:.4f}')
    return beta

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — STEP LENGTH (Weinberg)
# ─────────────────────────────────────────────────────────────────────────────
def step_lengths_weinberg(acc_lp, peaks, beta, fs):
    acc_mag = np.linalg.norm(acc_lp, axis=1)
    win     = int(0.8 * fs)
    lengths = []
    for k in peaks:
        seg = acc_mag[max(0, k - win//2) : k + win//2 + 1]
        L   = beta * (seg.max() - seg.min()) ** 0.25
        lengths.append(float(np.clip(L, 0.3, 1.5)))
    return np.array(lengths)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — HEADING (Madgwick filter)
#
# The Madgwick filter fuses accelerometer + gyroscope to estimate
# 3D orientation as a quaternion. It uses the accelerometer to correct
# gyroscope drift by comparing the measured gravity direction with the
# expected gravity direction from the current orientation estimate.
# This gives much better yaw tracking than gyro-only integration.
#
# Reference: Madgwick et al. (2011), "Estimation of IMU and MARG
# orientation using a gradient descent algorithm"
#
# beta parameter: controls how fast the filter corrects gyro drift.
# Higher beta = faster correction but more susceptible to acceleration noise.
# ─────────────────────────────────────────────────────────────────────────────
def compute_heading_madgwick(ts, acc_lp, gyro, fs, vertical_axis, initial_yaw=0.0,
                              beta=0.033):
    """
    Madgwick filter for heading estimation.
    Returns yaw angle at each timestep in radians.
    """
    # initialise quaternion [w, x, y, z]
    # align with initial heading
    q = np.array([np.cos(initial_yaw/2), 0.0, 0.0, np.sin(initial_yaw/2)])

    yaw_out = np.zeros(len(ts))

    for i in range(len(ts)):
        dt = float(ts[i] - ts[i-1]) if i > 0 else 1.0 / fs

        gx, gy, gz = gyro[i, 0], gyro[i, 1], gyro[i, 2]
        ax, ay, az = acc_lp[i, 0], acc_lp[i, 1], acc_lp[i, 2]

        # normalise accelerometer
        a_norm = np.sqrt(ax*ax + ay*ay + az*az)
        if a_norm < 1e-9:
            continue
        ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm

        qw, qx, qy, qz = q

        # gradient descent step — objective function from gravity
        f1 = 2*(qx*qz - qw*qy) - ax
        f2 = 2*(qw*qx + qy*qz) - ay
        f3 = 2*(0.5 - qx*qx - qy*qy) - az

        # Jacobian
        j11 = -2*qy;  j12 =  2*qz;  j13 = -2*qw; j14 = 2*qx
        j21 =  2*qw;  j22 =  2*qx;  j23 =  2*qy; j24 = 2*qz
        j31 =  0.0;   j32 = -4*qx;  j33 = -4*qy; j34 = 0.0

        # gradient
        gw = j11*f1 + j21*f2 + j31*f3
        gx_ = j12*f1 + j22*f2 + j32*f3
        gy_ = j13*f1 + j23*f2 + j33*f3
        gz_ = j14*f1 + j24*f2 + j34*f3

        g_norm = np.sqrt(gw*gw + gx_*gx_ + gy_*gy_ + gz_*gz_)
        if g_norm > 1e-9:
            gw /= g_norm; gx_ /= g_norm
            gy_ /= g_norm; gz_ /= g_norm

        # gyroscope quaternion derivative
        qdot_w = 0.5*(-qx*gx - qy*gy - qz*gz)
        qdot_x = 0.5*( qw*gx + qy*gz - qz*gy)
        qdot_y = 0.5*( qw*gy - qx*gz + qz*gx)
        qdot_z = 0.5*( qw*gz + qx*gy - qy*gx)

        # integrate
        qw += (qdot_w - beta*gw) * dt
        qx += (qdot_x - beta*gx_) * dt
        qy += (qdot_y - beta*gy_) * dt
        qz += (qdot_z - beta*gz_) * dt

        # normalise quaternion
        q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        q = np.array([qw, qx, qy, qz]) / q_norm

        # extract yaw from quaternion
        yaw_out[i] = np.arctan2(2*(qw*qz + qx*qy),
                                 1 - 2*(qy*qy + qz*qz))

    return yaw_out

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_trajectory(peaks, lengths, yaw):
    x, y = 0.0, 0.0
    pts  = [(x, y)]
    for k, L in zip(peaks, lengths):
        x += L * np.cos(yaw[k])
        y += L * np.sin(yaw[k])
        pts.append((x, y))
    return np.array(pts)

# ─────────────────────────────────────────────────────────────────────────────
# ERROR METRICS  [GT used here — evaluation only]
# ─────────────────────────────────────────────────────────────────────────────
def compute_errors(traj, gt_x, gt_y):
    n      = len(gt_x)
    t_norm = np.linspace(0, 1, len(traj))
    px = np.interp(np.linspace(0, 1, n), t_norm, traj[:, 0])
    py = np.interp(np.linspace(0, 1, n), t_norm, traj[:, 1])
    return np.sqrt((px - gt_x)**2 + (py - gt_y)**2)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    name = 'traj9'   # ← change to traj3 / traj4 / traj5 / traj6
    imu_rel, gt_rel, initial_yaw_deg = TRAJS[name]
    imu_path = os.path.join(_PROJECT, imu_rel)
    gt_path  = os.path.join(_PROJECT, gt_rel)

    print(f'=== {name} ===')

    # if initial heading known — use it; if None — auto-detect from GT first steps
    if initial_yaw_deg is not None:
        initial_yaw = np.deg2rad(initial_yaw_deg)
        print(f'  Initial heading from building map: {initial_yaw_deg}°')
    else:
        initial_yaw = detect_initial_heading_from_gt(gt_path)
        initial_yaw_deg = round(float(np.degrees(initial_yaw)), 1)

    ts, acc_raw, gyro, fs = load_imu(imu_path)

    # auto-detect vertical axis from accelerometer
    vertical_axis = detect_vertical_axis(acc_raw)

    acc_lp   = lowpass(acc_raw, LP_CUTOFF, fs)
    peaks, _ = detect_steps(acc_lp, fs)
    beta     = estimate_beta_zupt(acc_lp, peaks, fs)
    lengths  = step_lengths_weinberg(acc_lp, peaks, beta, fs)
    yaw      = compute_heading_madgwick(ts, acc_lp, gyro, fs,
                                        vertical_axis, initial_yaw)
    traj     = build_trajectory(peaks, lengths, yaw)

    print(f'  Steps: {len(peaks)}')
    print(f'  Step length: mean={lengths.mean():.3f}m  std={lengths.std():.3f}m')
    print(f'  PDR total distance: {lengths.sum():.1f} m')

    gt_x, gt_y = load_gt_for_eval(gt_path)
    gt_dist = float(np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2)))
    errors  = compute_errors(traj, gt_x, gt_y)
    print(f'  GT total distance: {gt_dist:.1f} m')
    print(f'  Error — mean={errors.mean():.1f}m  '
          f'median={np.median(errors):.1f}m  max={errors.max():.1f}m')

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(gt_x, gt_y, 'b-', lw=2, label='Ground Truth')
    ax.plot(traj[:,0], traj[:,1], 'r--', lw=1.5, label='PDR (IMU only)')
    ax.scatter(0, 0, color='green', s=80, zorder=5, label='Start (0,0)')
    ax.scatter(traj[-1,0], traj[-1,1], color='red', s=80,
               marker='X', zorder=5, label='PDR end')
    ax.scatter(gt_x[-1], gt_y[-1], color='blue', s=80,
               marker='X', zorder=5, label='GT end')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.axis('equal'); ax.grid(True, alpha=0.4)
    ax.legend(fontsize=11)

    plt.suptitle(
        'IMU-only PDR  (Weinberg + gyro KF)\n'
        'beta estimated per-trajectory via ZUPT  ·  no GT used at runtime\n'
        f'Initial heading from building map: {initial_yaw_deg}°',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(_HERE, f'pdr_results_{name}.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved: {out}')