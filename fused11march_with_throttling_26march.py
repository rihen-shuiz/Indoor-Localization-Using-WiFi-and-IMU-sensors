"""
PDR + WiFi Fusion — old dataset (20260224, 125Hz)
==================================================
PDR pipeline  : imported from pdr_only_IMU.py
WiFi localizer: WifiFileLocalizer2 (weighted KNN)
Fusion        : Extended Kalman Filter — state = [x, y, yaw_offset, step_scale]
"""

import sys, os, math, tempfile, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from wifi_localizer import WifiFileLocalizer2, EstimateMeta

# import PDR functions from pdr_only_IMU.py
from pdr_only_IMU import (
    lowpass, bandpass, wrap_pi,
    detect_vertical_axis,
    detect_steps as pdr_detect_steps,
    estimate_beta_zupt as pdr_estimate_beta,
    step_lengths_weinberg,
    compute_heading_madgwick as compute_heading_gyro,
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
_PROJECT = _HERE
BASE     = os.path.join(_PROJECT, 'floors', '4-floor')
RADIOMAP = os.path.join(_HERE, 'new_radiomap', 'radiomap.csv')

# ─────────────────────────────────────────────────────────────
# PDR CONFIG
# ─────────────────────────────────────────────────────────────
LP_CUTOFF = 5.0
# Initial heading is set to 0 for all trajectories.
# The EKF yaw_offset state learns the correct heading automatically
# from consecutive WiFi fixes — no manual configuration needed.

# ─────────────────────────────────────────────────────────────
# EKF CONFIG
# ─────────────────────────────────────────────────────────────
PDR_POS_NOISE_PER_STEP = 0.8
PDR_YAW_NOISE_PER_STEP = 0.04
PDR_SCALE_NOISE        = 0.002

WIFI_BASE_NOISE  = 3.0
WIFI_MIN_CONF    = 0.15
WIFI_MIN_OVERLAP = 20
WIFI_GATE_CHI2   = 30.0

# ─────────────────────────────────────────────────────────────
# THROTTLE EXPERIMENT CONFIG
# ─────────────────────────────────────────────────────────────
INTERVALS = [0, 10, 20, 30]
COLORS    = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

TRAJS = {

    'traj3': {
        'imu':      'traj3/imu_20260224_133821.csv',
        'gt':       'traj3/traj_20260224_133821.csv',
        'wifi_dir': 'traj3',
        'wifi_file':'wifi_20260224_133821',
    },
    'traj4': {
        'imu':      'traj4/imu_20260224_140955.csv',
        'gt':       'traj4/traj_20260224_140955.csv',
        'wifi_dir': 'traj4',
        'wifi_file':'wifi_20260224_140955',
    },
    'traj5': {
        'imu':      'traj5/imu_20260224_142341.csv',
        'gt':       'traj5/traj_20260224_142341.csv',
        'wifi_dir': 'traj5',
        'wifi_file':'wifi_20260224_142341',
    },
    'traj6': {
        'imu':      'traj6/imu_20260224_143542.csv',
        'gt':       'traj6/traj_20260224_143542.csv',
        'wifi_dir': 'traj6',
        'wifi_file':'wifi_20260224_143542',
    },
    'traj7': {
        'imu':      'traj7/imu_20260326_133856.csv',
        'gt':       'traj7/traj_20260326_133856.csv',
        'wifi_dir': 'traj7',
        'wifi_file':'wifi_20260326_133856',
        'base':     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'floors', '4-floor'),
    },
}

# ─────────────────────────────────────────────────────────────
# WIFI THROTTLING
# ─────────────────────────────────────────────────────────────
def throttle_scans(scan_ts_abs, t0_abs, interval_s):
    kept   = []
    last_t = -999.
    for ts_abs in scan_ts_abs:
        t_rel = (ts_abs - t0_abs) / 1000.0
        if t_rel - last_t >= interval_s:
            kept.append(ts_abs)
            last_t = t_rel
    return np.array(kept, dtype=np.int64)

# ─────────────────────────────────────────────────────────────
# FUSION — EKF
# ─────────────────────────────────────────────────────────────
def run_fusion(name, paths, localizer):
    print('=' * 50)
    print(name)
    print('=' * 50)

    # ── load IMU ──
    base = paths.get('base', BASE)
    imu = pd.read_csv(os.path.join(base, paths['imu']))
    gt  = pd.read_csv(os.path.join(base, paths['gt']))

    ts_raw = imu['ts'].values.astype(float)
    if np.median(np.diff(ts_raw)) > 1.0:
        ts_raw = ts_raw / 1000.0
    ts_raw -= ts_raw[0]
    fs = len(ts_raw) / (ts_raw[-1] - ts_raw[0])
    ts = ts_raw
    print(f"  Auto-detected FS: {fs:.1f} Hz")

    acc  = imu[['ax','ay','az']].values.astype(float)
    gyro = imu[['gx','gy','gz']].values.astype(float)

    # initial heading = 0 for all trajectories
    # EKF yaw_offset corrects this automatically from WiFi fixes
    initial_yaw = 0.0
    print(f"  Initial heading: 0° (EKF will correct via WiFi)")

    acc_lp        = lowpass(acc, LP_CUTOFF, fs)
    vertical_axis = detect_vertical_axis(acc)

    # detect_steps from pdr_only_IMU takes (acc_lp, fs)
    peaks, acc_bp = pdr_detect_steps(acc_lp, fs)
    acc_mag       = np.linalg.norm(acc_lp, axis=1)

    # estimate_beta_zupt from pdr_only_IMU takes (acc_lp, peaks, fs)
    beta = pdr_estimate_beta(acc_lp, peaks, fs)

    # step_lengths_weinberg takes (acc_lp, peaks, beta, fs)
    lens = step_lengths_weinberg(acc_lp, peaks, beta, fs)

    # compute_heading_gyro takes (ts, gyro, fs, vertical_axis, initial_yaw)
    yaw_rel = compute_heading_gyro(ts, acc_lp, gyro, fs, vertical_axis, initial_yaw)
    # initial_yaw already passed into Madgwick — no need to add offset separately

    print(f"  Steps: {len(peaks)},  beta: {beta:.4f}")

    # ── WiFi data ──
    wifi_csv    = pd.read_csv(os.path.join(base, paths['wifi_dir'],
                                           paths['wifi_file'] + '.csv'))
    scan_ts_abs = np.unique(wifi_csv['ts'].values.astype(np.int64))
    t0_abs      = int(imu['ts'].values[0])
    scan_ts_rel = (scan_ts_abs - t0_abs) / 1000.0

    gt_x = gt['x'].values.astype(float)
    gt_y = gt['y'].values.astype(float)

    # ── EKF initialisation ──
    ekf_x = np.array([0., 0., 0., 1.0])
    ekf_P = np.diag([100., 100., (np.pi)**2, 0.5**2])

    initialized    = False
    traj_pts       = []
    wifi_fixes_raw = []
    wifi_accepted  = 0
    wifi_rejected  = 0

    localizer.start_run(paths['wifi_file'])
    step_idx = 0
    scan_idx = 0

    for i in range(len(ts)):
        t = ts[i]

        # PDR prediction step
        while step_idx < len(peaks) and peaks[step_idx] <= i:
            k = peaks[step_idx]
            if initialized:
                yaw_total = wrap_pi(yaw_rel[k] + ekf_x[2])
                L = ekf_x[3] * lens[step_idx]
                ekf_x[0] += L * math.cos(yaw_total)
                ekf_x[1] += L * math.sin(yaw_total)
                F = np.eye(4)
                F[0, 2] = -L * math.sin(yaw_total)
                F[0, 3] =  lens[step_idx] * math.cos(yaw_total)
                F[1, 2] =  L * math.cos(yaw_total)
                F[1, 3] =  lens[step_idx] * math.sin(yaw_total)
                Q = np.diag([PDR_POS_NOISE_PER_STEP**2,
                             PDR_POS_NOISE_PER_STEP**2,
                             PDR_YAW_NOISE_PER_STEP**2,
                             PDR_SCALE_NOISE**2])
                ekf_P = F @ ekf_P @ F.T + Q
                traj_pts.append((ts[k], ekf_x[0], ekf_x[1]))
            step_idx += 1

        # WiFi measurement update
        while scan_idx < len(scan_ts_abs) and scan_ts_rel[scan_idx] <= t:
            ts_ms_abs = int(scan_ts_abs[scan_idx])
            scan_idx += 1

            (xw, yw), meta = localizer.estimate(
                file_name=paths['wifi_file'], t_ms=ts_ms_abs)

            if not meta.accepted:                       continue
            if meta.confidence < WIFI_MIN_CONF:         continue
            if meta.overlap_best < WIFI_MIN_OVERLAP:    continue

            if not initialized:
                ekf_x[0] = xw
                ekf_x[1] = yw
                ekf_P[0, 0] = WIFI_BASE_NOISE**2
                ekf_P[1, 1] = WIFI_BASE_NOISE**2
                initialized = True
                traj_pts.append((t, ekf_x[0], ekf_x[1]))
                print(f"  Initialized: ({xw:.1f}, {yw:.1f})  t={t:.1f}s")
                wifi_fixes_raw.append(t)
                wifi_accepted += 1
                continue

            quality = meta.confidence * min(meta.overlap_best / 30.0, 1.0)
            quality = max(quality, 0.1)
            R_std   = min(WIFI_BASE_NOISE / (quality ** 1.5), 25.0)
            R       = np.diag([R_std**2, R_std**2])

            H       = np.zeros((2, 4))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            innov   = np.array([xw, yw]) - ekf_x[:2]
            S       = H @ ekf_P @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            maha2 = float(innov @ S_inv @ innov)
            if maha2 > WIFI_GATE_CHI2:
                wifi_rejected += 1
                continue

            K        = ekf_P @ H.T @ S_inv
            ekf_x    = ekf_x + K @ innov
            ekf_x[2] = wrap_pi(ekf_x[2])
            ekf_x[3] = float(np.clip(ekf_x[3], 0.5, 2.0))
            ekf_P    = (np.eye(4) - K @ H) @ ekf_P
            # store time of this fix — will be projected onto GT line for visualization
            wifi_fixes_raw.append(t)
            wifi_accepted += 1

    # flush remaining steps
    while step_idx < len(peaks):
        k = peaks[step_idx]
        if initialized:
            yaw_total = wrap_pi(yaw_rel[k] + ekf_x[2])
            L = ekf_x[3] * lens[step_idx]
            ekf_x[0] += L * math.cos(yaw_total)
            ekf_x[1] += L * math.sin(yaw_total)
            traj_pts.append((ts[k], ekf_x[0], ekf_x[1]))
        step_idx += 1

    print(f"  WiFi accepted: {wifi_accepted},  rejected: {wifi_rejected}")
    print(f"  Final yaw offset: {math.degrees(ekf_x[2]):.1f}°  "
          f"step scale: {ekf_x[3]:.3f}")

    if not traj_pts:
        print("  [WARNING] No trajectory produced")
        return None, None, None, None, None

    traj = np.array([(p[1], p[2]) for p in traj_pts])

    # project WiFi fix times onto GT line for visualization
    # GT has one point per IMU sample — normalize time to [0,1]
    if wifi_fixes_raw:
        traj_times  = np.array([p[0] for p in traj_pts])
        total_time  = traj_times[-1] - traj_times[0] if len(traj_times) > 1 else 1.0
        fix_fracs   = [(ft - traj_times[0]) / total_time for ft in wifi_fixes_raw]
        fix_fracs   = np.clip(fix_fracs, 0, 1)
        gt_norm     = np.linspace(0, 1, len(gt_x))
        fix_gt_x    = np.interp(fix_fracs, gt_norm, gt_x)
        fix_gt_y    = np.interp(fix_fracs, gt_norm, gt_y)
        wifi_fixes  = np.column_stack([fix_gt_x, fix_gt_y])
    else:
        wifi_fixes  = np.empty((0, 2))

    # error metrics — no dx/dy shift, honest result
    n      = len(gt_x)
    t_norm = np.linspace(0, 1, len(traj))
    px     = np.interp(np.linspace(0, 1, n), t_norm, traj[:, 0])
    py     = np.interp(np.linspace(0, 1, n), t_norm, traj[:, 1])
    errors = np.sqrt((px - gt_x)**2 + (py - gt_y)**2)
    print(f"  Error — mean={errors.mean():.1f}m  "
          f"median={np.median(errors):.1f}m  max={errors.max():.1f}m")

    return traj, wifi_fixes, gt_x, gt_y, errors


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    for name, paths in TRAJS.items():
        base        = paths.get('base', BASE)
        wifi_csv    = pd.read_csv(os.path.join(base, paths['wifi_dir'],
                                               paths['wifi_file'] + '.csv'))
        imu_tmp     = pd.read_csv(os.path.join(base, paths['imu']))
        t0_abs      = int(imu_tmp['ts'].values[0])
        all_scan_ts = np.unique(wifi_csv['ts'].values.astype(np.int64))

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes      = axes.flatten()
        row_means = []

        for ax_idx, (interval, color) in enumerate(zip(INTERVALS, COLORS)):

            throttled    = throttle_scans(all_scan_ts, t0_abs, interval)
            throttled_df = wifi_csv[wifi_csv['ts'].isin(throttled)]

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                             delete=False, dir=None,
                                             encoding='utf-8', newline='') as f:
                tmp_path = f.name
                throttled_df.to_csv(f, index=False)

            tmp_name  = os.path.splitext(os.path.basename(tmp_path))[0]
            wifi_dir  = os.path.join(base, paths['wifi_dir'])
            shutil.copy(tmp_path, os.path.join(wifi_dir, tmp_name + '.csv'))

            orig_file          = paths['wifi_file']
            paths['wifi_file'] = tmp_name

            loc = WifiFileLocalizer2(
                radiomap_csv=RADIOMAP, k=3, min_overlap=20,
                dataset_dir=wifi_dir)

            traj, wifi_fixes, gt_x, gt_y, errors = run_fusion(name, paths, loc)

            paths['wifi_file'] = orig_file
            os.remove(os.path.join(wifi_dir, tmp_name + '.csv'))
            os.remove(tmp_path)

            if traj is None:
                axes[ax_idx].set_title(f'{interval}s — no result')
                continue

            row_means.append(errors.mean())
            ax = axes[ax_idx]
            ox, oy = gt_x[0], gt_y[0]
            ax.plot(gt_x-ox, gt_y-oy, 'b-', lw=2.5, label='Ground Truth', zorder=2)
            ax.plot(traj[:,0]-ox, traj[:,1]-oy, '-', color=color, lw=1.8,
                    label='PDR+WiFi Fusion', zorder=3)
            if len(wifi_fixes) > 0:
                ax.scatter(wifi_fixes[:,0]-ox, wifi_fixes[:,1]-oy,
                           c='orange', s=20, zorder=5, alpha=0.6,
                           label=f'WiFi fixes ({len(throttled)} scans)')
            ax.scatter(traj[0,0]-ox,  traj[0,1]-oy,  color='green', s=120,
                       zorder=6, label='Start')
            ax.scatter(traj[-1,0]-ox, traj[-1,1]-oy, color=color, s=120,
                       marker='X', zorder=6, label='End (fusion)')
            ax.scatter(gt_x[-1]-ox,   gt_y[-1]-oy,   color='blue', s=120,
                       marker='X', zorder=6, label='End (GT)')

            title = f'Wi-Fi scan every ~5s ({len(throttled)} scans)' \
                    if interval == 0 else \
                    f'Wi-Fi scan every {interval}s ({len(throttled)} scans)'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.axis('equal'); ax.grid(True, alpha=0.35)
            ax.legend(fontsize=10, loc='upper left')

        fig.suptitle(
            f'{name}  —  Effect of WiFi Scan Interval on Fusion Accuracy\n'
            f'EKF · ZUPT beta · Madgwick heading · Mahalanobis gate',
            fontsize=12, fontweight='bold')
        plt.tight_layout()
        out = os.path.join(_HERE, 'new', f'{name}_throttle.png')
        os.makedirs(os.path.join(_HERE, 'new'), exist_ok=True)
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")
        vals = '  '.join(f'{m:>8.1f}m' for m in row_means)
        print(f"{name:10} {vals}")