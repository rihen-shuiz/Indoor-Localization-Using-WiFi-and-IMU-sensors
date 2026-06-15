# Indoor Localization Using IMU Sensors and Wi-Fi Fusion

**MSc Thesis**  
School of Engineering and Digital Sciences 
Nazarbayev University, 2026

---

## Overview

This repository contains the source code and collected sensor data for the MSc thesis:

> *"Indoor Localization Using IMU Sensors and Wi-Fi Fusion"*

The system fuses IMU-based Pedestrian Dead Reckoning (PDR) with Wi-Fi fingerprinting via an Extended Kalman Filter (EKF) to produce continuous, drift-corrected indoor position estimates on an unmodified Android smartphone — without requiring any additional hardware or infrastructure changes.

---

## Repository Structure

- `pdr_only_IMU.py` — IMU-only PDR pipeline
- `fused11march_with_throttling_26march.py` — PDR + Wi-Fi EKF fusion with throttling experiment
- `wifi_localizer.py` — Wi-Fi wkNN localizer
- `floors/4-floor/traj7/` — IMU, Wi-Fi, and ground truth CSV files for trajectory 7
- `floors/4-floor/traj8/` — trajectory 8
- `floors/4-floor/traj9/` — trajectory 9
- `new_radiomap/radiomap.csv` — radio map with 395 reference points
- `README.md` — this file

---

## Data Format

**IMU CSV** (`imu_*.csv`):
| Column | Description |
|---|---|
| ts | Timestamp (ms) |
| ax, ay, az | Accelerometer (m/s²) |
| gx, gy, gz | Gyroscope (rad/s) |
| mx, my, mz | Magnetometer (μT) |

**Ground Truth CSV** (`traj_*.csv`):
| Column | Description |
|---|---|
| ts | Timestamp (ms) |
| x, y | Position coordinates (m) |

**Wi-Fi CSV** (`wifi_*.csv`):
| Column | Description |
|---|---|
| ts | Timestamp (ms) |
| bssid | Access point identifier |
| rssi | Received Signal Strength (dBm) |

**Radio Map CSV** (`radiomap.csv`):
| Column | Description |
|---|---|
| x, y | Reference point coordinates (m) |
| bssid | Access point identifier |
| rssi | Received Signal Strength (dBm) |

---

## Requirements
Python 3.8+
numpy
pandas
scipy
matplotlib

Install all dependencies:

```bash
pip install numpy pandas scipy matplotlib
```

---

## How to Run

### IMU-only PDR

```bash
python pdr_only_IMU.py
```

To change trajectory, edit the `name` variable at the bottom of the script:

```python
name = 'traj7'  # change to traj8 or traj9
```

### PDR + Wi-Fi EKF Fusion (with throttling experiment)

```bash
python fused11march_with_throttling_26march.py
```

This runs all trajectories across four Wi-Fi scan intervals: ~5 s, 10 s, 20 s, and 30 s.

---

## Device & Environment

| Parameter | Value |
|---|---|
| Device | Samsung Galaxy A52 (SM-A525F) |
| IMU sampling rate | ~48 Hz |
| Sensors recorded | Accelerometer, Gyroscope, Magnetometer |
| Environment | Floor 4, Block 7, Nazarbayev University |
| Floor area | ~1800 m² |
| Trajectories | 3 (traj7, traj8, traj9) |
| Radio map size | 395 reference points |

---

## Results Summary

| Approach | Mean Error (m) | Median Error (m) |
|---|---|---|
| IMU-only PDR | 7.8 – 28.5 | 8.0 – 21.8 |
| Wi-Fi-only | 3.6 – 4.5 | 3.0 – 3.7 |
| PDR + Wi-Fi fusion (~5 s) | 4.1 – 11.7 | 4.0 – 10.5 |
| PDR + Wi-Fi fusion (30 s) | 7.3 – 26.1 | 6.8 – 27.4 |

---

## Pipeline Overview
IMU sensor  →  Low-pass filter  →  Step detection (FFT)
→  Step length (Weinberg + ZUPT)
→  Heading (Madgwick filter)
↓
Wi-Fi scanner  →  wkNN matching  →  Position fix + confidence
↓
EKF Fusion [x, y, ψ_offset, s_scale]
↓
Corrected trajectory (x, y)

---

## License

This repository is private and intended for academic evaluation purposes only.# Indoor Localization Using IMU Sensors and Wi-Fi Fusion

