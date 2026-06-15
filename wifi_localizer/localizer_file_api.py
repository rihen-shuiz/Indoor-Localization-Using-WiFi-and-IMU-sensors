#localizer_file_api.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import re

import numpy as np
import pandas as pd


MAC_RE = re.compile(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})")


def norm_mac(s: str) -> Optional[str]:
    if s is None:
        return None
    m = MAC_RE.search(str(s))
    return m.group(1).lower() if m else None


def _masked_distances(train_X: np.ndarray, query_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = query_x[None, :]
    mask = np.isfinite(train_X) & np.isfinite(q)
    overlap = mask.sum(axis=1)

    diff = np.where(mask, train_X - q, 0.0)
    sse = (diff * diff).sum(axis=1)
    mse = np.where(overlap > 0, sse / np.maximum(overlap, 1), np.inf)
    dist = np.sqrt(mse)
    return dist, overlap


@dataclass
class EstimateMeta:
    accepted: bool
    reason: str
    chosen_scan_ts_ms: Optional[int]
    dt_ms: Optional[int]
    overlap_best: int
    confidence: float


def parse_wifi_csv(csv_path: str) -> pd.DataFrame:
    """
    Input wifi csv format:
      ts,bssid,ssid,rssi,freq

    Returns DataFrame with columns:
      ts_ms (int), bssid (str), rssi (float)
    """
    df = pd.read_csv(csv_path)

    required = {"ts", "bssid", "rssi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WiFi CSV missing required columns: {sorted(missing)}")

    out = df.copy()
    out["ts_ms"] = pd.to_numeric(out["ts"], errors="coerce")
    out["bssid"] = out["bssid"].map(norm_mac)
    out["rssi"] = pd.to_numeric(out["rssi"], errors="coerce")

    out = out.dropna(subset=["ts_ms", "bssid", "rssi"])
    out["ts_ms"] = out["ts_ms"].astype(np.int64)

    return out[["ts_ms", "bssid", "rssi"]].sort_values("ts_ms").reset_index(drop=True)


def group_wifi_rows_by_exact_ts(wifi_df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Groups rows by exact timestamp and returns:
      scan_ts_ms: (S,) exact timestamps
      scans: list[dict[bssid]=median_rssi] length S
    """
    if wifi_df.empty:
        return np.array([], dtype=np.int64), []

    scan_ts = []
    scans = []

    for ts, g in wifi_df.groupby("ts_ms", sort=True):
        # if same BSSID appears multiple times at same ts, keep median RSSI
        scan = g.groupby("bssid")["rssi"].median().to_dict()
        if scan:
            scan_ts.append(int(ts))
            scans.append({str(b): float(r) for b, r in scan.items() if np.isfinite(r)})

    return np.array(scan_ts, dtype=np.int64), scans


class WifiFileLocalizer2:
    """
    File+timestamp WiFi estimator.

    Now expects per-file WiFi CSVs with format:
      ts,bssid,ssid,rssi,freq

    Call:
      loc = WifiFileLocalizer(radiomap_csv, dataset_dir)
      xy, meta = loc.estimate("DATA_...", t_ms=...)
    """

    def __init__(
        self,
        radiomap_csv: str,
        dataset_dir: str,
        k: int = 3,
        min_overlap: int = 20,
        wifi_window_ms: int = 600,   # kept only for minimal API change; no longer used
        max_dt_ms: int = 50,         # kept only for minimal API change; exact ts is required now
    ):
        self.dataset_dir = dataset_dir
        self.k = int(k)
        self.min_overlap = int(min_overlap)
        self.wifi_window_ms = int(wifi_window_ms)
        self.max_dt_ms = int(max_dt_ms)

        self._load_radiomap(radiomap_csv)

        self._scan_cache: Dict[str, Tuple[np.ndarray, List[Dict[str, float]]]] = {}
        self._used_scan_ts_by_run: Dict[str, set] = {}
        self._current_run_id: Optional[str] = None

    def _load_radiomap(self, path: str):
        rm = pd.read_csv(path)
        if "x" not in rm.columns or "y" not in rm.columns:
            raise ValueError("Radiomap must contain x and y columns.")

        wifi_cols = [c for c in rm.columns if norm_mac(c) is not None]
        if not wifi_cols:
            raise ValueError("No BSSID columns found in radiomap.")

        rm[wifi_cols] = rm[wifi_cols].apply(pd.to_numeric, errors="coerce")
        rm["x"] = pd.to_numeric(rm["x"], errors="coerce")
        rm["y"] = pd.to_numeric(rm["y"], errors="coerce")
        rm = rm.dropna(subset=["x", "y"]).reset_index(drop=True)

        self.wifi_cols = wifi_cols
        self.feat_index = {norm_mac(c): i for i, c in enumerate(wifi_cols)}
        self.train_X = rm[wifi_cols].to_numpy(dtype=float)
        self.train_Y = rm[["x", "y"]].to_numpy(dtype=float)

    def _load_file_scans(self, file_name: str):
        if file_name in self._scan_cache:
            return self._scan_cache[file_name]

        csv_path = os.path.join(self.dataset_dir, file_name + ".csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing csv file: {csv_path}")

        wifi_df = parse_wifi_csv(csv_path)
        ts_arr, scans = group_wifi_rows_by_exact_ts(wifi_df)

        self._scan_cache[file_name] = (ts_arr, scans)
        return ts_arr, scans

    def _pick_scan(self, ts_arr: np.ndarray, scans: List[Dict[str, float]], t_ms: int):
        """
        Exact timestamp match only.
        """
        if ts_arr.size == 0:
            print("zero array")
            return None, None
        i = int(np.searchsorted(ts_arr, t_ms))
        cand = []
        if 0 <= i < len(ts_arr): cand.append(i)
        if i - 1 >= 0: cand.append(i - 1)
        if i + 1 < len(ts_arr): cand.append(i + 1)

        best_i = None
        best_dt = None
        for j in cand:
            dt = abs(int(ts_arr[j]) - int(t_ms))
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_i = j

        if best_i is None or best_dt is None or best_dt > self.max_dt_ms:
            return None, None
        return int(ts_arr[best_i]), scans[best_i]

    def _vectorize_scan(self, scan: Dict[str, float]) -> np.ndarray:
        q = np.full((len(self.wifi_cols),), np.nan, dtype=float)
        for b, r in scan.items():
            idx = self.feat_index.get(norm_mac(b))
            if idx is not None and np.isfinite(r):
                q[idx] = float(r)
        return q

    def estimate(self, file_name: str, t_ms: int) -> Tuple[Tuple[float, float], EstimateMeta]:
        ts_arr, scans = self._load_file_scans(file_name)
        chosen_ts, scan = self._pick_scan(ts_arr, scans, t_ms)
    
        if scan is None:
            meta = EstimateMeta(
                accepted=False,
                reason="no_exact_timestamp_scan",
                chosen_scan_ts_ms=None,
                dt_ms=None,
                overlap_best=0,
                confidence=0.0,
            )
            return (float("nan"), float("nan")), meta
    
        run_id = self._current_run_id or file_name
        used_scan_ts = self._used_scan_ts_by_run.setdefault(run_id, set())
    
        if chosen_ts in used_scan_ts:
            meta = EstimateMeta(
                accepted=False,
                reason="scan_timestamp_already_used",
                chosen_scan_ts_ms=chosen_ts,
                dt_ms=0,
                overlap_best=0,
                confidence=0.0,
            )
            return (float("nan"), float("nan")), meta
    
        used_scan_ts.add(chosen_ts)
    
        q = self._vectorize_scan(scan)
        dist, overlap = _masked_distances(self.train_X, q)
    
        dist = np.where(overlap >= self.min_overlap, dist, np.inf)
        finite = np.isfinite(dist)
        if not np.any(finite):
            meta = EstimateMeta(
                accepted=False,
                reason="no_candidates_min_overlap",
                chosen_scan_ts_ms=chosen_ts,
                dt_ms=0,
                overlap_best=0,
                confidence=0.0,
            )
            return (float("nan"), float("nan")), meta
    
        # weighted k-nearest neighbors (wkNN)
        k = min(self.k, np.sum(finite))
        k_indices = np.argsort(dist)[:k]
        k_dists = dist[k_indices]
        k_xy = self.train_Y[k_indices]
        
        weights = 1.0 / (k_dists + 1e-6)
        weights /= np.sum(weights)
        best_xy = np.average(k_xy, axis=0, weights=weights)
    
        ov_best = int(np.max(overlap[k_indices]))
        conf = float(np.clip((ov_best - self.min_overlap) / 25.0, 0.0, 1.0))
    
        meta = EstimateMeta(
            accepted=True,
            reason="ok",
            chosen_scan_ts_ms=chosen_ts,
            dt_ms=0,
            overlap_best=ov_best,
            confidence=conf,
        )
    
        return (float(best_xy[0]), float(best_xy[1])), meta

    def start_run(self, run_id: str, *, reset: bool = True, cooldown_steps: Optional[int] = None):
        """
        Call once at the start of a trajectory/simulation episode.
        cooldown_steps kept for compatibility, not used.
        """
        self._current_run_id = str(run_id)
        if reset or run_id not in self._used_scan_ts_by_run:
            self._used_scan_ts_by_run[run_id] = set()

    def end_run(self):
        """
        Optional. Clears current run pointer.
        """
        self._current_run_id = None