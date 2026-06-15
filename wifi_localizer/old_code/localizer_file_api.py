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


def parse_wifi_txt(txt_path: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      ts_ms (int), bssid (str), rssi (float)
    """
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.strip().split(";")
            if len(p) < 6:
                continue
            if p[0].strip().upper() != "WIFI":
                continue
            try:
                ts = int(float(p[2]))
                b = norm_mac(p[4])
                r = float(p[-1])
                if b is not None and np.isfinite(r):
                    rows.append((ts, b, r))
            except:
                continue
    df = pd.DataFrame(rows, columns=["ts_ms", "bssid", "rssi"])
    return df.sort_values("ts_ms").reset_index(drop=True)


def group_wifi_scans_median(wifi_df: pd.DataFrame, window_ms: int = 600) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Groups packets into scan windows and returns:
      scan_ts_ms: (S,) representative timestamp per scan
      scans: list[dict[bssid]=median_rssi] length S
    """
    if wifi_df.empty:
        return np.array([], dtype=np.int64), []

    scan_ts = []
    scans = []

    cur_start = int(wifi_df.loc[0, "ts_ms"])
    cur_end = cur_start
    cur: Dict[str, List[float]] = {}

    def flush():
        if not cur:
            return
        rep = (cur_start + cur_end) // 2
        scan = {b: float(np.median(v)) for b, v in cur.items() if len(v)}
        if scan:
            scan_ts.append(rep)
            scans.append(scan)

    for _, row in wifi_df.iterrows():
        ts = int(row["ts_ms"])
        b = row["bssid"]
        r = float(row["rssi"])

        if ts - cur_end <= window_ms:
            cur_end = ts
        else:
            flush()
            cur.clear()
            cur_start = ts
            cur_end = ts

        cur.setdefault(b, []).append(r)

    flush()
    return np.array(scan_ts, dtype=np.int64), scans


class WifiFileLocalizer:
    """
    File+timestamp WiFi estimator.

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
        wifi_window_ms: int = 600,
        max_dt_ms: int = 50,
    ):
        self.dataset_dir = dataset_dir
        self.k = int(k)
        self.min_overlap = int(min_overlap)
        self.wifi_window_ms = int(wifi_window_ms)
        self.max_dt_ms = int(max_dt_ms)

        self._load_radiomap(radiomap_csv)

        self._scan_cache: Dict[str, Tuple[np.ndarray, List[Dict[str, float]]]] = {}

        self._used_scan_ts_by_run: Dict[str, set] = {}

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
        # map normalized mac -> column index
        self.feat_index = {norm_mac(c): i for i, c in enumerate(wifi_cols)}
        self.train_X = rm[wifi_cols].to_numpy(dtype=float)
        self.train_Y = rm[["x", "y"]].to_numpy(dtype=float)

    def _load_file_scans(self, file_name: str):
        if file_name in self._scan_cache:
            return self._scan_cache[file_name]

        txt_path = os.path.join(self.dataset_dir, file_name + ".txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Missing txt file: {txt_path}")

        wifi_df = parse_wifi_txt(txt_path)
        ts_arr, scans = group_wifi_scans_median(wifi_df, window_ms=self.wifi_window_ms)

        self._scan_cache[file_name] = (ts_arr, scans)
        return ts_arr, scans

    def _pick_scan(self, ts_arr: np.ndarray, scans: List[Dict[str, float]], t_ms: int):
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
                reason="no_scan_within_tolerance",
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
                dt_ms=abs(chosen_ts - int(t_ms)),
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
                dt_ms=abs(chosen_ts - int(t_ms)),
                overlap_best=0,
                confidence=0.0,
            )
            return (float("nan"), float("nan")), meta

        ids = np.where(finite)[0]
        ids = ids[np.argsort(dist[finite])]  # closest first
       
        nn = ids[: self.k]
        nn = np.array(nn, dtype=int)
        nn_d = dist[nn]
        w = 1.0 / (nn_d + 1e-9)
        w = w / np.sum(w)
        est = (self.train_Y[nn] * w[:, None]).sum(axis=0)
            
        ov_best = int(overlap[ids[0]])
        # simple confidence based on overlap (stable)
        conf = float(np.clip((ov_best - self.min_overlap) / 25.0, 0.0, 1.0))

        meta = EstimateMeta(
            accepted=True,
            reason="ok",
            chosen_scan_ts_ms=chosen_ts,
            dt_ms=abs(chosen_ts - int(t_ms)),
            overlap_best=ov_best,
            confidence=conf,
        )
        return (float(est[0]), float(est[1])), meta
    def start_run(self, run_id: str, *, reset: bool = True, cooldown_steps: Optional[int] = None):
        """
        Call once at the start of a trajectory/simulation episode.
        """
        self._current_run_id = str(run_id)
        if reset or run_id not in self._used_scan_ts_by_run:
            self._used_scan_ts_by_run[run_id] = set()

    def end_run(self):
        """
        Optional. Clears current run pointer (does not delete stored sets unless you want).
        """
        self._current_run_id = None
