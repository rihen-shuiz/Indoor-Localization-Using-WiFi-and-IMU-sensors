#localizer_file_api.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import re

import numpy as np
import pandas as pd


MAC_RE = re.compile(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})")

# s = "Device_BSSID=00:1A:2B:3C:4D:5E, signal=-70" --> 00:1A:2B:3C:4D:5E
def norm_mac(s: str) -> Optional[str]:
    '''
    Extracts and normalizes a MAC address from a raw string.
    
    Args:
        s (str): The raw input string or log entry potentially containing 
            a MAC address (e.g., "BSSID: 00:1A:2B:3C:4D:5E").
    
    Returns:
        Optional[str]: The cleaned, lowercase MAC address string if a valid 
            pattern is found (e.g., "00:1a:2b:3c:4d:5e"); otherwise, None.  
    '''
    # Guard clause: Return early if the input variable is missing
    if s is None:
        return None
    
    # Scan the text string using the pre-compiled regex pattern
    m = MAC_RE.search(str(s))
    
    # If a match object exists, extract Capture Group 1 and lowercase it
    return m.group(1).lower() if m else None


def _masked_distances(train_X: np.ndarray, query_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the signal-space distance between a live scan and the radio map.

    This function computes a robust Root Mean Squared Error (RMSE) distance 
    by masking out missing signals (NaNs). It evaluates distance only on 
    the WiFi routers mutually observed by both the live query and the 
    database points, ensuring fair matching under sparse data conditions.

    Args:
        train_X (np.ndarray): 2D reference radio map matrix of shape 
            (num_locations, num_routers) containing historical RSSI values.
        query_x (np.ndarray): 1D live smartphone scan vector of shape 
            (num_routers,) containing real-time RSSI readings.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - dist (np.ndarray): 1D array of RMSE distances to each location. 
              Locations with zero overlapping routers are marked as np.inf.
            - overlap (np.ndarray): 1D array counting the number of mutually 
              observed routers for each location.
    """
    # 1. Reshape query to 2D (1, num_routers) to enable fast matrix broadcasting
    q = query_x[None, :]

    # 2. Identify coordinates where both matrices contain valid numbers (no NaNs)
    mask = np.isfinite(train_X) & np.isfinite(q)
    
    # 3. Count how many valid routers are shared with each database coordinate
    overlap = mask.sum(axis=1)

    # 4. Compute differences only where mask is True; safely force blanks to 0.0
    diff = np.where(mask, train_X - q, 0.0)
    
    # 5. Calculate Sum of Squared Errors (SSE) per row
    sse = (diff * diff).sum(axis=1) 

    # 6. Normalize by overlap count; if overlap is zero, assign infinite distance
    mse = np.where(overlap > 0, sse / np.maximum(overlap, 1), np.inf)

    # 7. Return the final square-rooted distance vector and the overlap counts
    dist = np.sqrt(mse)
    return dist, overlap


@dataclass
class EstimateMeta:
    """Metadata container describing the status and quality of a WiFi position estimate.

    Attributes:
        accepted (bool): True if the coordinate estimate is valid and passed all filters.
        reason (str): Diagnostic string explaining the outcome (e.g., 'ok' or failure reasons).
        chosen_scan_ts_ms (Optional[int]): The exact dataset timestamp of the matched WiFi scan.
        dt_ms (Optional[int]): Time offset between the requested timestamp and chosen scan.
        overlap_best (int): The number of mutual BSSIDs shared with the best radio map candidate.
        confidence (float): Trust metric scaled between 0.0 and 1.0 based on router overlap.
    """
    accepted: bool
    reason: str
    chosen_scan_ts_ms: Optional[int]
    dt_ms: Optional[int]
    overlap_best: int
    confidence: float


def parse_wifi_csv(csv_path: str) -> pd.DataFrame:
    """Loads, validates, and cleans a raw WiFi scan data file.

    Args:
        csv_path (str): The local system path to the target raw WiFi CSV file.

    Raises:
        ValueError: If the source CSV lacks the required 'ts', 'bssid', or 
            'rssi' column headers.

    Returns:
        pd.DataFrame: A clean DataFrame indexed sequentially with columns:
            - ts_ms (int64): Millisecond timestamps sorted chronologically.
            - bssid (str): Normalized lowercase MAC address identifiers.
            - rssi (float64): Clean numeric signal strength values.
    """
    df = pd.read_csv(csv_path)

    # Validate that all core variables exist in file header
    required = {"ts", "bssid", "rssi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WiFi CSV missing required columns: {sorted(missing)}")

    # Isolate data copy and force uniform formatting
    out = df.copy()
    out["ts_ms"] = pd.to_numeric(out["ts"], errors="coerce")
    out["bssid"] = out["bssid"].map(norm_mac)
    out["rssi"] = pd.to_numeric(out["rssi"], errors="coerce")

    # Purge any row containing missing or corrupted properties
    out = out.dropna(subset=["ts_ms", "bssid", "rssi"])
    out["ts_ms"] = out["ts_ms"].astype(np.int64)

    # Slice target tracking columns, sort sequentially, and rebuild row index
    return out[["ts_ms", "bssid", "rssi"]].sort_values("ts_ms").reset_index(drop=True)


def group_wifi_rows_by_exact_ts(wifi_df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Groups flat WiFi log records into chronologically synchronized scan packets.

    This function collapses a long table of continuous router entries into distinct 
    time buckets. For every unique millisecond timestamp, it consolidates all 
    detected access points. If a single router (BSSID) registers multiple signals 
    at the exact same millisecond, it calculates and stores the median RSSI to 
    filter out random multipath anomalies and duplicates.

    Args:
        wifi_df (pd.DataFrame): A parsed, cleaned WiFi data table containing 
            at least 'ts_ms', 'bssid', and 'rssi' columns.

    Returns:
        Tuple[np.ndarray, List[Dict[str, float]]]: A paired tuple containing:
            - scan_ts_ms (np.ndarray): A 1D array of shape (S,) holding sorted, 
              unique integer millisecond timestamps.
            - scans (List[Dict[str, float]]): A list of length S, where each item 
              is a dictionary mapping BSSID strings to their clean median RSSI values.
    """
    # 1. Guard clause: Return early with empty structures if the input table has no rows
    if wifi_df.empty:
        return np.array([], dtype=np.int64), []

    scan_ts = []
    scans = []

    # 2. Iterate through chronologically sorted time slices of the dataset
    for ts, g in wifi_df.groupby("ts_ms", sort=True):

        # 3. Collapse duplicate BSSIDs within the same millisecond group to their median RSSI
        scan = g.groupby("bssid")["rssi"].median().to_dict()
        
        # 4. If the resulting dictionary is valid and not empty, process its content
        if scan:
            scan_ts.append(int(ts))

            # 5. Build a clean dictionary subset filtering out any invalid/infinite signals
            scans.append({str(b): float(r) for b, r in scan.items() if np.isfinite(r)})

    # 6. Convert the timeline list to a high-precision NumPy array and return both tracking loops
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
    
        # TODO: Check if wkNN with k=3 is better than just k=1.
        #       Currently using k=1.
        """ # weighted k-nearest neighbors (wkNN)
        k = min(self.k, np.sum(finite))
        k_indices = np.argsort(dist)[:k]
        k_dists = dist[k_indices]
        k_xy = self.train_Y[k_indices]
        
        weights = 1.0 / (k_dists + 1e-6)
        weights /= np.sum(weights)
        best_xy = np.average(k_xy, axis=0, weights=weights)
    
        ov_best = int(np.max(overlap[k_indices]))
        conf = float(np.clip((ov_best - self.min_overlap) / 25.0, 0.0, 1.0)) """
    
        # strict nearest neighbor
        best_idx = int(np.argmin(dist))
        best_xy = self.train_Y[best_idx]
    
        ov_best = int(overlap[best_idx])
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