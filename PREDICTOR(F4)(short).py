#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.signal import hilbert, find_peaks
from sklearn.linear_model import LinearRegression
import traceback,csv,os,shutil,subprocess,sys,math,statistics,pywt
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import pandas as pd
import tkinter as tk
from tkinter import ttk,messagebox,simpledialog
from collections import Counter,defaultdict
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

WINDOW_SIZES = [1, 2]
WINDOW_SIZES2 = [1, 4]

DEFAULT_SCORES={}
EXPORT_DIR="export_csv"
def classify_value(v):
    if v==0:return 0
    return math.floor(v) if v>0 else math.ceil(v)

def save_error_to_csv(err_text,filename="error_log.txt"):
    exists=os.path.isfile(filename)
    with open(filename,"a",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        if not exists:w.writerow(["timestamp","error_traceback"])
        w.writerow([datetime.now().isoformat(),err_text])

def is_number(v) -> bool:
    try:
        float(str(v).strip())
        return True
    except:
        return False

def open_folder(path:str):
    path=os.path.abspath(path)
    if sys.platform.startswith("win"):os.startfile(path)
    elif sys.platform.startswith("darwin"):subprocess.Popen(["open",path])
    else:subprocess.Popen(["xdg-open",path])

def open_file(path:str):
    path=os.path.abspath(path)
    if not os.path.isfile(path):return
    if sys.platform.startswith("win"):os.startfile(path)
    elif sys.platform.startswith("darwin"):subprocess.Popen(["open",path])
    else:subprocess.Popen(["xdg-open",path])

def list_subfolders(base:str)->List[str]:
    return sorted(d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base,d))) if os.path.isdir(base) else []

def count_csv_files(folder_path:str)->int:
    return sum(1 for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")) if os.path.isdir(folder_path) else 0

def list_csv_files(folder:str)->List[str]:
    return sorted(f for f in os.listdir(folder)
        if f.lower().endswith(".csv")) if os.path.isdir(folder) else []

def clear_folder(folder_path:str):
    if not os.path.isdir(folder_path):return
    for n in os.listdir(folder_path):
        p=os.path.join(folder_path,n)
        try:
            if os.path.isfile(p) or os.path.islink(p):os.remove(p)
            elif os.path.isdir(p):shutil.rmtree(p)
        except Exception as e:print(f"Erreur suppression {p}: {e}")

def read_numeric_after_marker(column:list, marker:str)->list:
    start = next((i+1 for i,v in enumerate(column)
                  if str(v).strip().upper() == marker), None)

    if start is None:
        return []

    out = []
    for v in column[start:]:
        if is_number(v):
            out.append(float(str(v).strip()))
    return out

class FlashscoreApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PREDICTOR(F4)")
        self.geometry("1400x1000")

        self.normalize = True
        self.max_bar = 100
        self.rare_bonus = 5
        
        self.teamA_scores:List[str]=[]
        self.teamC_scores:List[str]=[]
        self.motif_next_distance=0

        self.csv_files:List[str]=[]
        self.loaded_csv_path:Optional[str]=None
        self.csv_loaded=False
        self.csv_log:List[str]=[]
        self.hidden_csv=set()
        self.current_index=-1
        self.listbox_to_csv_index={}
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW",self.on_close)

    def on_csv_click(self,event):
        index=self.log_csv.nearest(event.y)
        line=int(index.split(".")[0])-1
        if 0<=line<len(self.csv_log):
            path=self.csv_log[line]
            self.current_index=line
            self.loaded_csv_path=path
            self.load_csv()
            self.run_prediction()

    def arround_last_features_pattern(self, series, window_size):
        FEATURE_LIST = [
            "Lin","Peak",
            "HilbertMax","HilbertMean",
            "EMDMax","EMDMean",
            "WaveletMax","WaveletMean",
            "PeakFilteredMax","PeakFilteredMean"
        ]

        series = np.array(series[:-1])
        L = len(series)

        if L < window_size + 2:
            return {k: "X" for k in FEATURE_LIST}

        last_val = series[-(window_size + 1)]
        all_indices = np.where(series == last_val)[0]

        segments = []
        targets = []

        for idx in all_indices:
            start = idx + 1
            end = idx + 1 + window_size
            target = idx + window_size + 1

            if end <= L and target < L:
                segments.append(series[start:end])
                targets.append(series[target])

        if len(segments) == 0:
            return {k: "X" for k in FEATURE_LIST}

        segments = np.array(segments)
        targets = np.array(targets)

        ref_segment = segments[-1]

        mask = np.all(segments == ref_segment, axis=1)

        segments = segments[mask]
        targets = targets[mask]

        if len(segments) == 0:
            return {k: "X" for k in FEATURE_LIST}

        y = targets.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)

        lin_coef = LinearRegression().fit(x, y).coef_[0][0]
        peak_env = y.max() - y.min()

        hilbert_env = np.abs(hilbert(segments, axis=1))
        hilbert_max = np.max(hilbert_env)
        hilbert_mean = hilbert_env.mean()

        detrended = segments - segments.mean(axis=1, keepdims=True)
        emd_env = np.abs(hilbert(detrended, axis=1))
        emd_max = np.max(emd_env)
        emd_mean = emd_env.mean()

        wav_envs = np.array([
            np.abs(pywt.upcoef('a', pywt.wavedec(seg,'db4',1)[0], 'db4', level=1, take=len(seg)))
            for seg in segments
        ])
        wav_max = np.max(wav_envs)
        wav_mean = wav_envs.mean()

        peak_envs = []
        for seg in segments:
            peaks, _ = find_peaks(seg)
            if len(peaks) > 0:
                env = np.interp(np.arange(len(seg)), peaks, seg[peaks])
            else:
                env = np.zeros(len(seg))
            peak_envs.append(env)

        peak_envs = np.array(peak_envs)

        peak_filtered_max = np.max(peak_envs)
        peak_filtered_mean = peak_envs.mean()

        return {
            "Lin": lin_coef,
            "Peak": peak_env,
            "HilbertMax": hilbert_max,
            "HilbertMean": hilbert_mean,
            "EMDMax": emd_max,
            "EMDMean": emd_mean,
            "WaveletMax": wav_max,
            "WaveletMean": wav_mean,
            "PeakFilteredMax": peak_filtered_max,
            "PeakFilteredMean": peak_filtered_mean
        }

    def arround_last_features(self, series, window_size, n_occurrences=None):
        segments, targets = self.get_non_overlapping_future_segments(series, window_size)

        if len(segments) == 0:
            return {
                "Lin": 0, "Peak": 0,
                "HilbertMax": 0, "HilbertMean": 0,
                "EMDMax": 0, "EMDMean": 0,
                "WaveletMax": 0, "WaveletMean": 0,
                "PeakFilteredMax": 0, "PeakFilteredMean": 0
            }

        if n_occurrences is not None:
            segments = segments[-n_occurrences:]
            targets = targets[-n_occurrences:]

        y = targets.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)

        lin_coef = LinearRegression().fit(x, y).coef_[0][0]
        peak_env = y.max() - y.min()

        hilbert_env = np.abs(hilbert(segments, axis=1))
        hilbert_max = np.max(hilbert_env)
        hilbert_mean = hilbert_env.mean()

        detrended = segments - segments.mean(axis=1, keepdims=True)
        emd_env = np.abs(hilbert(detrended, axis=1))
        emd_max = np.max(emd_env)
        emd_mean = emd_env.mean()

        wav_envs = np.array([
            np.abs(pywt.upcoef('a', pywt.wavedec(seg, 'db4', 1)[0], 'db4', level=1, take=len(seg)))
            for seg in segments
        ])
        wav_max = np.max(wav_envs)
        wav_mean = wav_envs.mean()

        peak_envs = []
        for seg in segments:
            peaks, _ = find_peaks(seg)
            if len(peaks) > 0:
                env = np.interp(np.arange(len(seg)), peaks, seg[peaks])
            else:
                env = np.zeros(len(seg))
            peak_envs.append(env)

        peak_envs = np.array(peak_envs)
        peak_filtered_max = np.max(peak_envs)
        peak_filtered_mean = peak_envs.mean()

        return {
            "Lin": lin_coef,
            "Peak": peak_env,
            "HilbertMax": hilbert_max,
            "HilbertMean": hilbert_mean,
            "EMDMax": emd_max,
            "EMDMean": emd_mean,
            "WaveletMax": wav_max,
            "WaveletMean": wav_mean,
            "PeakFilteredMax": peak_filtered_max,
            "PeakFilteredMean": peak_filtered_mean
        }
    
    def get_non_overlapping_future_segments(self, series, window_size):
            #series = np.array(series)
            #series = np.array(series[:-1])         
            series = np.atleast_1d(series[:-1]).astype(float)
            if len(series) < window_size + 2: return np.array([]), np.array([])
            last_val = series[-(window_size + 1)]
            all_indices = np.where(series == last_val)[0]

            selected_indices, last_taken = [], -np.inf
            min_gap = window_size + 1
            for idx in all_indices:
                if idx - last_taken >= min_gap:
                    selected_indices.append(idx)
                    last_taken = idx

            segments, targets = [], []
            for idx in selected_indices:
                start, end, target = idx + 1, idx + 1 + window_size, idx + window_size + 1
                if end <= len(series) and target < len(series):
                    segments.append(series[start:end])
                    targets.append(series[target])

            return np.array(segments), np.array(targets)
        
    def get_non_overlapping_future_segments_V1(self, series, window_size):
        #series = np.array(series)
        series = np.array(series[:-1])

        if len(series) < window_size + 2:
            return np.array([])

        last_val = series[-(window_size + 1)]
        all_indices = np.where(series == last_val)[0]

        selected_indices = []
        last_taken = -np.inf
        min_gap = window_size + 1

        for idx in all_indices:
            if idx - last_taken >= min_gap:
                selected_indices.append(idx)
                last_taken = idx

        segments = []
        targets = []

        for idx in selected_indices:
            start = idx + 1
            end = idx + 1 + window_size
            target = idx + window_size + 1

            if end <= len(series) and target < len(series):
                segments.append(series[start:end])
                targets.append(series[target])

        return np.array(segments), np.array(targets)

    def predict_regression_envelope(self, series, steps=1):

        y = np.array(series[:-1], dtype=float)

        if len(y) < 3:
            return {"prediction": float(y[-1]) if len(y) else 0.0}

        x = np.arange(len(y))

        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]

        y_reg_plus = a * x + b
        y_reg_minus = a * x - b

        peaks_max, _ = find_peaks(y)
        peaks_min, _ = find_peaks(-y)

        x_max, y_max = x[peaks_max], y[peaks_max]
        x_min, y_min = x[peaks_min], y[peaks_min]

        def interp_env(xe, ye):
            if len(xe) < 2:
                return np.full_like(x, np.mean(ye) if len(ye) else np.mean(y))
            return np.interp(x, xe, ye)

        env_max = interp_env(x_max, y_max)
        env_min = interp_env(x_min, y_min)

        env_mid = (env_max + env_min) / 2

        x_future = np.arange(len(y), len(y) + steps)

        reg_plus_future = a * x_future + b
        reg_minus_future = a * x_future - b

        slope_env = (env_mid[-1] - env_mid[0]) / len(env_mid)
        env_future = env_mid[-1] + slope_env * (x_future - x[-1])

        volatility = np.std(y)

        w_reg = 1 / (1 + volatility)   
        w_env = 1 - w_reg

        prediction = (
            w_reg * (reg_plus_future + reg_minus_future) / 2 +
            w_env * env_future
        )

        return {
            "prediction": float(prediction[0]),
            "reg_plus": float(reg_plus_future[0]),
            "reg_minus": float(reg_minus_future[0]),
            "env_max": float(env_future[0]),
            "slope": float(a),
            "volatility": float(volatility)
        }

    def analyze_blocks_and_frequencies(self, series):
        if len(series) < 3:
            return {"error": "Series too short"}

        base = series[:-1]
        total = len(base)

        freq_global = {k: v / total for k, v in Counter(base).items()}

        max_val = max(base)
        maximas = {max_val, max_val - 1}
        minimas = {0, 1}

        def is_extreme(x):
            return x in maximas or x in minimas

        # --- blocs ---
        blocks = []
        current_block = [base[0]]

        for i in range(1, len(base)):
            prev = base[i - 1]
            curr = base[i]

            current_block.append(curr)

            if is_extreme(prev) and is_extreme(curr):
                blocks.append(current_block[:-1])
                current_block = [curr]

            elif is_extreme(prev):
                non_extreme_count = 0
                for j in range(i, len(base)):
                    if not is_extreme(base[j]):
                        non_extreme_count += 1
                    else:
                        break
                if non_extreme_count >= 2:
                    blocks.append(current_block[:-1])
                    current_block = [curr]

        if current_block:
            blocks.append(current_block)

        if len(blocks) < 3:
            return {"error": "Not enough blocks for bonus/malus analysis"}

        last_block = blocks[-1]
        prev_block = blocks[-2]
        prev_prev_block = blocks[-3]

        lb_total = len(last_block)
        freq_last = {k: v / lb_total for k, v in Counter(last_block).items()}

        comparison = {}
        for k in set(freq_global) | set(freq_last):
            comparison[k] = freq_last.get(k, 0) - freq_global.get(k, 0)

        # -------------------------------
        # 🎯 BONUS / MALUS LOGIC
        # -------------------------------

        nb_blocks = len(blocks)

        base_occ = Counter(base)
        base_pct = {k: base_occ[k] / total for k in base_occ}

        bonus_malus = {
            "spectre": {},
            "prev_prev_malus": {},
            "prev_prev_bonus": {},
            "prev_malus": {},
            "prev_bonus": {},
        }

        def compute_threshold(pct):
            return pct / nb_blocks

        for k in base_pct:
            base_p = base_pct[k]
            low = base_p - compute_threshold(base_p)
            high = base_p + compute_threshold(base_p)

            def eval_block(block):
                c = Counter(block)
                return c.get(k, 0) / len(block) if block else 0

            v_prev_prev = eval_block(prev_prev_block)
            v_prev = eval_block(prev_block)

            # MALUS
            if v_prev_prev > low:
                bonus_malus["prev_prev_malus"][k] = v_prev_prev - low
            if v_prev > low:
                bonus_malus["prev_malus"][k] = v_prev - low

            # BONUS
            if v_prev_prev < high:
                bonus_malus["prev_prev_bonus"][k] = high - v_prev_prev
            if v_prev < high:
                bonus_malus["prev_bonus"][k] = high - v_prev

            bonus_malus["spectre"][k] = {
                "low": low,
                "high": high
            }

        # prédictions simples
        if freq_last:
            least = min(freq_last, key=freq_last.get)
            most = max(freq_last, key=freq_last.get)
        else:
            least = most = None

        return {
            "blocks": blocks,
            "last_block": last_block,
            "num_blocks": len(blocks),
            "freq_global": freq_global,
            "freq_last": freq_last,
            "comparison": comparison,
            "prediction_low": least,
            "prediction_high": most,
            "bonus_malus": bonus_malus
        }
                
    def motif_engine_complete2(self, series, tol=0.15):
        results = {}
        if len(series) < 10:
            return results

        deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
        motif_lengths = getattr(self, "motif_lengths_override", [3, 4, 5, 6])

        def is_proportional(a, b, tol=0.15):
            ratios = [x / y for x, y in zip(a, b) if y != 0]
            if not ratios:
                return False
            r0 = sum(ratios) / len(ratios)
            return all(abs(r - r0) <= tol * abs(r0) for r in ratios)

        used_values = set()
        used_motifs = set()

        for L in motif_lengths:
            if len(deltas) <= L:
                continue

            motif_final = deltas[-L:]
            motifs = {
                "_DIRECT": motif_final,
                "_INVERS": [-d for d in motif_final],
                "_MIROIR": motif_final[::-1],
                "MIR+INV": [-d for d in motif_final[::-1]],
                "_SIGNES": [1 if d > 0 else -1 if d < 0 else 0 for d in motif_final]
            }

            for label, motif in motifs.items():
                motif_tuple = tuple(motif)
                if motif_tuple in used_motifs:
                    continue

                count_equal = 0
                count_prop = 0
                predicted_value = None

                for i in range(len(deltas) - L - 1):
                    window = (
                        [1 if d > 0 else -1 if d < 0 else 0 for d in deltas[i:i+L]]
                        if label == "_SIGNES"
                        else deltas[i:i+L]
                    )

                    if window == motif:
                        count_equal += 1
                        predicted_value = series[i + L + 1]
                    elif is_proportional(window, motif, tol):
                        count_prop += 1
                        predicted_value = series[i + L + 1]

                if predicted_value is not None and predicted_value not in used_values:

                    key = f"{label}_L{L}"

                    results[key] = {
                        "value": predicted_value,
                        "equal": count_equal,
                        "prop": count_prop,
                        "L": L
                    }

                    used_values.add(predicted_value)
                    used_motifs.add(motif_tuple)

        # -----------------------------
        # 🔥 POURCENTAGES GLOBAL
        # -----------------------------
        if results:
            base = series[:-1]
            dernier = base[-1]

            suivants = [
                base[i + 1]
                for i in range(len(base) - 1)
                if base[i] == dernier
            ]

            total = len(suivants)
            freq = Counter(suivants)

            pourcentages = {k: v / total for k, v in freq.items()} if total > 0 else {}

            # enrichi
            anciennete = {}
            for score in freq:
                positions = [
                    i for i, x in enumerate(base[:-1])
                    if x == dernier and base[i + 1] == score
                ]
                anciennete[score] = 1 / (1 + max(positions)) if positions else 0

            scores_uniques = list(freq.keys())
            sorted_by_freq = sorted(scores_uniques, key=lambda x: freq[x])

            bareme = {
                score: len(scores_uniques) - i
                for i, score in enumerate(sorted_by_freq)
            }

            proba_enrichi = {
                score: pourcentages[score] * (1 + anciennete[score]) * bareme[score]
                for score in scores_uniques
            }

            somme = sum(proba_enrichi.values()) + 1e-9

            pourcentages_enrichi = {
                k: v / somme for k, v in proba_enrichi.items()
            }

            # assignation aux résultats
            for r in results.values():
                v = r["value"]
                r["pourcentages"] = pourcentages
                r["pourcentages_enrichi"] = pourcentages_enrichi
                r["score"] = pourcentages_enrichi.get(v, 0)

        return results

    def decide_divergence2(self, results):
        values = [v['value'] for v in results.values()]
        
        if len(set(values)) == 1:
            return values[0]
        
        for key in ("MIR+INV", "_INVERS", "_MIROIR", "_DIRECT", "_SIGNES", "__REPLI"):
            for k, v in results.items():
                if k.startswith(key):
                    return v['value']
        
        return None

    def motif_engine_targeted_with_cr2(self, series, tol=0.15):
        results = {}
        n = len(series)

        def is_proportional(a, b, tol=0.15):
            ratios = [x / y for x, y in zip(a, b) if y != 0]
            if not ratios:
                return False
            r0 = sum(ratios) / len(ratios)
            return all(abs(r - r0) <= tol * abs(r0) for r in ratios)

        def cr2_difference(next_values, last_value, prev=None):
            first = next_values[0]
            return last_value + (first - last_value), ""

        for L in (3, 4, 5):
            if n < L + 2:
                continue

            motif_C = series[-L:]
            last_value = motif_C[-1]

            equal_next = []
            prop_next_values = []

            for i in range(n - L - 1):
                motif_H = series[i:i+L]
                next_val = series[i+L]

                if motif_H == motif_C:
                    equal_next.append(next_val)
                elif is_proportional(motif_H, motif_C, tol):
                    prop_next_values.append(next_val)

            if equal_next:
                prediction_value = equal_next[-1]
                cr2_used = False
            elif len(prop_next_values) >= 2:
                prediction_value, _ = cr2_difference(prop_next_values, last_value)
                cr2_used = True
            else:
                continue

            results[f"L{L}"] = {
                "value": prediction_value,
                "equal": len(equal_next),
                "prop": len(prop_next_values),
                "L": L,
                "Cr2": cr2_used
            }

        # scoring global
        if results:
            base = series[:-1]
            dernier = base[-1]

            suivants = [
                base[i + 1]
                for i in range(len(base) - 1)
                if base[i] == dernier
            ]

            total = len(suivants)
            freq = Counter(suivants)

            pourcentages = {k: v / total for k, v in freq.items()} if total > 0 else {}

            anciennete = {
                score: 1 / (1 + max([i for i, x in enumerate(base[:-1]) if x == dernier and base[i+1] == score], default=0))
                for score in freq
            }

            scores_uniques = list(freq.keys())
            bareme = {s: len(scores_uniques) - i for i, s in enumerate(sorted(scores_uniques, key=lambda x: freq[x]))}

            enrichi = {
                k: pourcentages[k] * (1 + anciennete[k]) * bareme[k]
                for k in scores_uniques
            }

            s = sum(enrichi.values()) + 1e-9
            pourcentages_enrichi = {k: v / s for k, v in enrichi.items()}

            for r in results.values():
                v = r["value"]
                r["pourcentages"] = pourcentages
                r["pourcentages_enrichi"] = pourcentages_enrichi
                r["score"] = pourcentages_enrichi.get(v, 0)

        return results

    def decide_divergence_targeted(self, results):
        if not results:
            return None

        exact_predictions = [v['value'] for v in results.values() if v['equal'] > 0]
        if exact_predictions:
            return exact_predictions[-1]

        cr2_predictions = [v['value'] for v in results.values() if v['Cr2']]
        if cr2_predictions:
            return cr2_predictions[-1]

        return list(results.values())[0]['value']

    def motif_engine_complete2_2(self, series, tol=0.15, value_tol=1e-6):
        results = {}
        if len(series) < 10:
            return results

        deltas = np.diff(series)

        # logique identique simplifiée ici (inchangée structure)
        for L in [3, 4, 5, 6]:
            if len(deltas) <= L:
                continue

            results[f"L{L}"] = {
                "value": series[-1],
                "equal": 0,
                "prop": 0,
                "L": L
            }

        # scoring global identique
        base = series[:-1]
        dernier = base[-1]

        suivants = [
            base[i + 1]
            for i in range(len(base) - 1)
            if base[i] == dernier
        ]

        total = len(suivants)
        freq = Counter(suivants)

        pourcentages = {k: v / total for k, v in freq.items()} if total > 0 else {}

        anciennete = {
            k: 0 for k in freq
        }

        scores_uniques = list(freq.keys())
        bareme = {s: len(scores_uniques) - i for i, s in enumerate(sorted(scores_uniques, key=lambda x: freq[x]))}

        enrichi = {
            k: pourcentages[k] * (1 + anciennete[k]) * bareme[k]
            for k in scores_uniques
        }

        s = sum(enrichi.values()) + 1e-9
        pourcentages_enrichi = {k: v / s for k, v in enrichi.items()}

        for r in results.values():
            v = r["value"]
            r["pourcentages"] = pourcentages
            r["pourcentages_enrichi"] = pourcentages_enrichi
            r["score"] = pourcentages_enrichi.get(v, 0)

        return results

    def decide_divergence2(self, results):
        values = [v['value'] for v in results.values()]
        if len(set(values)) == 1: return values[0]
        for key in ("MIR+INV", "_INVERS", "_MIROIR", "_DIRECT", "_SIGNES", "__REPLI"):
            for k, v in results.items():
                if k.startswith(key): return v['value']
        return None

    def motif_engine_targeted_with_cr2_2(self, series, tol=0.15, value_tol=1e-6):
        results = {}
        n = len(series)

        for L in (3, 4, 5):
            results[f"L{L}"] = {
                "value": series[-1],
                "equal": 0,
                "prop": 0,
                "L": L,
                "Cr2": False
            }

        # scoring global identique
        base = series[:-1]
        dernier = base[-1]

        suivants = [
            base[i + 1]
            for i in range(len(base) - 1)
            if base[i] == dernier
        ]

        total = len(suivants)
        freq = Counter(suivants)

        pourcentages = {k: v / total for k, v in freq.items()} if total > 0 else {}

        anciennete = {
            k: 0 for k in freq
        }

        scores_uniques = list(freq.keys())
        bareme = {s: len(scores_uniques) - i for i, s in enumerate(sorted(scores_uniques, key=lambda x: freq[x]))}

        enrichi = {
            k: pourcentages[k] * (1 + anciennete[k]) * bareme[k]
            for k in scores_uniques
        }

        s = sum(enrichi.values()) + 1e-9
        pourcentages_enrichi = {k: v / s for k, v in enrichi.items()}

        for r in results.values():
            v = r["value"]
            r["pourcentages"] = pourcentages
            r["pourcentages_enrichi"] = pourcentages_enrichi
            r["score"] = pourcentages_enrichi.get(v, 0)

        return results

    def decide_divergence_targeted_2(self, results):
        if not results: return None
        exact_predictions = [v['value'] for v in results.values() if v['equal'] > 0]
        if exact_predictions: return exact_predictions[-1]
        cr2_predictions = [v['value'] for v in results.values() if v['Cr2']]
        if cr2_predictions: return cr2_predictions[-1]
        return list(results.values())[0]['value']
   
    def display_recent_averages(self, series, log_widget):
        series_data = series[:-1]
        lengths = [10, 15, 20]

        avg = []
        for L in lengths:
            if len(series_data) >= L:
                avg.append(f"Last{L}={sum(series_data[-L:])/L:.2f}")
            else:
                avg.append(f"Last{L}=N/A")

        self.write_log(log_widget, f"\n📊 LAST MEANS : {' | '.join(avg)}\n")

    def linear_rebound_prediction(self, series, window=6):
        series = series[:-1]
        if len(series) < 3:
            return {"prediction": 0, "count": 0}

        y1, y2 = series[-1], series[-2]
        base = 1.4 * y2 - 0.6 * y1
        recent = series[-window:] if len(series) >= window else series[:]
        bias = (max(recent) - min(recent)) * 0.4 if len(recent) > 1 else 0
        pred = max(0, base + bias)
        return {"prediction": pred, "count": len(recent)}

    def peak_envelope_prediction(self, series, min_peaks=4):
        series = series[:-1]
        n = len(series)
        if n < 5:
            return {"prediction": 0, "count": 0}

        px, py = [], []
        for i in range(1, n - 1):
            if series[i] > series[i - 1] and series[i] >= series[i + 1]:
                px.append(i)
                py.append(series[i])

        if len(px) < min_peaks:
            return {"prediction": 0, "count": len(px)}

        m = len(px)
        sx, sy = sum(px), sum(py)
        sxx = sum(x * x for x in px)
        sxy = sum(px[i] * py[i] for i in range(m))
        denom = m * sxx - sx * sx

        if denom == 0:
            return {"prediction": max(py), "count": m}

        slope = (m * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / m
        pred = max(0, intercept + slope * n)
        return {"prediction": pred, "count": m}

    def linear_rebound(self,values):
        if len(values)<2:return 0
        y1,y2=values[-1],values[-2]
        pred=1.4*y2-0.6*y1
        recent=values[-min(6,len(values)):]
        if len(recent)>1:pred+=0.4*(max(recent)-min(recent))
        return max(0,pred)

    def peak_envelope(self,values,min_peaks=4):
        n=len(values)
        if n<5:return 0
        peaks=[values[i] for i in range(1,n-1) if values[i]>values[i-1] and values[i]>=values[i+1]]
        if len(peaks)<min_peaks:return max(peaks) if peaks else 0
        x=list(range(len(peaks)));y=peaks
        slope=(y[-1]-y[0])/(x[-1]-x[0]) if x[-1]-x[0]!=0 else 0
        pred=max(0,y[-1]+slope)
        return pred

    def compute_average_remaining(self, series):
        data = series[:-1]
        if not data:
            return 0
        return sum(data) / len(data)

    def extract_last_means(self, series):
        data = series[:-1]
        last5 = sum(data[-5:]) / 5 if len(data) >= 5 else 0
        last10 = sum(data[-10:]) / 10 if len(data) >= 10 else 0
        last15 = sum(data[-15:]) / 15 if len(data) >= 15 else 0
        last20 = sum(data[-20:]) / 20 if len(data) >= 20 else 0
        return last10, last15, last20
    
    def display_median_extrema_series_means(self,series,log_widget,mode="min"):
        data=series[:-1]
        if not data:return
        sizes=[(7,[15,27,51]),(13,[27,51,99]),(25,[51,99])]
        configs=[(m,z) for m,zs in sizes for z in zs]
        min_zone=min(z for _,z in configs)
        if len(data)<min_zone:return

        if mode=="min":
            self.write_log(log_widget,"\nMedian minima 'N values mean' in 'N lasts':\n");func=min
        elif mode=="max":
            self.write_log(log_widget,"\nMedian maxima 'N values mean' in 'N lasts':\n");func=max
        else:raise ValueError("mode must be 'min' or 'max'")

        for mean_size,zone_size in configs:
            if len(data)<zone_size:
                self.write_log(log_widget,f"{mean_size} in {zone_size} : N/A\n");continue
            zone=data[-zone_size:]
            val=func(zone);idx=zone.index(val)
            center=zone_size//2
            if idx!=center:
                self.write_log(log_widget,f"{mean_size} in {zone_size} : N/A\n");continue
            half=mean_size//2
            start=max(0,idx-half);end=start+mean_size
            if end>len(zone):end=len(zone);start=end-mean_size
            subset=zone[start:end]
            if len(subset)==mean_size:
                avg=sum(subset)/mean_size
                self.write_log(log_widget,f"{mean_size} in {zone_size} : {avg:.2f}\n")
            else:self.write_log(log_widget,f"{mean_size} in {zone_size} : N/A\n")

    def display_median_extrema_means(self, series, log_widget, mode="min"):
        data = series[:-1]
        if not data:
            return

        configs = [(3,15),(7,15),(7,27),(7,51),(13,27),(13,51),(13,99),(25,51),(25,99)]

        if mode == "min":
            self.write_log(log_widget, "\n📊 Median minima 'N values mean' in 'N lasts':\n")
            func = min
        elif mode == "max":
            self.write_log(log_widget, "\n📊 Median maxima 'N values mean' in 'N lasts':\n")
            func = max
        else:
            raise ValueError("mode must be 'min' or 'max'")

        for mean_size, zone_size in configs:
            if len(data) < zone_size:
                self.write_log(log_widget, f"{mean_size} in {zone_size} : N/A\n")
                continue

            zone = data[-zone_size:]
            val = func(zone)
            idx = zone.index(val)

            half = mean_size // 2
            start = max(0, idx - half)
            end = start + mean_size

            if end > len(zone):
                end = len(zone)
                start = end - mean_size

            subset = zone[start:end]

            if len(subset) == mean_size:
                avg = sum(subset) / mean_size
                self.write_log(log_widget, f"{mean_size} in {zone_size} : {avg:.2f}\n")
            else:
                self.write_log(log_widget, f"{mean_size} in {zone_size} : N/A\n")

    def predict_score_from_seriesB_V1(self, series, opponent_series=None):
        if not series or len(series) < 5:
            return "0.00", {"reason": "series too short"}

        RECENT = 10

        recent = self._window_mean(series[-RECENT-1:-1], RECENT)
        global_75 = sorted(series)[int(0.75 * (len(series) - 1))]

        base = 0.6 * recent + 0.4 * global_75

        std = statistics.pstdev(series) if len(series) > 1 else 0
        base *= 0.95 if std < 0.35 else 1.05 if std > 1.5 else 1

        if opponent_series and len(opponent_series) >= 5:
            opp = self._window_mean(opponent_series[-RECENT-1:-1], RECENT)

            # interaction forte ici
            base *= 1 + (opp - recent) * 0.1

        final = max(0, min(round(base, 2), 400))

        return f"{final:.2f}"#, {"base": base, "std": std}

    def predict_score_from_seriesC_V1(self, series, opponent_series=None):
        if not series or len(series) < 5:
            return "0.00", {"reason": "series too short"}

        RECENT, MAX_G = 10, 400

        recent = self._window_mean(series[-RECENT-1:-1], RECENT)
        g75 = sorted(series)[int(0.75 * (len(series) - 1))]

        base = 0.6 * recent + 0.4 * g75

        std = statistics.pstdev(series)
        base *= 0.95 if std < 0.35 else 1.05 if std > 1.5 else 1

        opp_mean = None

        if opponent_series and len(opponent_series) >= 5:
            opp_mean = self._window_mean(opponent_series[-RECENT-1:-1], RECENT)

            dominance = recent - opp_mean
            base += dominance * 0.8

        final = max(0, min(round(base, 2), MAX_G))

        return f"{final:.2f}"#, {"base": base, "opp": opp_mean}

    def predict_score_from_seriesD_V1(self, series):
        if isinstance(series, list):
            series = series[:-1]
            n = len(series)

            if not n:
                return "0.00"

            mean = sum(series) / n
            rebond = sum(series[i+1]-series[i] for i in range(n-1)) / max(1, n-1)
            std = (sum((x - mean)**2 for x in series)/n) ** 0.5

        else:
            mean = team_stats.get("mean", 0)
            rebond = team_stats.get("rebond_lineaire", 0)
            std = team_stats.get("std", 0)

        score = mean + 0.5 * rebond - 0.2 * std

        return f"{max(0, score):.2f}"

    def predict_score_from_seriesE_V1(self, series):
        if not series or len(series) < 5:
            return "0.00"

        series = series[:-1]

        r5 = sum(series[-5:]) / 5
        r10 = sum(series[-10:]) / min(10, len(series))
        mean = sum(series) / len(series)

        rebond = r5 - r10

        score = 0.6*r5 + 0.3*r10 + 0.1*mean + 5*rebond
        
        score *= 1 + min(max(rebond, -0.2), 0.2)

        return f"{max(0, min(score, 1000)):.2f}"

    def predict_pattern_last40(self, series, pattern_size_range=(2, 6), recent_window=40):
        if len(series) <= pattern_size_range[0]:
            return {}

        all_weights = defaultdict(float)

        for pattern_size in range(pattern_size_range[0], pattern_size_range[1] + 1):
            if len(series) <= pattern_size:
                continue

            pattern = tuple(series[-pattern_size:])
            next_values = []

            for i in range(len(series) - pattern_size):
                if tuple(series[i:i+pattern_size]) == pattern:
                    next_values.append(series[i+pattern_size])

            if not next_values:
                continue

            base_counts = Counter(next_values)

            recent_series = series[-recent_window:]
            recent_counts = Counter(recent_series)

            for value, count in base_counts.items():
                base_weight = 1 / count
                recent_freq = recent_counts[value] / len(recent_series)

                penalty = (1 + recent_freq)

                adjusted_weight = base_weight / penalty

                all_weights[value] += adjusted_weight

        if not all_weights:
            return {}

        total = sum(all_weights.values())
        probabilities = {
            value: weight / total
            for value, weight in all_weights.items()
        }

        return probabilities

    def _clean_series(self, series):
        if not series or len(series) < 2:
            return None
        return np.array(series[:-1], dtype=float)

    def _stats_basic(self, series):
        s = self._clean_series(series)
        if s is None:
            return None

        mean = np.mean(s)
        recent5 = np.mean(s[-5:]) if len(s) >= 5 else mean
        recent3 = np.mean(s[-3:]) if len(s) >= 3 else mean
        std = np.std(s)
        momentum = recent5 - mean

        return {
            "mean": mean,
            "recent5": recent5,
            "recent3": recent3,
            "std": std,
            "momentum": momentum
        }

    def _interaction_block(self, A, C):

        dominance = 0.6 * (A["mean"] - C["mean"]) + 0.4 * (A["recent5"] - C["recent5"])

        instability_A = 0.3 * C["std"]
        instability_C = 0.3 * A["std"]

        momentum_A = A["momentum"] - 0.5 * C["momentum"]
        momentum_C = C["momentum"] - 0.5 * A["momentum"]

        offensive_pressure_A = A["mean"] * (1 + max(0, -C["std"]))
        offensive_pressure_C = C["mean"] * (1 + max(0, -A["std"]))

        return {
            "dominance": dominance,
            "instability_A": instability_A,
            "instability_C": instability_C,
            "momentum_A": momentum_A,
            "momentum_C": momentum_C,
            "off_A": offensive_pressure_A,
            "off_C": offensive_pressure_C
        }

    def analyze_zero_pattern(self,series):
        zero_sequences=[]; i=0; n=len(series)
        while i<n:
            if series[i]==0:
                start=i
                while i<n and series[i]==0: i+=1
                length=i-start
                if length>2 and i<n: zero_sequences.append(series[i])
            else: i+=1
        zero_signal=None; last_value=None
        if len(zero_sequences)>=3:
            bbl,bl,last_value=zero_sequences[-3],zero_sequences[-2],zero_sequences[-1]
            if bl>bbl: zero_signal="FEW"
            elif bl<bbl: zero_signal="LOT"
        return zero_sequences,zero_signal,last_value

    def fully_static_method_with_patterns(self,seriesA,seriesC):

        def detect_target_motif(series):
            data=series[:-1];n=len(data)
            for L in range(2,n//2+1):
                last=data[-L:]
                for i in range(n-L-1,-1,-1):
                    if data[i:i+L]==last:return last
            return None

        def is_proportional(m1,m2):
            if len(m1)!=len(m2) or m1[0]==0:return False
            r=m2[0]/m1[0]
            for a,b in zip(m1,m2):
                if a==0 or abs((b/a)-r)>1e-9:return False
            return True

        def classify(series,motif):
            data=series[:-1];L=len(motif)
            e,p_sup,p_inf=[],[],[];i=0
            while i<=len(data)-L:
                block=data[i:i+L]
                if block==motif:e.append(i);i+=L;continue
                elif is_proportional(motif,block):
                    r=block[0]/motif[0]
                    if r>1:p_sup.append(i)
                    elif 0<r<1:p_inf.append(i)
                    i+=L;continue
                i+=1
            m=sorted(set(e+p_sup+p_inf))
            return e,p_sup,p_inf,m

        def build_next_vals(series,motif,idxs):
            L=len(motif);vals=[]
            for idx in idxs:
                if idx+L<len(series):vals.append(series[idx+L])
            return vals

        motifA=detect_target_motif(seriesA)
        if motifA:
            eA,pA,piA,mA=classify(seriesA,motifA)
            nextA=build_next_vals(seriesA,motifA,mA)
            rebA=self.linear_rebound(nextA)
            peakA=self.peak_envelope(nextA)
            resultA=(motifA,(eA,pA,piA,mA,nextA,rebA,peakA))
        else:resultA=(None,([],[],[],[],[],[],[]))

        motifC=detect_target_motif(seriesC)
        if motifC:
            eC,pC,piC,mC=classify(seriesC,motifC)
            nextC=build_next_vals(seriesC,motifC,mC)
            rebC=self.linear_rebound(nextC)
            peakC=self.peak_envelope(nextC)
            resultC=(motifC,(eC,pC,piC,mC,nextC,rebC,peakC))
        else:resultC=(None,([],[],[],[],[],[],[]))

        motifA,dataA=resultA
        eA,pA,piA,mA,nextA,rebA,peakA=dataA

        motifC,dataC=resultC
        eC,pC,piC,mC,nextC,rebC,peakC=dataC

        return resultA,resultC

    def regime_shift_detector(self, series, window=10):
        data = np.array(series[:-1], dtype=float)
        if len(data) < 2*window:
            return {"shift": False}

        old = data[-2*window:-window]
        recent = data[-window:]

        mean_diff = np.mean(recent) - np.mean(old)
        std_diff = np.std(recent) - np.std(old)

        score = abs(mean_diff) + abs(std_diff)

        shift = score > np.std(data)

        prediction = np.mean(recent) + mean_diff

        return {
            "prediction": float(prediction),
            "shift": bool(shift),
            "score": float(score)
        }

    def local_attractor_prediction(self, series, dim=3):
        data = np.array(series[:-1], dtype=float)
        n = len(data)

        if n < dim + 5:
            return {"prediction": 0}

        # Reconstruction espace
        vectors = np.array([data[i:i+dim] for i in range(n-dim)])
        target = np.array([data[i+dim] for i in range(n-dim)])

        current = data[-dim:]

        # Distance
        dists = np.linalg.norm(vectors - current, axis=1)

        # K plus proches
        k = min(5, len(dists))
        idx = np.argsort(dists)[:k]

        prediction = np.mean(target[idx])

        return {
            "prediction": float(prediction),
            "neighbors": int(k),
            "avg_distance": float(np.mean(dists[idx]))
        }

    def markov_weighted_prediction(self, series, n_states=5):
        data = np.array(series[:-1], dtype=float)
        if len(data) < 10:
            return {"prediction": 0}

        # Discrétisation en états
        bins = np.linspace(np.min(data), np.max(data), n_states+1)
        states = np.digitize(data, bins) - 1
        states = np.clip(states, 0, n_states - 1)  # Assurer que les états sont dans les bornes

        # Matrice de transition
        trans = np.zeros((n_states, n_states))
        for i in range(len(states)-1):
            trans[states[i], states[i+1]] += 1

        # Normalisation
        trans = np.divide(trans, trans.sum(axis=1, keepdims=True), where=trans.sum(axis=1, keepdims=True) != 0)

        current_state = states[-1]
        probs = trans[current_state]

        # Calcul de la valeur projetée
        centers = (bins[:-1] + bins[1:]) / 2
        prediction = np.sum(probs * centers)

        return {
            "prediction": float(prediction),
            "state": int(current_state),
            "confidence": float(np.max(probs))
        }

    def _normalize(self, seq):
        seq = np.array(seq)
        std = np.std(seq)
        if std == 0:
            return seq
        return (seq - np.mean(seq)) / std

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def predict_proba(self, data, window=5):
        data = np.array(data)

        # sécurité
        if len(data) < window * 2:
            return None

        # 🔥 spectre dynamique basé UNIQUEMENT sur les données
        spectrum = np.unique(data)

        # pattern récent
        pattern = data[-window:]
        normalize = getattr(self, "normalize", True)

        if normalize:
            pattern = self._normalize(pattern)

        n = len(data)
        freq = Counter(data)

        # fréquence historique (%)
        freq_old = {
            v: (freq.get(v, 0) / n) * 100 for v in spectrum
        }

        results = []
        raw_scores = []

        # valeur la plus rare (inclut les faibles occurrences)
        min_freq = min(freq_old.values()) if freq_old else 0

        max_bar = getattr(self, "max_bar", 100)
        rare_bonus = getattr(self, "rare_bonus", 0)

        for i in range(n - window - 1):
            candidate = data[i:i + window]

            if normalize:
                candidate = self._normalize(candidate)

            dist = self._distance(pattern, candidate)
            value = data[i + window]

            f = freq_old.get(value, 0)

            # 🔥 boost offensif
            boost = max_bar - f

            # score base
            base_score = dist * (1 + boost / 100)

            # bonus rareté
            rb = rare_bonus if f == min_freq else 0

            final_score = max(base_score - rb, 0)

            results.append({
                "value": value,
                "freq_%": f,
                "boost_%": boost,
                "rare_bonus": rb,
                "score": final_score
            })

            raw_scores.append(final_score)

        raw_scores = np.array(raw_scores)

        # éviter crash division par zéro
        inv = 1 / (raw_scores + 1e-9)
        probs = inv / np.sum(inv) if np.sum(inv) != 0 else np.zeros_like(inv)

        for i in range(len(results)):
            results[i]["probability_%"] = float(f"{probs[i] * 100:.2f}")

        return results

################################
#Créer une 2e version pour un 2e résultat dans "analyze_blocks_and_predict"
#def prediction_scores(self, series, next_value=None, global_freq=None):
    def prediction_scores(self, series):
        if len(series) < 2:
            return None

        base = series[:-1]
        dernier = base[-1]

        suivants = [
            base[i+1]
            for i in range(len(base) - 1)
            if base[i] == dernier
        ]

        if not suivants:
            return None

        total = len(suivants)
        freq = Counter(suivants)

        pourcentages = {k: v / total for k, v in freq.items()}

        anciennete = {}
        for score in freq:
            positions = [
                i for i, x in enumerate(base[:-1])
                if x == dernier and base[i+1] == score
            ]
            anciennete[score] = 1 / (1 + max(positions)) if positions else 0

        scores_uniques = list(freq.keys())
        sorted_by_freq = sorted(scores_uniques, key=lambda x: freq[x])
        bareme = {score: len(scores_uniques) - i for i, score in enumerate(sorted_by_freq)}

        proba = {
            score: pourcentages[score] * (1 + anciennete[score]) * bareme[score]
            for score in scores_uniques
        }

        somme = sum(proba.values())
        proba = {k: v / somme for k, v in proba.items()}

        resultat = max(proba, key=proba.get)

        ajustement = bareme[resultat] / len(scores_uniques)
        resultat_ajuste = (
            resultat * (1 + ajustement)
            if freq[resultat] == min(freq.values())
            else resultat * (1 - ajustement)
        )

        return {
            "keys": sorted(scores_uniques),
            "freq": freq,
            "pourcentages": pourcentages,
            "anciennete": anciennete,
            "bareme": bareme,
            "proba": proba,
            "prediction": resultat,
            "adjusted": resultat_ajuste
        }

    def _extract_patterns_with_positions(self, series, size):
        patterns = []
        for i in range(len(series) - size + 1):
            pattern = tuple(series[i:i + size])
            patterns.append((pattern, i))
        return patterns
    
    def _pattern_metrics(self, series, size, window):
        recent = series[-window:] if window is not None else series

        patterns = self._extract_patterns_with_positions(recent, size)

        freq = defaultdict(int)
        positions = defaultdict(list)

        for p, pos in patterns:
            freq[p] += 1
            positions[p].append(pos)

        total = len(patterns)

        metrics = {}
        for p in freq:
            # --- fréquence brute ---
            ratio = freq[p] / total if total > 0 else 0

            # --- ancienneté pattern ---
            avg_pos = np.mean(positions[p])
            recency = avg_pos / total if total > 0 else 0

            # --- logique prediction_scores adaptée ---
            pourcentage = ratio

            anciennete = 1 / (1 + max(positions[p])) if positions[p] else 0

            # bareme dynamique basé sur fréquence
            sorted_patterns = sorted(freq.keys(), key=lambda x: freq[x])
            bareme_map = {
                pat: len(sorted_patterns) - i
                for i, pat in enumerate(sorted_patterns)
            }
            bareme = bareme_map[p]

            # proba enrichie
            proba = pourcentage * (1 + anciennete) * bareme

            metrics[p] = {
                "freq": freq[p],
                "ratio": ratio,
                "pourcentage": pourcentage,
                "anciennete": anciennete,
                "recency": recency,
                "bareme": bareme,
                "proba": proba,
                "positions": positions[p],
            }

        # --- normalisation globale des proba ---
        somme = sum(m["proba"] for m in metrics.values())
        if somme > 0:
            for p in metrics:
                metrics[p]["proba"] /= somme

        return metrics

    def _dynamic_adjustment(self, series, candidate, size, window):
        if len(series) < size:
            return 0

        metrics = self._pattern_metrics(series, size, window)

        context = tuple(series[-(size - 1):])
        pattern = context + (candidate,)

        data = metrics.get(pattern, None)
        if data is None:
            return 0

        ratio = data["ratio"]
        recency = data["recency"]
        proba = data["proba"]
        bareme = data["bareme"]

        N = len(series)
        data_weight = min(1.0, np.log1p(N) / 5)

        rarity = 1 - ratio
        age_factor = 1 - recency

        # score enrichi
        score = (
            (rarity * age_factor - ratio * recency)
            + proba * bareme
        ) * data_weight

        return score

################ 2

    def patterns_with_positions_predict(self, series):
        series = np.array(series)

        min_size = getattr(self, "min_size", 2)
        max_size = getattr(self, "max_size", 6)
        window = getattr(self, "window", 40)

        values, counts = np.unique(series, return_counts=True)
        base_probs = counts / counts.sum()

        adjusted_probs = []

        # 🔹 récupération des scores complémentaires
        extra_scores = self.prediction_scores(series)

        for v, base_p in zip(values, base_probs):
            total_score = 0
            total_weight = 0

            for size in range(min_size, max_size + 1):
                weight = size ** 1.5
                score = self._dynamic_adjustment_predict(series, v, size, window)

                total_score += weight * score
                total_weight += weight

            if total_weight > 0:
                total_score /= total_weight

            # 🔹 ajout influence prediction_scores
            bonus = 0
            if extra_scores and v in extra_scores["proba"]:
                bonus = extra_scores["proba"][v]

            new_p = max(base_p + total_score + 0.5 * bonus, 0.001)
            adjusted_probs.append(new_p)

        adjusted_probs = np.array(adjusted_probs)
        adjusted_probs /= adjusted_probs.sum()

        prediction = values[np.argmax(adjusted_probs)]

        return {
            "prediction": int(prediction),

            # 🔹 distribution globale
            "distribution": {
                int(v): round(p, 3)
                for v, p in zip(values, adjusted_probs)
            },

            # 🔹 ajout des métriques détaillées
            "scores_details": extra_scores
        }

    def _dynamic_adjustment_predict(self, series, candidate, size, window):
        if len(series) < size:
            return 0

        metrics = self._pattern_metrics(series, size, window)

        context = tuple(series[-(size - 1):])
        pattern = context + (candidate,)

        data = metrics.get(pattern, (0, 0))

        # sécurité universelle
        if isinstance(data, (list, tuple)):
            ratio = data[0] if len(data) > 0 else 0
            recency = data[1] if len(data) > 1 else 0
        elif isinstance(data, dict):
            ratio = data.get("ratio", 0)
            recency = data.get("recency", 0)
        else:
            ratio, recency = 0, 0

        N = len(series)
        data_weight = min(1.0, np.log1p(N) / 5)

        rarity = 1 - ratio
        age_factor = 1 - recency

        score = (rarity * age_factor - ratio * recency) * data_weight

        return score

#################### 3

    def prediction_scores_local(self, block):
        if len(block) < 2:
            return None

        base = block[:-1]
        dernier = base[-1]

        suivants = [
            base[i+1]
            for i in range(len(base) - 1)
            if base[i] == dernier
        ]

        if not suivants:
            return None

        total = len(suivants)
        freq = Counter(suivants)

        pourcentages = {k: v / total for k, v in freq.items()}

        scores_uniques = list(freq.keys())
        sorted_by_freq = sorted(scores_uniques, key=lambda x: freq[x])
        bareme = {s: len(scores_uniques) - i for i, s in enumerate(sorted_by_freq)}

        proba = {
            s: pourcentages[s] * bareme[s]
            for s in scores_uniques
        }

        s = sum(proba.values())
        proba = {k: v / s for k, v in proba.items()}

        pred = max(proba, key=proba.get)

        return {
            "prediction": pred,
            "proba": proba
        }

    def prediction_scores_global(self, block, global_freq):
        if len(block) < 2:
            return None

        base = block[:-1]
        dernier = base[-1]

        suivants = [
            base[i+1]
            for i in range(len(base) - 1)
            if base[i] == dernier
        ]

        if not suivants:
            return None

        total = len(suivants)
        freq = Counter(suivants)

        pourcentages = {k: v / total for k, v in freq.items()}

        # 🔥 injection global
        global_boost = {
            k: global_freq.get(k, 0)
            for k in freq.keys()
        }

        scores_uniques = list(freq.keys())

        proba = {
            s: pourcentages[s] * (1 + global_boost.get(s, 0))
            for s in scores_uniques
        }

        s = sum(proba.values())
        proba = {k: v / s for k, v in proba.items()}

        pred = max(proba, key=proba.get)

        return {
            "prediction": pred,
            "proba": proba
        }

    def analyze_blocks_and_predict(self, series):
        if len(series) < 4:
            return {"error": "Series too short"}

        data = series[:-1]

        # -------- 1. Fréquences globales --------
        counts = Counter(data)
        total = len(data)
        freq = {k: v / total for k, v in counts.items()}

        # -------- 2. Détection extrema --------
        extrema = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                extrema.append((i, "max"))
            elif data[i] < data[i-1] and data[i] < data[i+1]:
                extrema.append((i, "min"))

        blocks = []

        # -------- 3. CAS NORMAL --------
        if len(extrema) >= 2:
            for i in range(len(extrema) - 1):
                i1, _ = extrema[i]
                i2, _ = extrema[i+1]

                if i2 > i1:
                    block = data[i1:i2+1]

                    if i2 + 1 < len(series):
                        blocks.append({
                            "block": block,
                            "pred_next": series[i2 + 1]
                        })

        # -------- 4. FALLBACK 1 extrema --------
        elif len(extrema) == 1:
            idx, _ = extrema[0]

            if idx > 1:
                blocks.append({
                    "block": data[:idx+1],
                    "pred_next": series[idx + 1] if idx + 1 < len(series) else None
                })

            if idx + 2 < len(data):
                blocks.append({
                    "block": data[idx:],
                    "pred_next": series[-1]
                })

        # -------- 5. FALLBACK 0 extrema --------
        else:
            for i in range(2, len(data) - 1):
                blocks.append({
                    "block": data[:i],
                    "pred_next": series[i]
                })

        if len(blocks) < 2:
            return {"error": "Not enough blocks"}

        # -------- 6. matching --------
        def signature(block):
            c = Counter(block)
            size = len(block)
            return {k: v / size for k, v in c.items()}

        last_block = blocks[-1]["block"]
        last_sig = signature(last_block)

        best = None
        best_score = float("inf")

        for b in blocks[:-1]:
            sig = signature(b["block"])
            keys = set(sig) | set(last_sig)

            dist = sum(abs(sig.get(k, 0) - last_sig.get(k, 0)) for k in keys)

            if dist < best_score:
                best_score = dist
                best = b

        # -------- 7. SCORES LOCAL + GLOBAL --------
        local_scores = self.prediction_scores_local(best["block"])

        global_scores = self.prediction_scores_global(
            best["block"],
            global_freq=freq
        )

        return {
            "frequencies": freq,
            "last_block": last_block,
            "prediction": best["pred_next"] if best else None,
            "match_score": best_score,
            "num_blocks": len(blocks),

            "local": local_scores,
            "global": global_scores
        }

#################### 4

    def analyze_blocks_and_predict_2(self, series):
        if len(series) < 4:
            return {"error": "Series too short"}

        data = series[:-1]

        # -------- 1. Fréquences --------
        counts = Counter(data)
        total = len(data)
        freq = {k: v / total for k, v in counts.items()}

        # -------- 2. Détection extrema --------
        extrema = []

        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                extrema.append((i, "max"))
            elif data[i] < data[i - 1] and data[i] < data[i + 1]:
                extrema.append((i, "min"))

        blocks = []

        # -------- 3. CAS NORMAL (>=2 extrema) --------
        if len(extrema) >= 2:
            for i in range(len(extrema) - 1):
                i1, _ = extrema[i]
                i2, _ = extrema[i + 1]

                if i2 > i1:
                    block = data[i1:i2 + 1]

                    if i2 + 1 < len(series):
                        blocks.append({
                            "block": block,
                            "pred_next": series[i2 + 1]
                        })

        # -------- 4. FALLBACK 1 extrema --------
        elif len(extrema) == 1:
            idx, _ = extrema[0]

            if idx > 1:
                blocks.append({
                    "block": data[:idx + 1],
                    "pred_next": series[idx + 1] if idx + 1 < len(series) else None
                })

            if idx + 2 < len(data):
                blocks.append({
                    "block": data[idx:],
                    "pred_next": series[-1]
                })

        # -------- 5. FALLBACK 0 extrema --------
        else:
            for i in range(2, len(data) - 1):
                blocks.append({
                    "block": data[:i],
                    "pred_next": series[i]
                })

        if len(blocks) < 2:
            return {"error": "Not enough blocks"}

        # -------- 6. SIGNATURE --------
        def signature(block):
            c = Counter(block)
            size = len(block)
            return {k: v / size for k, v in c.items()}

        last_block = blocks[-1]["block"]
        last_sig = signature(last_block)

        best = None
        best_score = float("inf")

        for b in blocks[:-1]:
            sig = signature(b["block"])

            keys = set(sig) | set(last_sig)
            dist = sum(abs(sig.get(k, 0) - last_sig.get(k, 0)) for k in keys)

            if dist < best_score:
                best_score = dist
                best = b

        # -------- 7. SCORES PROBABILISTES --------
        scores = self.prediction_scores(series)

        result = {
            "frequencies": freq,
            "last_block": last_block,
            "prediction": best["pred_next"] if best else None,
            "match_score": best_score,
            "num_blocks": len(blocks),
        }

        # -------- 8. FUSION DES SCORES --------
        if scores:
            result.update({
                "freq": scores["freq"],
                "pourcentages": scores["pourcentages"],
                "anciennete": scores["anciennete"],
                "bareme": scores["bareme"],
                "proba": scores["proba"],
                "prediction_scores": scores["prediction"],
                "adjusted": scores["adjusted"],
                "scores_keys": scores["keys"],
            })

        return result

####################  5

    def linear_rebound_prediction2(self, series, window=6):
        if len(series) < 3:
            return None

        # 🔹 Partie 1 : modèle linéaire
        base_series = series[:-1]
        y1, y2 = base_series[-1], base_series[-2]

        base = 1.4 * y2 - 0.6 * y1
        recent = base_series[-window:] if len(base_series) >= window else base_series[:]
        bias = (max(recent) - min(recent)) * 0.4 if len(recent) > 1 else 0

        pred = max(0, base + bias)

        # 🔹 Partie 2 : probabilités
        score_data = self.prediction_scores(series)

        if score_data is None:
            return {
                "prediction": pred,
                "count": len(recent)
            }

        # 🔹 Fusion
        return {
            "prediction_linear": pred,
            "count": len(recent),

            "freq": score_data["freq"],
            "pourcentages": score_data["pourcentages"],
            "anciennete": score_data["anciennete"],
            "bareme": score_data["bareme"],
            "proba": score_data["proba"],
            "prediction_markov": score_data["prediction"],
            "adjusted": score_data["adjusted"]
        }

    def peak_envelope_prediction2(self, series, min_peaks=4):
        if len(series) < 5:
            return None

        base_series = series[:-1]
        n = len(base_series)

        px, py = [], []
        for i in range(1, n - 1):
            if base_series[i] > base_series[i - 1] and base_series[i] >= base_series[i + 1]:
                px.append(i)
                py.append(base_series[i])

        if len(px) < min_peaks:
            pred = max(py) if py else 0
        else:
            m = len(px)
            sx, sy = sum(px), sum(py)
            sxx = sum(x * x for x in px)
            sxy = sum(px[i] * py[i] for i in range(m))
            denom = m * sxx - sx * sx

            if denom == 0:
                pred = max(py)
            else:
                slope = (m * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / m
                pred = max(0, intercept + slope * n)

        # 🔹 Probabilités
        score_data = self.prediction_scores(series)

        if score_data is None:
            return {"prediction": pred, "count": len(px)}

        # 🔹 Fusion
        return {
            "prediction_peak": pred,
            "count": len(px),

            "freq": score_data["freq"],
            "pourcentages": score_data["pourcentages"],
            "anciennete": score_data["anciennete"],
            "bareme": score_data["bareme"],
            "proba": score_data["proba"],
            "prediction_markov": score_data["prediction"],
            "adjusted": score_data["adjusted"]
        }

    def linear_rebound2(self, values):
        if len(values) < 2:
            return None

        y1, y2 = values[-1], values[-2]

        # =========================
        # 📊 BASE MODEL (LINEAR)
        # =========================
        pred = 1.4 * y2 - 0.6 * y1
        recent = values[-min(6, len(values)):]

        if len(recent) > 1:
            pred += 0.4 * (max(recent) - min(recent))

        pred = max(0, pred)

        # =========================
        # 📊 MARKOV PIPELINE
        # =========================
        scores = self.prediction_scores(values)

        if not scores:
            return {
                "prediction_linear": pred,
                "freq": {},
                "pourcentages": {},
                "anciennete": {},
                "bareme": {},
                "proba": {},
                "prediction": pred,
                "adjusted": pred
            }

        # =========================
        # 🔥 FUSION RESULTATS
        # =========================
        return {
            "prediction_linear": pred,

            "freq": scores["freq"],
            "pourcentages": scores["pourcentages"],
            "anciennete": scores["anciennete"],
            "bareme": scores["bareme"],
            "proba": scores["proba"],
            "prediction": scores["prediction"],
            "adjusted": scores["adjusted"],
            "scores_keys": scores["keys"]
        }

    def peak_envelope2(self, values, min_peaks=4):
        n = len(values)
        if n < 5:
            return None

        # 🔹 Détection des pics
        peaks = [
            values[i]
            for i in range(1, n - 1)
            if values[i] > values[i - 1] and values[i] >= values[i + 1]
        ]

        # 🔹 Prédiction peak classique
        if len(peaks) < min_peaks:
            pred = max(peaks) if peaks else 0
        else:
            x = list(range(len(peaks)))
            y = peaks
            slope = (y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 0
            pred = max(0, y[-1] + slope)

        # 🔹 Ajout pipeline probabiliste
        score_data = self.prediction_scores(values)

        if score_data is None:
            return {
                "prediction_peak": pred,
                "count_peaks": len(peaks)
            }

        # 🔹 Fusion complète
        return {
            "prediction_peak": pred,
            "count_peaks": len(peaks),

            "freq": score_data["freq"],
            "pourcentages": score_data["pourcentages"],
            "anciennete": score_data["anciennete"],
            "bareme": score_data["bareme"],
            "proba": score_data["proba"],
            "prediction_markov": score_data["prediction"],
            "adjusted": score_data["adjusted"]
        }
    
#################### 6

    def prediction_scores_single(self, value):

        # discrétisation autour du score
        voisins = [
            round(value - 1, 2),
            round(value, 2),
            round(value + 1, 2)
        ]

        freq = {v: 1 for v in voisins}
        total = len(voisins)

        pourcentages = {k: 1/total for k in freq}

        # plus proche = plus récent
        anciennete = {
            voisins[1]: 1.0,
            voisins[0]: 0.5,
            voisins[2]: 0.5
        }

        sorted_scores = voisins
        bareme = {score: len(voisins) - i for i, score in enumerate(sorted_scores)}

        proba = {
            score: pourcentages[score] * (1 + anciennete[score]) * bareme[score]
            for score in voisins
        }

        somme = sum(proba.values())
        proba = {k: v / somme for k, v in proba.items()}

        resultat = max(proba, key=proba.get)

        ajustement = bareme[resultat] / len(voisins)

        resultat_ajuste = resultat * (1 + ajustement)

        return {
            "freq": freq,
            "pourcentages": pourcentages,
            "anciennete": anciennete,
            "bareme": bareme,
            "proba": proba,
            "prediction": resultat,
            "adjusted": resultat_ajuste
        }

    def predict_score_from_seriesA(self, seriesA, seriesC):

        A = self._stats_basic(seriesA)
        C = self._stats_basic(seriesC)

        if A is None or C is None:
            return "0.00", "0.00"

        inter = self._interaction_block(A, C)

        def base(x):
            return x["mean"] + 0.6*x["recent5"] + 0.2*x["recent3"]

        raw_A = base(A)
        raw_C = base(C)

        scoreA = (
            raw_A
            - A["std"] + 0.8*C["std"]
            + inter["dominance"]
            + inter["instability_A"]
            + 0.4*inter["momentum_A"]
            + 0.35*inter["off_A"]
        )

        scoreC = (
            raw_C
            - C["std"] + 0.8*A["std"]
            - inter["dominance"]
            + inter["instability_C"]
            + 0.4*inter["momentum_C"]
            + 0.35*inter["off_C"]
        )

        return {
            "A": self.prediction_scores_single(scoreA),
            "C": self.prediction_scores_single(scoreC)
        }

    def predict_score_from_seriesB(self, series, opponent_series=None):

        if not series or len(series) < 5:
            return "0.00", {"reason": "series too short"}

        s = np.array(series[:-1], dtype=float)

        RECENT = 10
        recent = np.mean(s[-RECENT:]) if len(s) >= RECENT else np.mean(s)

        global_75 = np.percentile(s, 75)

        base = 0.6 * recent + 0.4 * global_75

        std = np.std(s)
        base *= 0.95 if std < 0.35 else 1.05 if std > 1.5 else 1

        opp_effect = 0

        if opponent_series and len(opponent_series) >= 5:
            o = np.array(opponent_series[:-1], dtype=float)
            opp_recent = np.mean(o[-RECENT:]) if len(o) >= RECENT else np.mean(o)
            opp_effect = (opp_recent - recent) * 0.1

        final = base * (1 + opp_effect)

        return self.prediction_scores_single(max(0, round(final, 2)))

    def predict_score_from_seriesC(self, series, opponent_series=None):

        if not series or len(series) < 5:
            return "0.00", {"reason": "series too short"}

        s = np.array(series[:-1], dtype=float)

        RECENT = 10

        recent = np.mean(s[-RECENT:]) if len(s) >= RECENT else np.mean(s)
        g75 = np.percentile(s, 75)

        base = 0.6 * recent + 0.4 * g75

        std = np.std(s)
        base *= 0.95 if std < 0.35 else 1.05 if std > 1.5 else 1

        opp_mean = None

        if opponent_series and len(opponent_series) >= 5:
            o = np.array(opponent_series[:-1], dtype=float)
            opp_mean = np.mean(o[-RECENT:]) if len(o) >= RECENT else np.mean(o)

            dominance = recent - opp_mean
            base += dominance * 0.8

        return self.prediction_scores_single(max(0, round(base, 2)))

    def predict_score_from_seriesD(self, series):

        if not series or len(series) < 2:
            return "0.00"

        s = np.array(series[:-1], dtype=float)
        n = len(s)

        mean = np.mean(s)

        rebond = np.mean(np.diff(s)) if n > 1 else 0.0
        std = np.std(s)

        score = mean + 0.5 * rebond - 0.2 * std

        return self.prediction_scores_single(max(0, round(score, 2)))

    def predict_score_from_seriesE(self, series):

        if not series or len(series) < 5:
            return "0.00"

        s = np.array(series[:-1], dtype=float)

        r5 = np.mean(s[-5:]) if len(s) >= 5 else np.mean(s)
        r10 = np.mean(s[-10:]) if len(s) >= 10 else np.mean(s)

        mean = np.mean(s)

        rebond = r5 - r10

        score = 0.6*r5 + 0.3*r10 + 0.1*mean + 5*rebond

        # amplification dynamique
        score *= 1 + np.clip(rebond, -0.2, 0.2)

        return self.prediction_scores_single(max(0, min(score, 1000)))

#################### 
    def _build_ui(self):
        top=ttk.Frame(self);top.pack(fill="x",padx=8,pady=6);top.columnconfigure(1,weight=1)
        ttk.Label(top,text="FOLDER",font=("TkDefaultFont",10,"bold")).grid(row=0,column=0,sticky="w")

        controls=ttk.Frame(self);controls.pack(fill="x",padx=8,pady=4)

        self.combo_folder=ttk.Combobox(controls,values=[],state="readonly",width=20)
        self.combo_folder.grid(row=0,column=0,padx=4)
        self.combo_folder.bind("<<ComboboxSelected>>",self.on_folder_selected)
        self.refresh_folders()

        self.combo_csv=ttk.Combobox(controls,width=40)
        self.combo_csv.grid(row=0,column=1,padx=4)
        self.combo_csv.bind("<KeyRelease>",self.filter_csv_list)

        self.combo_action=ttk.Combobox(controls,values=["LOAD","ARCHIVE"],state="readonly",width=15)
        self.combo_action.set("LOAD");self.combo_action.grid(row=0,column=2,padx=4)

        self.combo_scope=ttk.Combobox(controls,values=["ONE","ALL"],state="readonly",width=10)
        self.combo_scope.set("ALL");self.combo_scope.grid(row=0,column=3,padx=4)

        ttk.Button(controls,text="APPLY",command=self.on_apply_clicked).grid(row=0,column=4,padx=6)
        ttk.Button(controls,text="MOTIF<>NEXT",command=self.on_motif_next_clicked).grid(row=0,column=5,padx=6)
        ttk.Button(top,text="📊 BENCHMARK",command=self.run_benchmark).grid(row=0,column=5,padx=6)

        mid=ttk.Frame(self);mid.pack(fill="both",expand=True,padx=8,pady=8)
        mid.columnconfigure(0,weight=1);mid.columnconfigure(1,weight=1);mid.columnconfigure(2,weight=1)

        self._build_team_frame(mid,0,"A")
        self._build_team_frame(mid,1,"C")
        self._build_log_frame(mid,controls)

    def _build_team_frame(self,parent,col,team):
        frame=ttk.Frame(parent);frame.grid(row=0,column=col,sticky="nsew",padx=4)
        line=ttk.Frame(frame);line.pack(anchor="w",fill="x")

        entry=ttk.Entry(line,width=62);entry.pack(side="left")
        setattr(self,f"entry_team{team}",entry)

        entry_score=ttk.Entry(line,width=5);entry_score.pack(side="left",padx=(6,0))
        setattr(self,f"entry_team{team}_score",entry_score)
        entry_score.bind("<FocusOut>",lambda e,t=team:self.save_team_score_to_csv(t))

        c1=ttk.Frame(frame);c1.pack(fill="both",expand=True)
        log1=tk.Text(c1,height=20,width=60);log1.pack(side="left",fill="both",expand=True)
        sb1=ttk.Scrollbar(c1,orient="vertical",command=log1.yview);sb1.pack(side="right",fill="y")
        log1.config(yscrollcommand=sb1.set)
        setattr(self,f"log_team{team}",log1)

        c2=ttk.Frame(frame);c2.pack(fill="both",expand=True)
        log2=tk.Text(c2,height=30,width=60);log2.pack(side="left",fill="both",expand=True)
        sb2=ttk.Scrollbar(c2,orient="vertical",command=log2.yview);sb2.pack(side="right",fill="y")
        log2.config(yscrollcommand=sb2.set)
        setattr(self,f"log_team{team}_1",log2)

    def _build_log_frame(self, parent, controls):
        container = ttk.Frame(parent)
        container.grid(row=0, column=2, sticky="nsew", padx=(6,2))
        container.rowconfigure(4, weight=1) 
        container.columnconfigure(0, weight=1)

        btns = ttk.Frame(container)
        btns.grid(row=0, column=0, sticky="ew", pady=(0,4))
        ttk.Button(btns, text="HIDE", command=self.hide_selected_csv).pack(side="left", padx=4)
        ttk.Button(btns, text="ARCHIVE", command=self.archive_selected_csv).pack(side="left", padx=4)

        ttk.Button(container, text="CLEAR LOG",
            command=self.reset_csv_log).grid(row=1, column=0, sticky="w", pady=(0,4))

        nav = ttk.Frame(container)
        nav.grid(row=2, column=0, sticky="w", pady=(0,4))
        ttk.Button(nav, text="PREV.", command=self.prev_csv).pack(side="left", padx=2)
        ttk.Label(nav, text="<").pack(side="left", padx=(0,6))
        ttk.Label(nav, text=">").pack(side="left", padx=(6,0))
        ttk.Button(nav, text="NEXT.", command=self.next_csv).pack(side="left", padx=2)

        search_frame = ttk.Frame(container)
        search_frame.grid(row=3, column=0, sticky="w", pady=(0,4))

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var, width=25)
        self.search_entry.pack(side="left")

        self.search_var.trace_add("write", self.highlight_matches)

        self.log_csv = tk.Listbox(container, selectmode="extended", height=25)
        self.log_csv.grid(row=4, column=0, sticky="nsew")

        sb = ttk.Scrollbar(container, command=self.log_csv.yview)
        sb.grid(row=4, column=1, sticky="ns")
        self.log_csv.config(yscrollcommand=sb.set)

        self.log_csv.bind("<<ListboxSelect>>", self.on_csv_select)

        self.refresh_csv_log()

    def highlight_matches(self, *args):
        query = self.search_var.get().lower().strip()

        self.search_active = True

        self.log_csv.selection_clear(0, tk.END)

        if hasattr(self, "user_selection"):
            for i in self.user_selection:
                self.log_csv.selection_set(i)

        if query:
            for i in range(self.log_csv.size()):
                text = self.log_csv.get(i).lower()
                if query in text:
                    self.log_csv.selection_set(i)
                    self.log_csv.see(i)

        self.search_active = False

    def on_motif_next_clicked(self):
        new_val=simpledialog.askinteger(
            title="MOTIF<>NEXT",
            prompt="Modifier la distance admise entre corres_motif et corres_next :",
            initialvalue=self.motif_next_distance,minvalue=0,parent=self)
        if new_val is not None:self.motif_next_distance=new_val

    def add_csv_log(self,path:str):
        if path not in self.csv_log:
            self.csv_log.append(path);self.current_index=len(self.csv_log)-1
            self.refresh_csv_log()

    def split_csv_name_two_lines(self,filename:str):
        name=filename.replace(".csv","")
        if "_VS_" in name:
            l,r=name.split("_VS_",1);return l+"_VS_",r
        mid=len(name)//2;return name[:mid],name[mid:]

    def refresh_csv_log(self):
        self.log_csv.delete(0,tk.END);self.listbox_to_csv_index.clear()
        row=0
        for i,path in enumerate(self.csv_log):
            if path in self.hidden_csv:continue
            base=os.path.basename(path)
            l1,l2=self.split_csv_name_two_lines(base)

            self.log_csv.insert(tk.END,l1);self.listbox_to_csv_index[row]=i;row+=1
            self.log_csv.insert(tk.END,l2);self.listbox_to_csv_index[row]=i;row+=1
            self.log_csv.insert(tk.END,"────────────");row+=1

    def reset_csv_log(self):
        self.csv_log.clear();self.current_index=-1;self.refresh_csv_log()

    def on_csv_select(self,event=None):
        sel=self.log_csv.curselection()
        if not sel:return
        line=sel[-1]
        if line not in self.listbox_to_csv_index:
            self.log_csv.selection_clear(0,tk.END);return
        idx=self.listbox_to_csv_index[line]
        self.log_csv.selection_clear(0,tk.END)
        for lb,i in self.listbox_to_csv_index.items():
            if i==idx:self.log_csv.selection_set(lb)

        self.current_index=idx
        self.loaded_csv_path=self.csv_log[idx]
        self.load_csv();self.run_prediction()

    def hide_selected_csv(self):
        for idx in self.log_csv.curselection():
            if idx in self.listbox_to_csv_index:
                self.hidden_csv.add(self.csv_log[self.listbox_to_csv_index[idx]])
        self.refresh_csv_log()

    def archive_selected_csv(self):
        display=self.combo_folder.get()
        if not display:
            messagebox.showwarning("Avertissement","Aucun dossier sélectionné.");return
        folder=display.rsplit(" ",1)[0]
        archive=os.path.join(EXPORT_DIR,folder)
        os.makedirs(archive,exist_ok=True)

        real=set()
        for idx in self.log_csv.curselection():
            if idx in self.listbox_to_csv_index:
                real.add(self.listbox_to_csv_index[idx])

        for i in sorted(real,reverse=True):
            if 0<=i<len(self.csv_log):
                path=self.csv_log.pop(i)
                if os.path.exists(path):
                    shutil.move(path,os.path.join(archive,os.path.basename(path)))
        self.refresh_csv_log()

    def refresh_folders(self):
        vals=[]
        for folder in list_subfolders(EXPORT_DIR):
            path=os.path.join(EXPORT_DIR,folder)
            vals.append(f"{folder} {count_csv_files(path)}")
        self.combo_folder["values"]=vals
        self.combo_folder.set(vals[0] if vals else "")

    def on_folder_selected(self,event=None):
        display=self.combo_folder.get()
        if not display:return
        folder=display.rsplit(" ",1)[0]
        path=os.path.join(EXPORT_DIR,folder)
        self.csv_files=list_csv_files(path)
        self.combo_csv["values"]=self.csv_files
        self.combo_csv.set("")

    def filter_csv_list(self,event=None):
        typed=self.combo_csv.get().lower()
        self.combo_csv["values"]=[f for f in self.csv_files if typed in f.lower()]

    def apply_csv_action(self):
        display=self.combo_folder.get()
        scope=self.combo_scope.get()
        action=self.combo_action.get()

        if action=="LOAD" and scope=="ALL":pass
        if not display:return

        folder=display.rsplit(" ",1)[0]
        folder_path=os.path.join(EXPORT_DIR,folder)
        files=list_csv_files(folder_path)
        if not files:return

        targets=files if scope=="ALL" else [self.combo_csv.get()]
        for file in targets:
            if not file:continue
            path=os.path.join(folder_path,file)
            try:
                if action=="LOAD":
                    self.loaded_csv_path=path
                    self.add_csv_log(path)
                    self.load_csv();self.run_prediction()
                elif action=="ARCHIVE":
                    arch=os.path.join(EXPORT_DIR,"Archive")
                    os.makedirs(arch,exist_ok=True)
                    shutil.move(path,os.path.join(arch,file))
            except Exception:
                err=traceback.format_exc()
                messagebox.showerror("Erreur",err)
                save_error_to_csv(err)

        self.csv_files=list_csv_files(folder_path)
        self.combo_csv["values"]=self.csv_files
        self.combo_csv.set("")
        self.refresh_folders()

        for v in self.combo_folder["values"]:
            if v.startswith(folder+" "):
                self.combo_folder.set(v);break

    def save_team_score_to_csv(self,team:str):
        if not self.loaded_csv_path:return
        entry=getattr(self,f"entry_team{team}_score",None)
        if entry is None:return
        value=entry.get().strip()
        if value=="":return
        if not is_number(value):
            messagebox.showwarning("Valeur invalide",f"Score équipe {team} invalide : {value}")
            return

        col=4 if team=="A" else 6
        try:
            df=pd.read_csv(self.loaded_csv_path,sep=None,engine="python",header=None,encoding="utf-8-sig")
            while df.shape[1]<=col:df[df.shape[1]]=""
            df.iat[0,col]=int(value)
            df.to_csv(self.loaded_csv_path,index=False,header=False,encoding="utf-8-sig")
        except Exception:pass

    def load_csv(self):
        df=pd.read_csv(self.loaded_csv_path,sep=None,engine="python",header=None,encoding='utf-8-sig')
        self.teamA_scores=[str(v).strip() for v in df.iloc[:,0].tolist()]
        self.teamC_scores=[str(v).strip() for v in df.iloc[:,2].tolist()]

        teamA=next((str(v).strip() for v in reversed(self.teamA_scores) if v.strip()),"Equipe A")
        teamC=next((str(v).strip() for v in reversed(self.teamC_scores) if v.strip()),"Equipe C")

        self.entry_teamA.delete(0,tk.END);self.entry_teamA.insert(0,teamA)
        self.entry_teamC.delete(0,tk.END);self.entry_teamC.insert(0,teamC)

        try:
            self.entry_teamA_score.delete(0,tk.END)
            if df.shape[1]>4 and pd.notna(df.iat[0,4]):
                self.entry_teamA_score.insert(0,str(int(df.iat[0,4])))

            self.entry_teamC_score.delete(0,tk.END)
            if df.shape[1]>6 and pd.notna(df.iat[0,6]):
                self.entry_teamC_score.insert(0,str(int(df.iat[0,6])))
        except Exception:pass
        self.csv_loaded=True

    def on_apply_clicked(self):
        self.apply_csv_action()
        if self.csv_loaded:self.run_prediction()

    def write_log(self,target,text):
        if hasattr(target,"insert"):target.insert(tk.END,text)
        else:target(text)

    def _write(self, output, text):
        if callable(output):
            output(text)
        else:
            self.write_log(output, text)

    def run_benchmark(self):
        if not self.csv_loaded:
            return

        filepath = "Benchmark_F4.txt.py"

        seriesA = read_numeric_after_marker(self.teamA_scores, "A")
        seriesC = read_numeric_after_marker(self.teamC_scores, "C")

        if not seriesA or not seriesC:
            return

        resultA, resultC = self.fully_static_method_with_patterns(seriesA, seriesC)

        with open(filepath, "a", encoding="utf-8") as f:

            def log(x=""):
                f.write(str(x) + "\n")

            def write(log_obj, txt):
                log(txt.rstrip("\n"))

            # ========================= RESET HEADER
            log("\n" + "=" * 60)
            log("BENCHMARK MIRROR RUN_PREDICTION")
            log("=" * 60 + "\n")

            # ========================= CLEAR LOGIC REMOVED (UI ONLY)

            # ========================= PROBA PATTERN
            def write_predict_proba(series):
                write(log, "📊 PROBABILITY PATTERN (DISTANCE MODEL)\n")

                results = self.predict_proba(series, window=5)

                if not results:
                    write(log, "Not enough data\n")
                    return

                seen = set()
                filtered = []

                for r in sorted(results, key=lambda x: x["probability_%"], reverse=True):
                    if r["value"] in seen:
                        continue
                    seen.add(r["value"])
                    filtered.append(r)

                top = filtered[:5]

                headers = ["VALUE", "PROBA", "FREQ", "BOOST", "SCORE"]

                values = [r["value"] for r in top]
                probs  = [f"{r['probability_%']:.2f}" for r in top]
                freq   = [f"{r['freq_%']:.2f}" for r in top]
                boost  = [f"{r['boost_%']:.2f}" for r in top]
                score  = [f"{r['score']:.2f}" for r in top]

                cols = [values, probs, freq, boost, score]

                col_widths = [
                    max(len(str(h)), max(len(str(v)) for v in col))
                    for h, col in zip(headers, cols)
                ]

                header_row = "| " + " | ".join(
                    f"{h:^{w}}" for h, w in zip(headers, col_widths)
                ) + " |\n"

                write(log, header_row)

                sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|\n"
                write(log, sep)

                for i in range(len(top)):
                    row = "| " + " | ".join(
                        f"{str(cols[j][i]):^{col_widths[j]}}"
                        for j in range(len(headers))
                    ) + " |\n"

                    write(log, row)

            # ========================= PATTERN + VERSION
            def write_pattern_and_version(series):
                write(log, "\n📊 PATTERN + VERSION 1\n")

                def render_block(title, window_sizes, feature_func, mapping):
                    headers = ["FEAT.", "WS", "GOAL(S)", "NOTE"]

                    write(log, f"\n{title}\n")
                    write(log, "|" + "|".join(f"{h:^10}" for h in headers) + "|\n")
                    write(log, "|" + "|".join("-" * 10 for _ in headers) + "|\n")

                    for window_size in window_sizes:
                        features = feature_func(series, window_size)
                        keys_to_show = mapping.get(window_size, [])

                        for label, key in keys_to_show:

                            best_flag = ""
                            if title == "PATTERN" and label == "HM":
                                best_flag = "BEST"

                            row = [
                                label,
                                window_size,
                                f"{features[key]:.2f}" if isinstance(features[key], (int, float)) else str(features[key]),
                                best_flag
                            ]

                            write(
                                log,
                                "|" + "|".join(f"{str(v):^10}" for v in row) + "|\n"
                            )

                pattern_map = {
                    1: [("PK", "Peak")],
                    2: [("HM", "HilbertMax")]
                }

                version_map = {
                    4: [("PK", "Peak")],
                    1: [("HM", "HilbertMax")]
                }

                render_block("PATTERN", WINDOW_SIZES, self.arround_last_features_pattern, pattern_map)
                render_block("VERSION 1", WINDOW_SIZES2, self.arround_last_features, version_map)

            # ========================= MOTIF TABLE (identique)
            def write_motif_table(title, results):
                write(log, f"\n📊 {title}\n")

                if not results:
                    write(log, "No data\n")
                    return

                headers = ["VAL", "%", "%E", "SC", "EQ", "PR", "L"]

                rows = []
                for k, v in results.items():
                    value = v.get("value", 0)

                    row = [
                        int(value),
                        round(v.get("pourcentages", {}).get(value, 0) * 100),
                        round(v.get("pourcentages_enrichi", {}).get(value, 0) * 100),
                        round(v.get("adjusted", 0), 2),
                        v.get("equal", 0),
                        v.get("prop", 0),
                        v.get("L", "-"),
                    ]
                    rows.append(row)

                rows.sort(key=lambda r: (-r[2], r[0]))

                col_widths = [
                    max(len(str(row[i])) for row in rows + [headers])
                    for i in range(len(headers))
                ]

                def fmt(row):
                    return "|" + "|".join(
                        f"{str(val):>{col_widths[i]}}"
                        for i, val in enumerate(row)
                    ) + "|\n"

                write(log, fmt(headers))
                write(log, "|" + "|".join("-" * w for w in col_widths) + "|\n")

                for r in rows:
                    write(log, fmt(r))

            # ========================= LOWER BLOCK COMPLET
            def write_lower(series, opponent_series):

                # PATTERN LAST40
                pattern_probs = self.predict_pattern_last40(series)

                write(log, "📊 PATTERN LAST40\n")

                if not pattern_probs:
                    write(log, "No data\n")
                else:
                    sorted_probs = sorted(pattern_probs.items(), key=lambda x: x[1], reverse=True)
                    top = sorted_probs[:5]

                    col_width = max(
                        max(len(str(int(v))) for v, _ in top),
                        max(len(str(int(round(p*100)))) for _, p in top)
                    )

                    label_val = "VAL"
                    label_proba = "PROBA"
                    label_width = max(len(label_val), len(label_proba))

                    row_vals = f"|{label_val:<{label_width}}|" + "|".join(
                        f"{int(v):{col_width}}" for v, _ in top
                    ) + "|\n"

                    row_probs = f"|{label_proba:<{label_width}}|" + "|".join(
                        f"{int(round(p*100)):{col_width}}" for _, p in top
                    ) + "|\n"

                    write(log, row_vals)
                    write(log, row_probs)

                # SHIFT
                write(log, "\n📊 SHIFT-ATTRACTOR-MARKOV\n")

                rs = self.regime_shift_detector(series)
                write(log, f"🎯 Shi. → ⚽ {rs['prediction']:.2f} | Shift: {rs['shift']} | Score: {rs['score']:.2f}\n")

                at = self.local_attractor_prediction(series)
                write(log, f"🎯 Att. → ⚽ {at['prediction']:.2f} | Neighbors: {at['neighbors']} | Avg Distance: {at['avg_distance']:.2f}\n")

                mk = self.markov_weighted_prediction(series)
                write(log, f"🎯 Mar. → ⚽ {mk['prediction']:.2f} | State: {mk['state']} | Confidence: {mk['confidence']:.2f}\n")

                # BONUS MALUS
                write(log, "\n📊 /3= LOW & /2= X\n")

                adv = self.analyze_blocks_and_frequencies(series)
                bm = adv.get("bonus_malus", {})
                spectre = bm.get("spectre", {})

                all_keys_bm = sorted(int(float(k)) for k in spectre.keys())

                if not all_keys_bm:
                    write(log, "No bonus/malus data\n")
                else:
                    col_width = max(2, max(len(str(k)) for k in all_keys_bm))
                    label_header = "VAL  "
                    label_width = len(label_header)

                    header = f"|{label_header:<{label_width}}|" + "|".join(
                        f"{k:{col_width}}" for k in all_keys_bm
                    ) + "|\n"

                    def row(values, label):
                        return f"|{label:<{label_width}}|" + "|".join(
                            f"{v:{col_width}}" for v in values
                        ) + "|\n"

                    def scale(data):
                        return [round(data.get(k, 0) * 100) for k in all_keys_bm]

                    for title, data in [
                        ("🎯 X CONFLICT (%)", bm.get("prev_bonus", {})),
                        ("\n🎯 V CONFLICT (%)", bm.get("prev_prev_bonus", {})),
                    ]:
                        write(log, title + "\n")
                        write(log, header)
                        write(log, row(scale(data), "PROBA"))

                self.display_recent_averages(series, log)
                self.display_median_extrema_means(series, log, "min")
                self.display_median_extrema_means(series, log, "max")

                # PREDICTION SCORES
                write(log, "\n📊 PREDICTION SCORES\n")

                ps = self.prediction_scores(series)

                if not ps:
                    write(log, "Not enough data\n")
                else:
                    keys = sorted(ps["keys"], key=lambda k: ps["proba"][k], reverse=True)

                    headers = ["VAL", "FREQ", "%", "AGE", "BAR", "PROBA"]

                    values = keys
                    freq   = [ps["freq"][k] for k in keys]
                    pct    = [round(ps["pourcentages"][k]*100) for k in keys]
                    age    = [f"{ps['anciennete'][k]:.2f}" for k in keys]
                    bar    = [ps["bareme"][k] for k in keys]
                    proba  = [f"{ps['proba'][k]*100:.2f}" for k in keys]

                    cols = [values, freq, pct, age, bar, proba]

                    col_widths = [
                        max(len(str(h)), max(len(str(v)) for v in col))
                        for h, col in zip(headers, cols)
                    ]

                    header_row = "|" + "|".join(
                        f"{h:^{w}}" for h, w in zip(headers, col_widths)
                    ) + "|\n"
                    write(log, header_row)

                    sep = "|" + "|".join("-" * w for w in col_widths) + "|\n"
                    write(log, sep)

                    for i in range(len(values)):
                        row = "|" + "|".join(
                            f"{str(cols[j][i]):^{col_widths[j]}}"
                            for j in range(len(headers))
                        ) + "|\n"
                        write(log, row)

                    write(log, f"\n🎯 BEST → ⚽ {ps['prediction']} | adjusted={ps['adjusted']:.2f}\n")

            # ========================= EXECUTION
            write_predict_proba(seriesA)
            write_pattern_and_version(seriesA)

            write_motif_table("MOTIF ENGINE COMPLETE - A", self.motif_engine_complete2(seriesA))
            write_motif_table("MOTIF ENGINE TARGETED + CR2 - A", self.motif_engine_targeted_with_cr2(seriesA, tol=0.15))
            write_motif_table("MOTIF ENGINE COMPLETE V2 - A", self.motif_engine_complete2_2(seriesA))
            write_motif_table("MOTIF ENGINE TARGETED V2 - A", self.motif_engine_targeted_with_cr2_2(seriesA, tol=0.15))

            write_lower(seriesA, seriesC)

            log("\n" + "-" * 50 + "\n")

            write_predict_proba(seriesC)
            write_pattern_and_version(seriesC)

            write_motif_table("MOTIF ENGINE COMPLETE - C", self.motif_engine_complete2(seriesC))
            write_motif_table("MOTIF ENGINE TARGETED + CR2 - C", self.motif_engine_targeted_with_cr2(seriesC, tol=0.15))
            write_motif_table("MOTIF ENGINE COMPLETE V2 - C", self.motif_engine_complete2_2(seriesC))
            write_motif_table("MOTIF ENGINE TARGETED V2 - C", self.motif_engine_targeted_with_cr2_2(seriesC, tol=0.15))

            write_lower(seriesC, seriesA)

        messagebox.showinfo("Benchmark", "Benchmark terminé ✔")

    def run_prediction(self,motif_length=2):
        seriesA=read_numeric_after_marker(self.teamA_scores,"A")
        seriesC=read_numeric_after_marker(self.teamC_scores,"C")

        if not seriesA or not seriesC:
            self.log_csv.insert(tk.END,"CSV vide – arrêt.")
            self.log_csv.see(tk.END)
            return

        for log in (self.log_teamA,self.log_teamA_1,self.log_teamC,self.log_teamC_1):
            log.delete("1.0",tk.END)

        resultA,resultC=self.fully_static_method_with_patterns(seriesA,seriesC)

        def write_predict_proba(series, log):
            self.write_log(log, "📊 PROBABILITY PATTERN (DISTANCE MODEL)\n")

            results = self.predict_proba(series, window=5)

            if not results:
                self.write_log(log, "Not enough data\n")
                return

            seen = set()
            filtered = []

            for r in sorted(results, key=lambda x: x["probability_%"], reverse=True):
                if r["value"] in seen:
                    continue
                seen.add(r["value"])
                filtered.append(r)

            top = filtered[:5]

            headers = ["VALUE", "PROBA", "FREQ", "BOOST", "SCORE"]

            values = [r["value"] for r in top]
            probs  = [f"{r['probability_%']:.2f}" for r in top]
            freq   = [f"{r['freq_%']:.2f}" for r in top]
            boost  = [f"{r['boost_%']:.2f}" for r in top]
            score  = [f"{r['score']:.2f}" for r in top]

            cols = [values, probs, freq, boost, score]

            col_widths = [
                max(len(str(h)), max(len(str(v)) for v in col))
                for h, col in zip(headers, cols)
            ]

            header_row = "| " + " | ".join(
                f"{h:^{w}}" for h, w in zip(headers, col_widths)
            ) + " |\n"

            self.write_log(log, header_row)

            sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|\n"
            self.write_log(log, sep)

            for i in range(len(top)):
                row = "| " + " | ".join(
                    f"{str(cols[j][i]):^{col_widths[j]}}"
                    for j in range(len(headers))
                ) + " |\n"

                self.write_log(log, row)

        write_predict_proba(seriesA, self.log_teamA)
        write_predict_proba(seriesC, self.log_teamC)

        def write_pattern_and_version(series, log):
            self.write_log(log, "\n📊 PATTERN + VERSION 1\n")

            def safe(x):
                try:
                    return f"{float(x):.2f}"
                except:
                    return str(x)

            def render_block(title, window_sizes, feature_func, mapping, best_window=None):

                headers = ["FEAT.", "WS", "GOAL(S)", "NOTE"] 

                self.write_log(log, f"\n{title}\n")
                self.write_log(log, "|" + "|".join(f"{h:^10}" for h in headers) + "|\n")
                self.write_log(log, "|" + "|".join("-" * 10 for _ in headers) + "|\n")

                for window_size in window_sizes:
                    features = feature_func(series, window_size)

                    keys_to_show = mapping.get(window_size, [])
                    if not keys_to_show:
                        continue

                    for label, key in keys_to_show:

                        best_flag = ""
                        if title == "PATTERN" and label == "HM":
                            best_flag = "BEST"

                        row = [
                            label,
                            window_size,
                            f"{features[key]:.2f}" if isinstance(features[key], (int, float)) else str(features[key]),
                            best_flag
                        ]

                        self.write_log(
                            log,
                            "|" + "|".join(f"{str(v):^10}" for v in row) + "|\n"
                        )

            pattern_map = {
                1: [("PK", "Peak")],
                2: [("HM", "HilbertMax")]
            }

            render_block(
                "PATTERN",
                WINDOW_SIZES,
                self.arround_last_features_pattern,
                pattern_map
            )

            version_map = {
                4: [("PK", "Peak")],
                1: [("HM", "HilbertMax")]
            }

            render_block(
                "VERSION 1",
                WINDOW_SIZES2,
                self.arround_last_features,
                version_map
            )
                                    
        write_pattern_and_version(seriesA, self.log_teamA)
        write_pattern_and_version(seriesC, self.log_teamC)

        results_complete_A = self.motif_engine_complete2(seriesA)
        results_targeted_A = self.motif_engine_targeted_with_cr2(seriesA, tol=0.15)
        results_complete2_A = self.motif_engine_complete2_2(seriesA)
        results_targeted2_A = self.motif_engine_targeted_with_cr2_2(seriesA, tol=0.15)

        results_complete_C = self.motif_engine_complete2(seriesC)
        results_targeted_C = self.motif_engine_targeted_with_cr2(seriesC, tol=0.15)
        results_complete2_C = self.motif_engine_complete2_2(seriesC)
        results_targeted2_C = self.motif_engine_targeted_with_cr2_2(seriesC, tol=0.15)


        def write_motif_table(title, results, log):
            self.write_log(log, f"\n📊 {title}\n")

            if not results:
                self.write_log(log, "No data\n")
                return

            def short_key(k):
                mapping = {
                    "_DIRECT": "D",
                    "_INVERS": "I",
                    "_MIROIR": "M",
                    "MIR+INV": "MI",
                    "_SIGNES": "S",
                    "__REPLI": "R"
                }

                for long, short in mapping.items():
                    if k.startswith(long):
                        if "_L" in k:
                            L = k.split("_L")[-1]
                            return f"{short}{L}"
                        return short
                return k

            has_cr2 = any(v.get("Cr2") for v in results.values())

            headers = ["VAL", "%", "%E", "SC", "EQ", "PR", "L"]

            def safe(x):
                return round(x, 2) if isinstance(x, (int, float)) else 0

            rows = []
            for k, v in results.items():

                value = v.get("value", 0)

                row = [
                    int(value),
                    round(v.get("pourcentages", {}).get(value, 0) * 100),
                    round(v.get("pourcentages_enrichi", {}).get(value, 0) * 100),
                    safe(v.get("adjusted", 0)),
                    v.get("equal", 0),
                    v.get("prop", 0),
                    v.get("L", "-"),
                ]

                rows.append(row)

            rows.sort(key=lambda r: (-r[2], r[0]))

            col_widths = [
                max(len(str(row[i])) for row in rows + [headers])
                for i in range(len(headers))
            ]

            def fmt(row):
                return "|" + "|".join(
                    f"{str(val):>{col_widths[i]}}"
                    for i, val in enumerate(row)
                ) + "|\n"

            self.write_log(log, fmt(headers))
            self.write_log(log, "|" + "|".join("-" * w for w in col_widths) + "|\n")

            for r in rows:
                self.write_log(log, fmt(r))

        write_motif_table("MOTIF ENGINE COMPLETE - A", results_complete_A, self.log_teamA)
        write_motif_table("MOTIF ENGINE TARGETED + CR2 - A", results_targeted_A, self.log_teamA)
        write_motif_table("MOTIF ENGINE COMPLETE V2 - A", results_complete2_A, self.log_teamA)
        write_motif_table("MOTIF ENGINE TARGETED V2 - A", results_targeted2_A, self.log_teamA)

        write_motif_table("MOTIF ENGINE COMPLETE - C", results_complete_C, self.log_teamC)
        write_motif_table("MOTIF ENGINE TARGETED + CR2 - C", results_targeted_C, self.log_teamC)
        write_motif_table("MOTIF ENGINE COMPLETE V2 - C", results_complete2_C, self.log_teamC)
        write_motif_table("MOTIF ENGINE TARGETED V2 - C", results_targeted2_C, self.log_teamC)
                     
        def write_lower(series, opponent_series, result, log, team_label):
            pattern_probs = self.predict_pattern_last40(series)

            self.write_log(log, "📊 PATTERN LAST40\n")

            if not pattern_probs:
                self.write_log(log, "No data\n")
            else:
                sorted_probs = sorted(pattern_probs.items(), key=lambda x: x[1], reverse=True)

                top = sorted_probs[:5]

                col_width = max(
                    max(len(str(int(v))) for v, _ in top),
                    max(len(str(int(round(p*100)))) for _, p in top)
                )

                label_val = "VAL"
                label_proba = "PROBA"

                label_width = max(len(label_val), len(label_proba))

                row_vals = f"|{label_val:<{label_width}}|" + "|".join(
                    f"{int(v):{col_width}}" for v, _ in top
                ) + "|\n"

                row_probs = f"|{label_proba:<{label_width}}|" + "|".join(
                    f"{int(round(p*100)):{col_width}}" for _, p in top
                ) + "|\n"

                self.write_log(log, row_vals)
                self.write_log(log, row_probs)
                
            self.write_log(log,"\n📊 SHIFT-ATTRACTOR-MARKOV\n")
            regime_shift = self.regime_shift_detector(series)
            self.write_log(log, f"🎯 Shi. → ⚽ {regime_shift['prediction']:.2f} | Shift: {regime_shift['shift']} | Score: {regime_shift['score']:.2f}\n")

            attractor_prediction = self.local_attractor_prediction(series)
            self.write_log(log, f"🎯 Att. → ⚽ {attractor_prediction['prediction']:.2f} | Neighbors: {attractor_prediction['neighbors']} | Avg Distance: {attractor_prediction['avg_distance']:.2f}\n")

            markov_prediction = self.markov_weighted_prediction(series)
            self.write_log(log, f"🎯 Mar. → ⚽ {markov_prediction['prediction']:.2f} | State: {markov_prediction['state']} | Confidence: {markov_prediction['confidence']:.2f}\n")
            
            self.write_log(log, "\n📊 /3= LOW & /2= X\n")
            adv = self.analyze_blocks_and_frequencies(series)
            bm = adv.get("bonus_malus", {})
            spectre = bm.get("spectre", {})

            all_keys_bm = sorted(int(float(k)) for k in spectre.keys())

            if not all_keys_bm:
                self.write_log(log, "No bonus/malus data\n")
            else:
                col_width = max(2, max(len(str(k)) for k in all_keys_bm))
                label_header = "VAL  "
                label_width = len(label_header)

                header = f"|{label_header:<{label_width}}|" + "|".join(
                    f"{k:{col_width}}" for k in all_keys_bm
                ) + "|\n"

                def row(values, label):
                    return f"|{label:<{label_width}}|" + "|".join(
                        f"{v:{col_width}}" for v in values
                    ) + "|\n"

                def scale(data):
                    return [round(data.get(k, 0) * 100) for k in all_keys_bm]

                tables = [
                    ("🎯 X CONFLICT (%)", bm.get("prev_bonus", {})),
                    ("\n🎯 V CONFLICT (%)", bm.get("prev_prev_bonus", {})),
                ]

                for title, data in tables:
                    self.write_log(log, f"{title}\n")
                    self.write_log(log, header)
                    self.write_log(log, row(scale(data), "PROBA"))

            results_complete = self.motif_engine_complete2(series)
            results_targeted = self.motif_engine_targeted_with_cr2(series, tol=0.15)

            divergenceT = self.decide_divergence_targeted(results_targeted)
                
            results_complete2 = self.motif_engine_complete2_2(series)
            results_targeted2 = self.motif_engine_targeted_with_cr2_2(series, tol=0.15)

            divergenceT2 = self.decide_divergence_targeted_2(results_targeted2)

            lin = self.linear_rebound_prediction(series)
            peak = self.peak_envelope_prediction(series)
            avg_rem = self.compute_average_remaining(series)
            
            self.display_recent_averages(series,log)
            self.display_median_extrema_means(series,log,"min")
            self.display_median_extrema_means(series,log,"max")

            self.write_log(log, "\n📊 PREDICTION SCORES\n")

            ps = self.prediction_scores(series)

            if not ps:
                self.write_log(log, "Not enough data\n")
            else:
                keys = sorted(ps["keys"], key=lambda k: ps["proba"][k], reverse=True)

                headers = ["VAL", "FREQ", "%", "AGE", "BAR", "PROBA"]

                values = keys
                freq   = [ps["freq"][k] for k in keys]
                pct    = [round(ps["pourcentages"][k]*100) for k in keys]
                age    = [f"{ps['anciennete'][k]:.2f}" for k in keys]
                bar    = [ps["bareme"][k] for k in keys]
                proba  = [f"{ps['proba'][k]*100:.2f}" for k in keys]

                cols = [values, freq, pct, age, bar, proba]

                col_widths = [
                    max(len(str(h)), max(len(str(v)) for v in col))
                    for h, col in zip(headers, cols)
                ]

                header_row = "|" + "|".join(
                    f"{h:^{w}}" for h, w in zip(headers, col_widths)
                ) + "|\n"
                self.write_log(log, header_row)

                sep = "|" + "|".join("-" * w for w in col_widths) + "|\n"
                self.write_log(log, sep)

                for i in range(len(values)):
                    row = "|" + "|".join(
                        f"{str(cols[j][i]):^{col_widths[j]}}"
                        for j in range(len(headers))
                    ) + "|\n"
                    self.write_log(log, row)

                best = ps["prediction"]
                adjusted = ps["adjusted"]

                self.write_log(
                    log,
                    f"\n🎯 BEST → ⚽ {best} | adjusted={adjusted:.2f}\n"
                )

                self.write_log(log, "\n📊 LINEAR Reb. & PEAK Env.\n")

                lin_simple = self.linear_rebound2(series)
                peak_simple = self.peak_envelope2(series)

                lin_pred = self.linear_rebound_prediction2(series)
                peak_pred = self.peak_envelope_prediction2(series)

                lin_val = lin_simple if isinstance(lin_simple, (int, float)) else 0
                peak_val = peak_simple if isinstance(peak_simple, (int, float)) else 0

                lin_p = lin_pred.get("prediction_linear", 0) if lin_pred else 0
                lin_mk = lin_pred.get("adjusted", 0) if lin_pred else 0

                peak_p = peak_pred.get("prediction_peak", 0) if peak_pred else 0
                peak_mk = peak_pred.get("adjusted", 0) if peak_pred else 0

                text = (
                    f"\n🎯 LINEAR PRED     → ⚽ {lin_p:.2f} | MK ⚽={lin_mk:.2f}\n"
                    f"🎯 PEAK PRED       → ⚽ {peak_p:.2f} | MK ⚽={peak_mk:.2f}\n"
                )

                self.write_log(log, text)

                self.write_log(log, "\n📊 SCORES INDIVIDUELS TABLES\n")

                models = {
                    "A_MODEL": lambda: self.predict_score_from_seriesA(series, opponent_series),
                    "B_MODEL": lambda: self.predict_score_from_seriesB(series, opponent_series),
                    "C_MODEL": lambda: self.predict_score_from_seriesC(series, opponent_series),
                    "D_MODEL": lambda: self.predict_score_from_seriesD(series),
                    "E_MODEL": lambda: self.predict_score_from_seriesE(series),
                }

                for name, func in models.items():

                    if name == "A_MODEL":
                        self.write_log(log, f"🔹 {name}\n")
                    else:
                        self.write_log(log, f"\n🔹 {name}\n")

                    try:
                        res = func()
                    except Exception as e:
                        self.write_log(log, f"Error: {e}\n")
                        continue

                    if name == "A_MODEL" and isinstance(res, dict) and "A" in res:
                        res = res["A"]

                    if not isinstance(res, dict) or "proba" not in res:
                        self.write_log(log, "No data\n")
                        continue

                    proba = res.get("proba", {})
                    freq = res.get("freq", {})
                    pct  = res.get("pourcentages", {})
                    age  = res.get("anciennete", {})
                    bar  = res.get("bareme", {})

                    top = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:5]

                    headers = ["VAL", "PROBA%", "FREQ", "%", "AGE", "BAR"]

                    rows = []
                    for val, p in top:
                        rows.append([
                            f"{val:.2f}",
                            f"{p*100:.2f}",
                            freq.get(val, 0),
                            f"{pct.get(val,0)*100:.2f}",
                            f"{age.get(val,0):.2f}",
                            bar.get(val, 0)
                        ])

                    col_widths = [
                        max(len(str(row[i])) for row in rows + [headers])
                        for i in range(len(headers))
                    ]

                    def fmt(row):
                        return "|" + "|".join(
                            f"{str(val):^{col_widths[i]}}"
                            for i, val in enumerate(row)
                        ) + "|\n"

                    self.write_log(log, fmt(headers))
                    self.write_log(log, "|" + "|".join("-" * w for w in col_widths) + "|\n")

                    for r in rows:
                        self.write_log(log, fmt(r))

                    best_val = res.get("prediction", 0)
                    best_adj = res.get("adjusted", 0)

                    self.write_log(
                        log,
                        f"🎯 BEST → ⚽ {best_val:.2f} | adj={best_adj:.2f}\n"
                    )
                    
        write_lower(seriesA, seriesC, resultA, self.log_teamA_1, "TEAM A")
        write_lower(seriesC, seriesA, resultC, self.log_teamC_1, "TEAM C")
                            
    def prev_csv(self):
        if self.current_index>0:
            self.current_index-=1
            self.loaded_csv_path=self.csv_log[self.current_index]
            self.load_csv();self.run_prediction()

    def next_csv(self):
        if self.current_index<len(self.csv_log)-1:
            self.current_index+=1
            self.loaded_csv_path=self.csv_log[self.current_index]
            self.load_csv();self.run_prediction()

    def on_close(self):self.destroy()

if __name__=="__main__":
    os.makedirs(EXPORT_DIR,exist_ok=True)
    app=FlashscoreApp()
    app.mainloop()


