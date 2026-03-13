#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import traceback
import csv
import os
from datetime import datetime
import shutil
import subprocess
import sys
from typing import List, Optional, Dict
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import simpledialog
from collections import Counter
from itertools import combinations
import math

DEFAULT_SCORES = {}
EXPORT_DIR = "export_csv" # proposer un chemin configurable

def classify_value(v):
    if v == 0:
        return 0
    if v > 0:
        return math.floor(v)
    if v < 0:
        return math.ceil(v)

def save_error_to_csv(err_text, filename="error_log.txt"):
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "error_traceback"])

        writer.writerow([datetime.now().isoformat(), err_text])

def is_number(v) -> bool:
    try:
        int(v)
        return True
    except Exception:
        return False
    
def open_folder(path: str):
    path = os.path.abspath(path)
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def open_file(path: str):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def list_subfolders(base: str) -> List[str]:
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))

def count_csv_files(folder_path: str) -> int:
    if not os.path.isdir(folder_path):
        return 0
    return sum(1 for f in os.listdir(folder_path) if f.lower().endswith(".csv"))

def list_csv_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))

def clear_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        return
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Erreur suppression {path}: {e}")

def read_numeric_after_marker(column: list, marker: str) -> list:
    start = next(
        (i + 1 for i, v in enumerate(column)
         if str(v).strip().upper() == marker),
        None
    )
    if start is None:
        return []

    return [
        int(v) for v in column[start:]
        if is_number(v)
    ]

class FlashscoreApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PREDICTOR(F4)")
        self.geometry("1400x1000")
        self.teamA_scores: List[str] = []
        self.teamC_scores: List[str] = []
        self.motif_next_distance = 0
        
        self.motif_configs = [            
            ("0= OVER", 2, 30), # colonne 2
            ("0= OVER", 3, 30), # colonne 3
            ("0= OVER", 4, 30), # colonne 4
            ("0= OVER", 5, 30), # colonne 5
            ("0= OVER", 6, 30), # colonne 6
            ("0= OVER", 2, 60), # colonne 7
            
            ("(-)= OVER", 3, 60), # colonne 8
            ("(-)= OVER", 4, 60), # colonne 9
            ("(-)= OVER", 5, 60), # colonne 10
            ("(-)= OVER", 6, 60), # colonne 11

            ("(-)= OVER", 2, 10), # colonne 12
            ("(-)= OVER", 2, 20), # colonne 13
            ("(-)= OVER", 2, 40), # colonne 14
            ("(-)= OVER", 2, 50), # colonne 15
            ("(-)= OVER", 2, 70), # colonne 16

        ]

        self.motif_configs2 = [            
            ("1.BASE", 2, 3), #
            ("2.TREND", 3, 4), #
            ("3.TREND", 4, 5), #
            ("4.TREND", 5, 6), #
            ("5.LIMIT", 3, 6), #

            ("11.BASE", 6, 7), #
            ("22.TREND", 7, 8), #
            ("33.TREND", 8, 9), #
            ("44.TREND", 9, 10), #
            ("55.TREND", 10, 11), #
            #("66.TREND", 11, 12), #
            #("77.LIMIT", 6, 12), #

            #("SHORT", 3, 15), #
            #("WIDE", 3, 30), #
            #("3 >", 2, 45), #
            #("4 >", 5, 15), #
            #("WIDE 2", 5, 30), #
            #("6 >", 5, 45), #
            #("7 >", 7, 15), #
            #("WIDE 3", 7, 30), #
            #("9 >", 7, 45), #
            #("WIDE L", 3, 60), #
            #("11 >", 5, 60), #
            #("WIDE XL", 7, 60), #
        ]
        
        self.csv_files: List[str] = []
        self.loaded_csv_path: Optional[str] = None
        self.csv_loaded: bool = False

        self.csv_log: List[str] = []
        self.hidden_csv: set = set()   
        self.current_index: int = -1

        self.listbox_to_csv_index = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)        

    def on_csv_click(self, event):
        #index = self.log_csv.index(f"@{event.x},{event.y}")
        index = self.log_csv.nearest(event.y) # <== supprimer en cas de bug
        line = int(index.split(".")[0]) - 1  

        if 0 <= line < len(self.csv_log):
            path = self.csv_log[line]

            self.current_index = line
            self.loaded_csv_path = path

            self.load_csv()
            self.run_prediction()
    
    def analyze_zero_pattern(self, series):
        zero_sequences = []

        i = 0
        n = len(series)

        while i < n:
            if series[i] == 0:
                start = i
                while i < n and series[i] == 0:
                    i += 1
                length = i - start

                if length > 2 and i < n:
                    zero_sequences.append(series[i])
            else:
                i += 1

        zero_signal = None
        last_value = None

        if len(zero_sequences) >= 3:

            before_before_last = zero_sequences[-3]
            before_last = zero_sequences[-2]
            last_value = zero_sequences[-1]

            if before_last > before_before_last:
                zero_signal = "Under"
            elif before_last < before_before_last:
                zero_signal = "Over"

        return zero_sequences, zero_signal, last_value

    def display_recent_averages(self, series, log_widget):
        series_data = series[:-1]  
        lengths = [5, 10, 15]

        averages = []
        for L in lengths:
            if len(series_data) >= L:
                avg = sum(series_data[-L:]) / L
                averages.append(f"Last{L}={avg:.2f}")
            else:
                averages.append(f"Last{L}=N/A")

        line = " | ".join(averages)
        self.write_log(log_widget, f"\n📊 LAST MEANS : {line}\n")

    def fully_static_method_pattern1(self, series, motif_length, block_size, motif_next_distance=0):
        series_data = series[:-1]
        base_pred = series[-1]
        n = len(series_data)

        last_bloc = series_data[-block_size:]
        max_val = max(last_bloc)
        min_val = min(last_bloc)

        target_motif = None
        for idxs in combinations(range(len(last_bloc)), motif_length):
            target_motif = [last_bloc[i] for i in idxs]
            break

        def weighted_linear_regression(values):
            n = len(values)
            if n < 2:
                return sum(values)

            x = list(range(n))
            y = values
            w = [n - i for i in range(n)]

            sum_w = sum(w)
            sum_wx = sum(w[i] * x[i] for i in range(n))
            sum_wy = sum(w[i] * y[i] for i in range(n))
            sum_wxx = sum(w[i] * x[i] * x[i] for i in range(n))
            sum_wxy = sum(w[i] * x[i] * y[i] for i in range(n))

            denom = (sum_w * sum_wxx - sum_wx * sum_wx)

            if denom == 0:
                return sum(values)

            slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
            intercept = (sum_wy - slope * sum_wx) / sum_w

            next_x = n
            pred = intercept + slope * next_x

            return pred

        def remove_farthest_occurrence(value):
            indices = [i for i, v in enumerate(last_bloc) if v == value]
            if not indices:
                return None
            farthest = max(indices)
            motif = target_motif.copy()
            if value in motif:
                motif.remove(value)
                return motif
            return None

        motif_excl_max = remove_farthest_occurrence(max_val)
        motif_excl_min = remove_farthest_occurrence(min_val)

        search_zones = []
        for i in range(0, n - block_size, block_size):
            sim_bloc = series_data[i:i+block_size]
            if i + block_size < len(series_data):
                corres_next = series_data[i+block_size]
                search_zones.append((i, sim_bloc, corres_next))

        store = {f"pred_results{i}": [] for i in range(1, 8)}

        def ordered_match(sim_bloc, motif):
            pos = 0
            for val in sim_bloc:
                if pos < len(motif) and val == motif[pos]:
                    pos += 1
            return pos == len(motif)

        def delta_calc(i, corres_next):
            return max(0, corres_next - i)

        for i, sim_bloc, corres_next in search_zones:

            delta = delta_calc(i, corres_next)
            matched_primary = False

            if ordered_match(sim_bloc, target_motif):
                store["pred_results1"].append(delta)
                matched_primary = True

            if not matched_primary:
                sum_target = sum(target_motif)
                ratio = sum(sim_bloc) / sum_target if sum_target != 0 else 1
                if ratio > 1 or (0 < ratio < 1):
                    store["pred_results2"].append(delta)
                    matched_primary = True

            if motif_excl_max and ordered_match(sim_bloc, motif_excl_max):
                store["pred_results4"].append(delta)

            if motif_excl_min and ordered_match(sim_bloc, motif_excl_min):
                store["pred_results5"].append(delta)

            if motif_excl_max and ordered_match(sim_bloc, motif_excl_max):
                if delta not in store["pred_results4"]:
                    store["pred_results6"].append(delta)

            if motif_excl_min and ordered_match(sim_bloc, motif_excl_min):
                if delta not in store["pred_results5"]:
                    store["pred_results7"].append(delta)

        pred_results3 = store["pred_results1"] + store["pred_results2"]

        pred_results12 = (
            pred_results3
            + store["pred_results4"]
            + store["pred_results5"]
            + store["pred_results6"]
            + store["pred_results7"]
        )

        combined_map = {
            3: pred_results3,
            12: pred_results12
        }

        def stats(lst):

            if not lst:
                return {
                    "prediction": 0,
                    "count": 0
                }

            return {
                "prediction": weighted_linear_regression(lst),
                "count": len(lst)
            }

        final_results = {}

        for i in range(1, 8):
            final_results[f"pred_results{i}"] = stats(store[f"pred_results{i}"])

        for k, v in combined_map.items():
            final_results[f"pred_results{k}"] = stats(v)

        return final_results

    def fully_static_method_pattern0(self, series, motif_length, block_size, motif_next_distance):
        series_data = series[:-1]
        n = len(series_data)

        last_bloc = series_data[-block_size:]
        base_pred = series[-1]

        # --- Détection du target_motif dans last_bloc ---
        from itertools import combinations

        target_motif = None
        for idxs in combinations(range(len(last_bloc)), motif_length):
            target_motif_candidate = [last_bloc[i] for i in idxs]
            target_motif = target_motif_candidate
            break  # on prend le premier motif valide

        # --- Préparer zones de recherche ---
        sim_blocs_list = []
        corres_next_list = []
        for i in range(0, n - block_size, block_size):
            sim_blocs = series_data[i:i+block_size]
            corres_next = series_data[i+block_size:i+block_size+1]
            sim_blocs_list.append(sim_blocs)
            corres_next_list.append(corres_next)

        # --- Détection des corres_motif ---
        pred_results1, pred_results2, pred_results3 = [], [], []
        pred_results4, pred_results5, pred_results6, pred_results7 = [], [], [], []

        def check_corres_motif(sim_blocs, target):
            return all(val in sim_blocs for val in target)

        delta_sim_pred = []
        for i, sim_blocs in enumerate(sim_blocs_list):
            corres_next = corres_next_list[i][0] if corres_next_list[i] else None
            if corres_next is None:
                continue

            if check_corres_motif(sim_blocs, target_motif):
                if abs(corres_next - base_pred) == motif_next_distance:
                    delta = max(0, corres_next - i)
                    delta_sim_pred.append(delta)
                    pred_results1.append(delta)
            else:
                ratio = sum(sim_blocs)/sum(target_motif) if sum(target_motif) != 0 else 1
                if ratio > 1:
                    delta = max(0, corres_next - i - 1)
                    pred_results2.append(delta)
                    delta_sim_pred.append(delta)
                elif 0 < ratio < 1:
                    delta = max(0, corres_next - i + 1)
                    pred_results2.append(delta)
                    delta_sim_pred.append(delta)

        # Combinaison des résultats
        pred_results3 = pred_results1 + pred_results2
        pred_results6 = pred_results1 + pred_results2 + pred_results4
        pred_results7 = pred_results1 + pred_results2 + pred_results5

        # Résultat final formaté
        results = {
            "pred_results1": {"sum": sum(pred_results1), "count": len(pred_results1)},
            "pred_results2": {"sum": sum(pred_results2), "count": len(pred_results2)},
            "pred_results3": {"sum": sum(pred_results3), "count": len(pred_results3)},
            "pred_results4": {"sum": sum(pred_results4), "count": len(pred_results4)},
            "pred_results5": {"sum": sum(pred_results5), "count": len(pred_results5)},
            "pred_results6": {"sum": sum(pred_results6), "count": len(pred_results6)},
            "pred_results7": {"sum": sum(pred_results7), "count": len(pred_results7)},
        }

        return results

    def fully_static_method_pattern2(self, series, motif_length, block_size, motif_next_distance=0):
        series_data = series[:-1] 
        n = len(series_data)

        last_bloc = series_data[-block_size:]
        max_val = max(last_bloc)
        min_val = min(last_bloc)

        target_motif = None
        for idxs in combinations(range(len(last_bloc)), motif_length):
            target_motif = [last_bloc[i] for i in idxs]
            break

        def weighted_linear_regression(values):
            n = len(values)
            if n < 2:
                return sum(values)
            x = list(range(n))
            y = values
            w = [n - i for i in range(n)]
            sum_w = sum(w)
            sum_wx = sum(w[i] * x[i] for i in range(n))
            sum_wy = sum(w[i] * y[i] for i in range(n))
            sum_wxx = sum(w[i] * x[i] * x[i] for i in range(n))
            sum_wxy = sum(w[i] * x[i] * y[i] for i in range(n))
            denom = (sum_w * sum_wxx - sum_wx * sum_wx)
            if denom == 0:
                return sum(values)
            slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
            intercept = (sum_wy - slope * sum_wx) / sum_w
            pred = intercept + slope * n
            return pred

        def remove_farthest_occurrence(value):
            indices = [i for i, v in enumerate(last_bloc) if v == value]
            if not indices:
                return None
            farthest = max(indices)
            motif = target_motif.copy()
            if value in motif:
                motif.remove(value)
                return motif
            return None

        motif_excl_max = remove_farthest_occurrence(max_val)
        motif_excl_min = remove_farthest_occurrence(min_val)

        search_zones = []
        for i in range(0, n - block_size, block_size):
            sim_bloc = series_data[i:i + block_size]
            if i + block_size < len(series_data):
                corres_next = series_data[i + block_size]
                search_zones.append((i, sim_bloc, corres_next))

        store = {f"pred_results{i}": [] for i in range(1, 8)}

        def ordered_match(sim_bloc, motif):
            pos = 0
            for val in sim_bloc:
                if pos < len(motif) and val == motif[pos]:
                    pos += 1
            return pos == len(motif)

        def delta_calc(i, corres_next):
            return max(0, corres_next - i)

        for i, sim_bloc, corres_next in search_zones:
            delta = delta_calc(i, corres_next)
            matched_primary = False

            if ordered_match(sim_bloc, target_motif):
                store["pred_results1"].append(delta)
                matched_primary = True

            if not matched_primary:
                sum_target = sum(target_motif)
                ratio = sum(sim_bloc) / sum_target if sum_target != 0 else 1
                if ratio > 1 or (0 < ratio < 1):
                    store["pred_results2"].append(delta)
                    matched_primary = True

            if motif_excl_max and ordered_match(sim_bloc, motif_excl_max):
                store["pred_results4"].append(delta)

            if motif_excl_min and ordered_match(sim_bloc, motif_excl_min):
                store["pred_results5"].append(delta)

            if motif_excl_max and ordered_match(sim_bloc, motif_excl_max):
                if delta not in store["pred_results4"]:
                    store["pred_results6"].append(delta)

            if motif_excl_min and ordered_match(sim_bloc, motif_excl_min):
                if delta not in store["pred_results5"]:
                    store["pred_results7"].append(delta)

        def stats(lst):
            if not lst:
                return {"linear_rebound": 0, "peak_envelope": 0, "count": 0}
            lin = self.linear_rebound_prediction(lst)  
            peak = self.peak_envelope_linear_prediction(lst)
            return {"linear_rebound": lin["prediction"], "peak_envelope": peak["prediction"], "count": len(lst)}

        final_results = {}
        for i in range(1, 8):
            final_results[f"pred_results{i}"] = stats(store[f"pred_results{i}"])

        combined_list = (
            store["pred_results1"] +
            store["pred_results2"] +
            store["pred_results4"] +
            store["pred_results5"] +
            store["pred_results6"] +
            store["pred_results7"]
        )
        final_results["pred_results12"] = stats(combined_list)

        return final_results

    def display_consecutive_stats(self, series, log_widget):
        data = series[:-1]
        if not data:
            return
        last_val = data[-1]
        count_last = 0
        i = len(data) - 1
        while i >= 0 and data[i] == last_val:
            count_last += 1
            i -= 1

        zero_last = count_last if last_val == 0 else 0
        one_last  = count_last if last_val == 1 else 0

        self.write_log(
            log_widget,
            f"\n📊 Consecutive last: 0={zero_last}* | 1={one_last}*\n"
        )

        last15 = data[-15:] if len(data) >= 15 else data

        max0 = max1 = current0 = current1 = 0

        for v in last15:
            if v == 0:
                current0 += 1
                max0 = max(max0, current0)
                current1 = 0
            elif v == 1:
                current1 += 1
                max1 = max(max1, current1)
                current0 = 0
            else:
                current0 = current1 = 0

        self.write_log(
            log_widget,
            f"\n📊 Consecutive in 15 last: 0={max0}* | 1={max1}*\n"
        )

    def linear_rebound_prediction(self, series, window=6):
        series = series[:-1]
        if len(series) < 3:
            return {
                "prediction": 0,
                "count": 0
            }

        y_prev1 = series[-1]
        y_prev2 = series[-2]

        base_pred = 1.4 * y_prev2 - 0.6 * y_prev1

        recent = series[-window:] if len(series) >= window else series[:] # 1212
        #recent = series[:]  # utiliser toute la série

        if len(recent) > 1:
            amplitude = max(recent) - min(recent)
            bias = amplitude * 0.4
        else:
            bias = 0

        pred = base_pred + bias

        if pred < 0:
            pred = 0

        return {
            "prediction": pred,
            "count": len(recent)
        }

    def peak_envelope_linear_prediction(self, series, min_peaks=4):
        series = series[:-1]
        n = len(series)

        if n < 5:
            return {
                "prediction": 0,
                "count": 0
            }

        peaks_x = []
        peaks_y = []

        for i in range(1, n - 1):
            if series[i] > series[i-1] and series[i] >= series[i+1]:
                peaks_x.append(i)
                peaks_y.append(series[i])

        if len(peaks_x) < min_peaks:
            return {
                "prediction": 0,
                "count": len(peaks_x)
            }

        m = len(peaks_x)

        sum_x = sum(peaks_x)
        sum_y = sum(peaks_y)
        sum_xx = sum(x*x for x in peaks_x)
        sum_xy = sum(peaks_x[i] * peaks_y[i] for i in range(m))

        denom = m * sum_xx - sum_x * sum_x

        if denom == 0:
            return {
                "prediction": max(peaks_y),
                "count": m
            }

        slope = (m * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / m

        next_x = n
        pred = intercept + slope * next_x

        if pred < 0:
            pred = 0

        return {
            "prediction": pred,
            "count": m
        }

    def display_median_extrema_series_means(self, series, log_widget, mode="min"):
        data = series[:-1]
        if not data:
            return

        sizes = [
            (7, [15, 27, 51]),
            (13, [27, 51, 99]),
            (25, [51, 99])
        ]
        configs = [(mean, zone) for mean, zones in sizes for zone in zones]

        min_zone = min(zone for _, zone in configs)
        if len(data) < min_zone:
            return

        if mode == "min":
            self.write_log(log_widget, "\nMedian minima 'N values mean' in 'N lasts':\n")
            extremum_func = min
        elif mode == "max":
            self.write_log(log_widget, "\nMedian maxima 'N values mean' in 'N lasts':\n")
            extremum_func = max
        else:
            raise ValueError("mode must be 'min' or 'max'")

        for mean_size, zone_size in configs:
            if len(data) < zone_size:
                self.write_log(log_widget, f"{mean_size} in {zone_size} : N/A\n")
                continue

            zone = data[-zone_size:]
            extremum_val = extremum_func(zone)
            idx = zone.index(extremum_val)

            center = zone_size // 2
            if idx != center:
                self.write_log(log_widget, f"{mean_size} in {zone_size} : N/A\n")
                continue

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

    def display_median_extrema_means(self, series, log_widget, mode="min"):
        data = series[:-1]  
        if not data:
            return

        configs = [
            (7, 15), (7, 27), (7, 51),
            (13, 27), (13, 51), (13, 99),
            (25, 51), (25, 99)
        ]

        if mode == "min":
            self.write_log(log_widget, "\n📊 Median minima 'N values mean' in 'N lasts':\n")
            extremum_func = min
        elif mode == "max":
            self.write_log(log_widget, "\n📊 Median maxima 'N values mean' in 'N lasts':\n")
            extremum_func = max
        else:
            raise ValueError("mode must be 'min' or 'max'")

        for mean_size, zone_size in configs:
            if len(data) < zone_size:
                self.write_log(log_widget, f"{mean_size} in {zone_size} : N/A\n")
                continue

            zone = data[-zone_size:]

            extremum_val = extremum_func(zone)
            idx = zone.index(extremum_val)

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

    def compute_blocks_with_gaps(self, series, min_block=2, gap_between=None, log_widget=None):
        series_data = series[:-1]
        n = len(series_data)
        i = n - 1
        results = []
        block_averages = []

        while i >= min_block:
            target = series_data[i]
            found_block = False

            for j in range(i - min_block, -1, -1):
                block = series_data[j:i+1]

                if len(block) >= (min_block + 1) and sum(block)/len(block) == target:
                    block_averages.append(target)
                    i = j - 1
                    found_block = True
                    break

            if not found_block:
                i -= 1

   
        if block_averages:
            lin_res = self.linear_rebound_prediction(block_averages)
            peak_res = self.peak_envelope_linear_prediction(block_averages)

 
            remaining_start = n - len(block_averages)
            remaining = series_data[remaining_start:] if remaining_start < n else []
            avg_remaining = sum(remaining)/len(remaining) if remaining else 0

            results.append({
                "block_values": block_averages,
                "linear_rebound": lin_res['prediction'],
                "peak_envelope": peak_res['prediction'],
                "count": lin_res['count'],
                "remaining_values": remaining,
                "average_remaining": avg_remaining
            })



        return results

    def fully_static_method_with_patterns(self, seriesA, seriesC):

        # --- Fonctions internes ---
        def detect_target_motif(series):
            data = series[:-1]
            n = len(data)
            for L in range(2, n // 2 + 1):
                last_pattern = data[-L:]
                for i in range(n - L - 1, -1, -1):
                    if data[i:i+L] == last_pattern:
                        return last_pattern
            return None

        def is_proportional(motif1, motif2):
            if len(motif1) != len(motif2) or motif1[0] == 0:
                return False
            ratio = motif2[0] / motif1[0]
            for a, b in zip(motif1, motif2):
                if a == 0 or abs((b / a) - ratio) > 1e-9:
                    return False
            return True

        def classify(series, motif):
            data = series[:-1]
            L = len(motif)
            e_list, p_super_list, p_infer_list = [], [], []
            i = 0
            while i <= len(data) - L:
                block = data[i:i+L]
                if block == motif:
                    e_list.append(i)
                    i += L
                    continue
                elif is_proportional(motif, block):
                    ratio = block[0] / motif[0]
                    if ratio > 1:
                        p_super_list.append(i)
                    elif 0 < ratio < 1:
                        p_infer_list.append(i)
                    i += L
                    continue
                i += 1
            m_list = sorted(set(e_list + p_super_list + p_infer_list))
            return e_list, p_super_list, p_infer_list, m_list

        def build_next_vals(series, motif, indices):
            L = len(motif)
            vals = []
            for idx in indices:
                if idx + L < len(series):
                    vals.append(series[idx + L])
            return vals


        # --- Traitement pour seriesA ---
        motifA = detect_target_motif(seriesA)
        if motifA:
            eA, pA, piA, mA = classify(seriesA, motifA)
            next_valsA = build_next_vals(seriesA, motifA, mA)
            reboundA = self.linear_rebound(next_valsA)
            peaksA = self.peak_envelope(next_valsA)
            resultA = (motifA, (eA, pA, piA, mA, next_valsA, reboundA, peaksA))
        else:
            resultA = (None, ([], [], [], [], [], [], []))

        # --- Traitement pour seriesC ---
        motifC = detect_target_motif(seriesC)
        if motifC:
            eC, pC, piC, mC = classify(seriesC, motifC)
            next_valsC = build_next_vals(seriesC, motifC, mC)
            reboundC = self.linear_rebound(next_valsC)
            peaksC = self.peak_envelope(next_valsC)
            resultC = (motifC, (eC, pC, piC, mC, next_valsC, reboundC, peaksC))
        else:
            resultC = (None, ([], [], [], [], [], [], []))

        # --- Affichage en passant les fonctions comme arguments ---
        motifA, dataA = resultA
        eA, pA, piA, mA, nextA, rebA, peakA = dataA
        self.display_motifs(
            seriesA, motifA, eA, pA, piA, mA, nextA, rebA, peakA,
            self.log_teamA_1, "motifA", self.linear_rebound, self.peak_envelope
        )

        motifC, dataC = resultC
        eC, pC, piC, mC, nextC, rebC, peakC = dataC
        self.display_motifs(
            seriesC, motifC, eC, pC, piC, mC, nextC, rebC, peakC,
            self.log_teamC_1, "motifC", self.linear_rebound, self.peak_envelope
        )

        return resultA, resultC

    def linear_rebound(self, values):
        # On prend les deux derniers points pour une prédiction unique
        if len(values) < 2:
            return 0
        y_prev1 = values[-1]
        y_prev2 = values[-2]
        # Base linéaire simple similaire à votre pipeline précédent
        pred = 1.4 * y_prev2 - 0.6 * y_prev1
        # Ajuster par amplitude des derniers points
        window = min(6, len(values))
        recent = values[-window:]
        if len(recent) > 1:
            amplitude = max(recent) - min(recent)
            pred += 0.4 * amplitude
        if pred < 0:
            pred = 0
        return pred

    def peak_envelope(self, values, min_peaks=4):
        n = len(values)
        if n < 5:
            return 0
        peaks = [values[i] for i in range(1, n-1) if values[i] > values[i-1] and values[i] >= values[i+1]]
        if len(peaks) < min_peaks:
            return max(peaks) if peaks else 0
        # On peut faire une prédiction linéaire comme dans peak_envelope_linear_prediction
        m = len(peaks)
        x = list(range(len(peaks)))
        y = peaks
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] - x[0] != 0 else 0
        pred = y[-1] + slope  # prédiction simple
        if pred < 0:
            pred = 0
        return pred

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Charger par dossier", font=("TkDefaultFont", 10, "bold"))\
            .grid(row=0, column=0, sticky="w")

        nav = ttk.Frame(top)
        nav.grid(row=0, column=1, sticky="e")
        ttk.Button(nav, text="<===", command=self.prev_csv).pack(side="left", padx=2)
        ttk.Label(nav, text="Préc.").pack(side="left", padx=(0, 6))
        ttk.Label(nav, text="Suiv.").pack(side="left", padx=(6, 0))
        ttk.Button(nav, text="===>", command=self.next_csv).pack(side="left", padx=2)

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=8, pady=4)

        self.combo_folder = ttk.Combobox(controls, values=[], state="readonly", width=20)
        self.combo_folder.grid(row=0, column=0, padx=4)
        self.combo_folder.bind("<<ComboboxSelected>>", self.on_folder_selected)
        self.refresh_folders()

        self.combo_csv = ttk.Combobox(controls, width=40)
        self.combo_csv.grid(row=0, column=1, padx=4)
        self.combo_csv.bind("<KeyRelease>", self.filter_csv_list)

        self.combo_action = ttk.Combobox(controls, values=["Charger", "Archiver"], state="readonly", width=15)
        self.combo_action.set("Charger")
        self.combo_action.grid(row=0, column=2, padx=4)

        self.combo_scope = ttk.Combobox(controls, values=["Un", "Tout"], state="readonly", width=10)
        self.combo_scope.set("Tout")
        self.combo_scope.grid(row=0, column=3, padx=4)

        ttk.Button(controls, text="Appliquer", command=self.on_apply_clicked)\
            .grid(row=0, column=4, padx=6)

        ttk.Button(controls, text="MOTIF<>NEXT", command=self.on_motif_next_clicked)\
            .grid(row=0, column=5, padx=6)

        ttk.Button(top, text="📊 Benchmark", command=self.run_benchmark).grid(row=0, column=5, padx=6)

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=8, pady=8)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.columnconfigure(2, weight=1)

        self._build_team_frame(mid, 0, "A")
        self._build_team_frame(mid, 1, "C")
        self._build_log_frame(mid, controls)

    def _build_team_frame(self, parent, col, team):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, sticky="nsew", padx=4)

        line = ttk.Frame(frame)
        line.pack(anchor="w", fill="x")

        entry = ttk.Entry(line, width=62)
        entry.pack(side="left")
        setattr(self, f"entry_team{team}", entry)

        entry_score = ttk.Entry(line, width=5)
        entry_score.pack(side="left", padx=(6, 0))
        setattr(self, f"entry_team{team}_score", entry_score)

        entry_score.bind(
            "<FocusOut>",
            lambda e, t=team: self.save_team_score_to_csv(t)
        )

        log1_container = ttk.Frame(frame)
        log1_container.pack(fill="both", expand=True)
        log1 = tk.Text(log1_container, height=15, width=60)
        log1.pack(side="left", fill="both", expand=True)
        scroll1 = ttk.Scrollbar(log1_container, orient="vertical", command=log1.yview)
        scroll1.pack(side="right", fill="y")
        log1.config(yscrollcommand=scroll1.set)
        setattr(self, f"log_team{team}", log1)

        log2_container = ttk.Frame(frame)
        log2_container.pack(fill="both", expand=True)
        log2 = tk.Text(log2_container, height=35, width=60)
        log2.pack(side="left", fill="both", expand=True)
        scroll2 = ttk.Scrollbar(log2_container, orient="vertical", command=log2.yview)
        scroll2.pack(side="right", fill="y")
        log2.config(yscrollcommand=scroll2.set)
        setattr(self, f"log_team{team}_1", log2)

    def _build_log_frame(self, parent, controls):
        container = ttk.Frame(parent)
        container.grid(row=0, column=2, sticky="nsew", padx=(6, 2))
        container.rowconfigure(2, weight=1)
        container.columnconfigure(0, weight=1)

        btns = ttk.Frame(container)
        btns.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ttk.Button(btns, text="Masquer", command=self.hide_selected_csv)\
            .pack(side="left", padx=4)

        ttk.Button(btns, text="Archiver", command=self.archive_selected_csv)\
            .pack(side="left", padx=4)

        ttk.Button(
            container,
            text="Réinitialiser Log CSV",
            command=self.reset_csv_log
        ).grid(row=1, column=0, sticky="w", pady=(0, 4))

        self.log_csv = tk.Listbox(
            container,
            selectmode="extended",
            height=25
        )
        self.log_csv.grid(row=2, column=0, sticky="nsew")

        sb = ttk.Scrollbar(container, command=self.log_csv.yview)
        sb.grid(row=2, column=1, sticky="ns")
        self.log_csv.config(yscrollcommand=sb.set)

        self.log_csv.bind("<<ListboxSelect>>", self.on_csv_select)

        self.refresh_csv_log()

    def on_motif_next_clicked(self):
        new_val = simpledialog.askinteger(
            title="MOTIF<>NEXT",
            prompt="Modifier la distance admise entre corres_motif et corres_next :",
            initialvalue=self.motif_next_distance,
            minvalue=0,
            parent=self
        )

        if new_val is not None:
            self.motif_next_distance = new_val


    def add_csv_log(self, path: str):
        if path not in self.csv_log:
            self.csv_log.append(path)
            self.current_index = len(self.csv_log)-1
            self.refresh_csv_log()

    def split_csv_name_two_lines(self, filename: str):
        name = filename.replace(".csv", "")
        if "_VS_" in name:
            left, right = name.split("_VS_", 1)
            return left + "_VS_", right
        else:
            mid = len(name) // 2
            return name[:mid], name[mid:]

    def refresh_csv_log(self):
        self.log_csv.delete(0, tk.END)
        self.listbox_to_csv_index.clear()

        row = 0
        for csv_index, path in enumerate(self.csv_log):
            if path in self.hidden_csv:
                continue

            base = os.path.basename(path)
            line1, line2 = self.split_csv_name_two_lines(base)

            self.log_csv.insert(tk.END, line1)
            self.listbox_to_csv_index[row] = csv_index
            row += 1

            self.log_csv.insert(tk.END, line2)
            self.listbox_to_csv_index[row] = csv_index
            row += 1

            self.log_csv.insert(tk.END, "────────────")
            row += 1

    def reset_csv_log(self):
        self.csv_log.clear()
        self.current_index=-1
        self.refresh_csv_log()

    def on_csv_select(self, event=None):
        selection = self.log_csv.curselection()
        if not selection:
            return

        line = selection[-1]

        if line not in self.listbox_to_csv_index:
            self.log_csv.selection_clear(0, tk.END)
            return

        csv_index = self.listbox_to_csv_index[line]

        self.log_csv.selection_clear(0, tk.END)
        for lb_line, idx in self.listbox_to_csv_index.items():
            if idx == csv_index:
                self.log_csv.selection_set(lb_line)

        self.current_index = csv_index
        self.loaded_csv_path = self.csv_log[csv_index]
        self.load_csv()
        self.run_prediction()

    def hide_selected_csv(self):
        selection = self.log_csv.curselection()
        for idx in selection:
            if idx in self.listbox_to_csv_index:
                real_idx = self.listbox_to_csv_index[idx]
                self.hidden_csv.add(self.csv_log[real_idx])

        self.refresh_csv_log()

    def archive_selected_csv(self):
        display_name = self.combo_folder.get()
        if not display_name:
            messagebox.showwarning("Avertissement", "Aucun dossier sélectionné.")
            return

        folder = display_name.rsplit(" ", 1)[0]
        archive_dir = os.path.join(EXPORT_DIR, folder)
        os.makedirs(archive_dir, exist_ok=True)

        selection = self.log_csv.curselection()
        real_indices = set()

        for idx in selection:
            if idx in self.listbox_to_csv_index:
                real_indices.add(self.listbox_to_csv_index[idx])

        for real_idx in sorted(real_indices, reverse=True):
            if 0 <= real_idx < len(self.csv_log):
                path = self.csv_log.pop(real_idx)
                if os.path.exists(path):
                    shutil.move(
                        path,
                        os.path.join(archive_dir, os.path.basename(path))
                    )

        self.refresh_csv_log()

    def refresh_folders(self):
        folders_display = []
        for folder in list_subfolders(EXPORT_DIR):
            path = os.path.join(EXPORT_DIR, folder)
            csv_count = count_csv_files(path)
            folders_display.append(f"{folder} {csv_count}")

        self.combo_folder["values"] = folders_display
        if folders_display:
            self.combo_folder.set(folders_display[0])
        else:
            self.combo_folder.set("") 

    def on_folder_selected(self, event=None):
        display_name = self.combo_folder.get()
        if not display_name:
            return

        folder = display_name.rsplit(" ", 1)[0]
        path = os.path.join(EXPORT_DIR, folder)
        self.csv_files = list_csv_files(path)
        self.combo_csv["values"] = self.csv_files
        self.combo_csv.set("")

    def filter_csv_list(self, event=None):
        typed = self.combo_csv.get().lower()
        self.combo_csv["values"] = [f for f in self.csv_files if typed in f.lower()]

    def apply_csv_action(self):
        display_name = self.combo_folder.get()
        scope = self.combo_scope.get()
        action = self.combo_action.get()
        if action == "Charger" and scope == "Tout":
            pass 

        if not display_name:
            return

        folder = display_name.rsplit(" ", 1)[0]
        folder_path = os.path.join(EXPORT_DIR, folder)
        files = list_csv_files(folder_path)
        if not files:
            return
        targets = files if scope=="Tout" else [self.combo_csv.get()]
        for file in targets:
            if not file:
                continue
            path = os.path.join(folder_path, file)
            try:
                if action=="Charger":
                    self.loaded_csv_path = path
                    self.add_csv_log(path)
                    self.load_csv()
                    self.run_prediction()
                elif action=="Archiver":
                    archive_dir = os.path.join(EXPORT_DIR,"Archive")
                    os.makedirs(archive_dir,exist_ok=True)
                    shutil.move(path, os.path.join(archive_dir,file))
                    
            except Exception:
                err = traceback.format_exc()
                messagebox.showerror("Erreur", err)
                save_error_to_csv(err)

        self.csv_files = list_csv_files(folder_path)
        self.combo_csv["values"] = self.csv_files
        self.combo_csv.set("")
        self.refresh_folders()
        for val in self.combo_folder["values"]:
            if val.startswith(folder+" "):
                self.combo_folder.set(val)
                break

    def save_team_score_to_csv(self, team: str):
        if not self.loaded_csv_path:
            return

        entry = getattr(self, f"entry_team{team}_score", None)
        if entry is None:
            return

        value = entry.get().strip()
        if value == "":
            return

        if not is_number(value):
            messagebox.showwarning(
                "Valeur invalide",
                f"Score équipe {team} invalide : {value}"
            )
            return

        col = 4 if team == "A" else 6

        try:
            df = pd.read_csv(
                self.loaded_csv_path,
                sep=None,
                engine="python",
                header=None,
                encoding="utf-8-sig"
            )

            while df.shape[1] <= col:
                df[df.shape[1]] = ""

            df.iat[0, col] = int(value)

            df.to_csv(
                self.loaded_csv_path,
                index=False,
                header=False,
                encoding="utf-8-sig"
            )
        except Exception:
            pass

    def load_csv(self):
        df = pd.read_csv(self.loaded_csv_path, sep=None, engine="python", header=None, encoding='utf-8-sig')
        self.teamA_scores = [str(v).strip() for v in df.iloc[:,0].tolist()]
        self.teamC_scores = [str(v).strip() for v in df.iloc[:,2].tolist()]

        teamA_name = next((str(v).strip() for v in reversed(self.teamA_scores) if v.strip()), "Equipe A")
        teamC_name = next((str(v).strip() for v in reversed(self.teamC_scores) if v.strip()), "Equipe C")
        self.entry_teamA.delete(0, tk.END)
        self.entry_teamA.insert(0, teamA_name)
        self.entry_teamC.delete(0, tk.END)
        self.entry_teamC.insert(0, teamC_name)

        try:
            self.entry_teamA_score.delete(0, tk.END)
            if df.shape[1] > 4 and pd.notna(df.iat[0, 4]):
                self.entry_teamA_score.insert(0, str(int(df.iat[0, 4])))

            self.entry_teamC_score.delete(0, tk.END)
            if df.shape[1] > 6 and pd.notna(df.iat[0, 6]):
                self.entry_teamC_score.insert(0, str(int(df.iat[0, 6])))
        except Exception:
            pass
        self.csv_loaded = True

    def on_apply_clicked(self):
        self.apply_csv_action()

        if self.csv_loaded:
            self.run_prediction()

    def build_prediction_table(self, results_per_column):

        row_keys = [
            ("Id","pred_results1"),
            ("Pr","pred_results2"),
            ("Mx","pred_results4"),
            ("Mn","pred_results5"),
            ("Mx2","pred_results6"),
            ("Mn2","pred_results7"),
            ("Tt","pred_results12"),
        ]

        # 1 colonne TYPE + 15 colonnes résultats
        headers_top = [" "] + ["0"]*5 + ["(-)"]*10
        headers_mid = [" "] + ["="]*15
        headers_low = [" "] + ["+"]*15

        cellw = 3

        def fmt(x):
            return f"{str(x):>{cellw}}"

        lines = []

        def build_row(values):
            return "|".join(fmt(v) for v in values) + "|"

        lines.append(build_row(headers_top))
        lines.append(build_row(headers_mid))
        lines.append(build_row(headers_low))

        lines.append("-" * len(lines[0]))

        for row_label, key in row_keys:

            row = [row_label]

            for col_results in results_per_column:

                count = col_results[key]["count"]
                prediction = col_results[key]["prediction"]

                if count == 0:
                    row.append("X")
                else:
                    row.append(classify_value(prediction))

            lines.append(build_row(row))

        return "\n".join(lines)

    def build_result_table(self, all_results):
        row_labels = ["1Id", "2Pr", "312", "4Mx", "5Mn", "634", "735"]

        key_map = {
            "1Id": "pred_results1",
            "2Pr": "pred_results2",
            "312": "pred_results4",
            "4Mx": "pred_results5",
            "5Mn": "pred_results6",
            "634": "pred_results7",
            "735": "pred_results12"
        }

        col_titles = [cfg[0] for cfg in self.motif_configs2]

        col_w = 4

        def cell(v):
            return str(v).center(col_w)

        lines = []

        #header = " " * col_w + "|" + "|".join(cell("G") for _ in col_titles) + "|"
        #lines.append(header)

        #header2 = " " * col_w + "|" + "|".join(cell("=") for _ in col_titles) + "|"
        #lines.append(header2)

        #sep = "-" * len(header)
        #lines.append(sep)

        for r in row_labels:
            row = r.ljust(col_w)
            for title in col_titles:
                results = all_results.get(title, {})
                key = key_map.get(r)

                if key not in results:
                    v = ""
                else:
                    count = results[key]["count"]
                    s = results[key]["sum"]
                    v = "X" if count == 0 else s

                row += "|" + cell(v)

            row += "|"
            lines.append(row)

        return "\n".join(lines)

    def display_motifs(self, series, motif, e_list, prop_superieur, prop_inferieur,
                       m_list, next_vals, rebounds, peaks, log_widget, label_prefix,
                       linear_rebound_func, peak_envelope_func):

            lin_pred = linear_rebound_func(next_vals)
            peak_pred = peak_envelope_func(next_vals)

            self.write_log(log_widget, f"\n📊 LR1 ⚽={lin_pred:.2f} | PE1 ⚽={peak_pred:.2f}\n")
            #self.write_log(log_widget, f"LR1, PE1 Now={next_vals}\n")
            
        
    def write_log(self, target, text):

        if hasattr(target, "insert"):
            target.insert(tk.END, text)
        else:
            target.write(text)

    def run_benchmark(self):
        if not self.csv_loaded:
            return

        filepath = "Benchmark_F4.txt"
        file_exists = os.path.isfile(filepath)

        full_series_A = read_numeric_after_marker(self.teamA_scores, "A")
        full_series_C = read_numeric_after_marker(self.teamC_scores, "C")

        if not full_series_A or not full_series_C:
            return

        with open(filepath, "a", encoding="utf-8") as f:

            if not file_exists:
                f.write("=========== BENCHMARK FLASH SCORE ===========\n\n")

            f.write(f"\n\n=========== {os.path.basename(self.loaded_csv_path)} ===========\n")

            def write_team_analysis(team_label, series):

                f.write(f"\n================ TEAM {team_label} ================\n")

                # ======================
                # ===== METHODE 1 ======
                # ======================
                f.write("\n===> 📊 METHODE 1 <===\n")

                all_results = []

                for title, ml, blocN in self.motif_configs:

                    results = self.fully_static_method_pattern1(
                        series,
                        motif_length=ml,
                        block_size=blocN,
                        motif_next_distance=self.motif_next_distance
                    )

                    all_results.append(results)

                table = self.build_prediction_table(all_results)
                f.write(table + "\n")

                # ======================
                # ===== METHODE 2 ======
                # ======================
                f.write("\n===> 📊 METHODE 2 <===\n")

                all_results = {}

                for title, ml, blocN in self.motif_configs2:

                    results = self.fully_static_method_pattern0(
                        series,
                        motif_length=ml,
                        block_size=blocN,
                        motif_next_distance=self.motif_next_distance
                    )

                    all_results[title] = results

                table = self.build_result_table(all_results)
                f.write(table + "\n")

                # ======================
                # ===== METHODE 3 ======
                # ======================
                f.write("\n===> 📊 METHODE 3 <===\n")

                linear_Id = []
                linear_Pr = []
                peak_Id = []
                peak_Pr = []

                for title, ml, blocN in self.motif_configs:

                    results = self.fully_static_method_pattern2(
                        series,
                        motif_length=ml,
                        block_size=blocN,
                        motif_next_distance=self.motif_next_distance
                    )

                    for key, arr in [("pred_results1", linear_Id), ("pred_results2", linear_Pr)]:
                        val = results[key]["linear_rebound"]
                        arr.append(classify_value(val))

                    for key, arr in [("pred_results1", peak_Id), ("pred_results2", peak_Pr)]:
                        val = results[key]["peak_envelope"]
                        arr.append(classify_value(val))

                def format_row(label, values):
                    return f"{label:<2}|" + "|".join(f"{v:^2}" for v in values) + "|"

                separator = "-" * (2 * len(self.motif_configs))

                f.write("Rebond linéaires:\n")
                f.write(f"{separator}\n")
                f.write(f"{format_row('Id', linear_Id)}\n")
                f.write(f"{format_row('Pr', linear_Pr)}\n")

                f.write("\nPeak envelope:\n")
                f.write(f"{separator}\n")
                f.write(f"{format_row('Id', peak_Id)}\n")
                f.write(f"{format_row('Pr', peak_Pr)}\n")

                # ======================
                # ===== GLOBAL LR/PE ===
                # ======================

                lin = self.linear_rebound_prediction(series)
                peak = self.peak_envelope_linear_prediction(series)

                f.write(
                    f"\n📊 LR ⚽={round(lin['prediction'],3)} | PE ⚽={round(peak['prediction'],3)}\n"
                )

                # ======================
                # ===== MOTIFS =========
                # ======================

                motif, data = self.fully_static_method_with_patterns(series, series)[0]

                e, p, pi, m, next_v, reb, peak_v = data

                f.write(f"\n📊 LR1 ⚽={reb} | PE1 ⚽={peak_v}\n")
                f.write(f"LR1, PE1 Now={next_v}\n")

                # ======================
                # ===== ZERO PATTERN ===
                # ======================

                zero_seq, zero_signal, last_zero = self.analyze_zero_pattern(series)

                if zero_seq:

                    f.write("\n📊 OVER/UNDER\n")

                    if len(zero_seq) < 3:
                        f.write("Comparaison impossible (<3 blocs)\n")
                    else:
                        if zero_signal == "Under":
                            f.write(f"⚽= Under {last_zero}\n")
                        elif zero_signal == "Over":
                            f.write(f"⚽= Over {last_zero}\n")

                # ======================
                # ===== LAST MEANS =====
                # ======================

                f.write("\n📊 LAST MEANS\n")

                for n in [5, 10, 15]:
                    if len(series) >= n:
                        avg = sum(series[-n:]) / n
                        f.write(f"AVG{n}={round(avg,3)}\n")

                # ======================
                # ===== BLOCKS =========
                # ======================

                f.write("\n📊 BLOCS AVEC MOYENNES ET GAP\n")

                blocks = self.compute_blocks_with_gaps(
                    series,
                    min_block=2,
                    gap_between=None
                )

                for block in blocks:

                    lin = block.get("linear_rebound",0)
                    peak = block.get("peak_envelope",0)
                    avg_now = block.get("average_remaining",0)
                    values_now = block.get("remaining_values",[])

                    f.write(
                        f"LR ⚽={lin:.3f} | PE ⚽={peak:.3f}\n"
                        f"AVG(now)={avg_now:.3f}\n"
                        f"VALUES(now)={values_now}\n"
                    )

                # ======================
                # ===== STATS ==========
                # ======================

                self.display_consecutive_stats(series, f)
                self.display_median_extrema_means(series, f, mode="min")
                self.display_median_extrema_means(series, f, mode="max")

            write_team_analysis("A", full_series_A)
            write_team_analysis("C", full_series_C)
            
        messagebox.showinfo("Benchmark", "Benchmark terminé ✔")

    def run_prediction(self, motif_length=2):
        
        full_series_A = read_numeric_after_marker(self.teamA_scores, "A")
        full_series_C = read_numeric_after_marker(self.teamC_scores, "C")

        seriesA = full_series_A
        seriesC = full_series_C

        resultA, resultC = self.fully_static_method_with_patterns(seriesA, seriesC)

        if not seriesA or not seriesC:
            msg = "Séries A ou C vides – arrêt de la prédiction."
            self.log_csv.insert(tk.END, "[WARNING] " + msg)
            self.log_csv.see(tk.END)
            return

        for log in (self.log_teamA, self.log_teamA_1, self.log_teamC, self.log_teamC_1):
            log.delete("1.0", tk.END)

        start_A = next((i+1 for i,v in enumerate(self.teamA_scores) if str(v).strip().upper()=="A"), 0)
        start_C = next((i+1 for i,v in enumerate(self.teamC_scores) if str(v).strip().upper()=="C"), 0)

        title, ml, blocN = self.motif_configs[0]
  
        # ====== log ======

        # --- fully_static_method_pattern1 ---
        self.write_log(self.log_teamA, "===> 📊 METHODE 1 <===\n")

        all_results_A = []

        for title, ml, blocN in self.motif_configs:

            results = self.fully_static_method_pattern1(
                seriesA,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            all_results_A.append(results)

        tableA = self.build_prediction_table(all_results_A)

        self.write_log(self.log_teamA, tableA + "\n")

        # --- fully_static_method_pattern1 ---
        self.write_log(self.log_teamC, "===> 📊 METHODE 1 <===\n")

        all_results_C = []

        for title, ml, blocN in self.motif_configs:

            results = self.fully_static_method_pattern1(
                seriesC,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            all_results_C.append(results)

        tableC = self.build_prediction_table(all_results_C)

        self.write_log(self.log_teamC, tableC + "\n")
        
        # =======================
        # ====== UPPER LOG ======
        # =======================
        # 
        # LOG TEAM A
        #      
        # --- fully_static_method_pattern2 ---
        self.write_log(self.log_teamA,  f"\n===> 📊 METHODE 3 <===\n")
        for title, ml, blocN in self.motif_configs:
            
            results = self.fully_static_method_pattern2(
                seriesA,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

        linear_Id_A = []
        linear_Pr_A = []
        peak_Id_A = []
        peak_Pr_A = []

        for title, ml, blocN in self.motif_configs:

            results = self.fully_static_method_pattern2(
                seriesA,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            for key, arr in [("pred_results1", linear_Id_A), ("pred_results2", linear_Pr_A)]:
                val = results[key]["linear_rebound"]
                arr.append(classify_value(val))

            for key, arr in [("pred_results1", peak_Id_A), ("pred_results2", peak_Pr_A)]:
                val = results[key]["peak_envelope"]
                arr.append(classify_value(val))


        def format_row(label, values):
            return f"{label:<2}|" + "|".join(f"{v:^2}" for v in values) + "|"

        separator = "-" * (2 * len(self.motif_configs))

        self.write_log(self.log_teamA,f"Rebond linéaires:")
        self.write_log(self.log_teamA,f"{separator}")
        self.write_log(self.log_teamA,f"\n{format_row("Id", linear_Id_A)}")
        self.write_log(self.log_teamA,f"\n{format_row("Pr", linear_Pr_A)}")

        self.write_log(self.log_teamA,f"\nPeak envelope:")
        self.write_log(self.log_teamA,f"{separator}")
        self.write_log(self.log_teamA,f"\n{format_row("Id", peak_Id_A)}")
        self.write_log(self.log_teamA,f"\n{format_row("Pr", peak_Pr_A)}")

        # =======================
        # ====== UPPER LOG ======
        # =======================
        # 
        # LOG TEAM C
        #      
        # --- fully_static_method_pattern2 ---
        self.write_log(self.log_teamC,  f"\n===> 📊 METHODE 3 <===\n")
        for title, ml, blocN in self.motif_configs:

            results = self.fully_static_method_pattern2(
                seriesC,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

        linear_Id_C = []
        linear_Pr_C = []
        peak_Id_C = []
        peak_Pr_C = []

        for title, ml, blocN in self.motif_configs:

            results = self.fully_static_method_pattern2(
                seriesC,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            for key, arr in [("pred_results1", linear_Id_C), ("pred_results2", linear_Pr_C)]:
                val = results[key]["linear_rebound"]
                arr.append(classify_value(val))

            for key, arr in [("pred_results1", peak_Id_C), ("pred_results2", peak_Pr_C)]:
                val = results[key]["peak_envelope"]
                arr.append(classify_value(val))

        # Fonction de formatage déjà définie
        separator = "-" * (2 * len(self.motif_configs))

        # --- Affichage TeamC ---
        self.write_log(self.log_teamC,f"Rebond linéaires:")
        self.write_log(self.log_teamC,f"{separator}")
        self.write_log(self.log_teamC,f"\n{format_row("Id", linear_Id_C)}")
        self.write_log(self.log_teamC,f"\n{format_row("Pr", linear_Pr_C)}")

        self.write_log(self.log_teamC,f"\nPeak envelope:")
        self.write_log(self.log_teamC,f"{separator}")
        self.write_log(self.log_teamC,f"\n{format_row("Id", peak_Id_C)}")
        self.write_log(self.log_teamC,f"\n{format_row("Pr", peak_Pr_C)}")

        # =======================
        # ====== LOWER LOG ======
        # =======================
        #
        # LOG TEAM A
        # 
        self.write_log(self.log_teamA_1, "=========== CORRECT SCORE SET ===========\n")

        # --- fully_static_method_pattern 0 ---
        self.write_log(self.log_teamA_1, f"\n===> GOAL PROB <===\n")
        
        all_results = {}

        for title, ml, blocN in self.motif_configs2:
            results = self.fully_static_method_pattern0(
                seriesA,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            all_results[title] = results

        table = self.build_result_table(all_results)
        self.write_log(self.log_teamA_1, table + "\n")
           
        # lin 1212
        linA = self.linear_rebound_prediction(seriesA)
        peakA = self.peak_envelope_linear_prediction(seriesA)
        self.write_log(
            self.log_teamA_1,
            f"\n📊 LR ⚽={round(linA['prediction'],3)} | PE ⚽={round(peakA['prediction'],3)}\n" 
        )

        # 
        motifA, dataA = resultA

        eA, pA, piA, mA, nextA, rebA, peakA = dataA

        self.display_motifs(
            seriesA, motifA, eA, pA, piA, mA, nextA, rebA, peakA,
            self.log_teamA_1, "motifA", self.linear_rebound, self.peak_envelope
        )
      
        # ===== ZERO PATTERN A =====
        zero_seq_A, zero_signal_A, last_zero_A = self.analyze_zero_pattern(seriesA)
        if zero_seq_A:
            self.write_log(self.log_teamA_1, "\n📊 GLOBAL PROB\n")

            if len(zero_seq_A) < 3:
                self.write_log(self.log_teamA_1, "Comparaison impossible (<3 blocs)\n")
            else:
                if zero_signal_A == "Under":
                    self.write_log(
                        self.log_teamA_1,
                        f"⚽= Under {last_zero_A}\n"
                    )
                elif zero_signal_A == "Over":
                    self.write_log(
                        self.log_teamA_1,
                        f"⚽= Over {last_zero_A}\n"
                    )
        
        # 5 10 15 means
        self.display_recent_averages(seriesA, self.log_teamA_1)
        
        # median_extrema
        self.display_median_extrema_means(seriesA, self.log_teamA_1, mode="min")
        self.display_median_extrema_means(seriesA, self.log_teamA_1, mode="max")

        # A revoir ====================================
        self.write_log(self.log_teamA_1, "\n=============================================================\n")

        # ===== BLOCS MOYENNES + GAP ===== 1212
        self.write_log(self.log_teamA_1, "\n📊 BLOCS AVEC MOYENNES ET GAP\n")

        blocks_results_A = self.compute_blocks_with_gaps(seriesA, min_block=2, gap_between=None, log_widget=self.log_teamA_1)

        for block_res in blocks_results_A:
            block_values = block_res.get("block_values", [])
            lin_pred = block_res.get("linear_rebound", 0)
            peak_pred = block_res.get("peak_envelope", 0)
            count = block_res.get("count", 0)
            remaining = block_res.get("remaining_values", [])
            avg_remaining = block_res.get("average_remaining", 0)

            self.write_log(
                self.log_teamA_1,
                f"LR ⚽={lin_pred:.3f} | PE ⚽={peak_pred:.3f}\n"
                f"AVG(now): {avg_remaining:.3f}\n"
                #f"VALUES(now): {remaining}\n"
            )

        # Consecutive 0 & 1...
        self.display_consecutive_stats(seriesA, self.log_teamA_1)

        # =======================
        # ====== LOWER LOG ======
        # =======================
        #
        # LOG TEAM C
        # 
        self.write_log(self.log_teamC_1, "=========== CORRECT SCORE SET ===========\n")

        # --- fully_static_method_pattern 0 ---
        self.write_log(self.log_teamC_1, f"\n===> GOAL PROB <===\n")
        
        all_results = {}

        for title, ml, blocN in self.motif_configs2:
            results = self.fully_static_method_pattern0(
                seriesC,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            all_results[title] = results

        table = self.build_result_table(all_results)
        self.write_log(self.log_teamC_1, table + "\n")
           
        # lin 1212
        linC = self.linear_rebound_prediction(seriesC)
        peakC = self.peak_envelope_linear_prediction(seriesC)
        self.write_log(
            self.log_teamC_1,
            f"\n📊 LR ⚽={round(linC['prediction'],3)} | PE ⚽={round(peakC['prediction'],3)}\n" 
        )
        
        # 
        motifC, dataC = resultC

        eC, pC, piC, mC, nextC, rebC, peakC = dataC

        self.display_motifs(
            seriesC, motifC, eC, pC, piC, mC, nextC, rebC, peakC,
            self.log_teamC_1, "motifC", self.linear_rebound, self.peak_envelope
        )
        
        # ===== ZERO PATTERN C =====
        zero_seq_C, zero_signal_C, last_zero_C = self.analyze_zero_pattern(seriesC)

        if zero_seq_C:
            self.write_log(self.log_teamC_1, "\n📊 GLOBAL PROB\n")

            if len(zero_seq_C) < 3:
                self.write_log(self.log_teamC_1, "Comparaison impossible (<3 blocs)\n")
            else:
                if zero_signal_C == "Under":
                    self.write_log(
                        self.log_teamC_1,
                        f"⚽= Under {last_zero_C}\n"
                    )
                elif zero_signal_C == "Over":
                    self.write_log(
                        self.log_teamC_1,
                        f"⚽= Over {last_zero_C}\n"
                    )
        
        # 5 10 15 means
        self.display_recent_averages(seriesC, self.log_teamC_1)
           
        # median_extrema        
        self.display_median_extrema_means(seriesC, self.log_teamC_1, mode="min")
        self.display_median_extrema_means(seriesC, self.log_teamC_1, mode="max")

        # A revoir ====================================
        self.write_log(self.log_teamC_1, "\n=============================================================\n")

        # ===== BLOCS MOYENNES + GAP ===== 1212
        self.write_log(self.log_teamC_1, "\n📊 BLOCS AVEC MOYENNES ET GAP\n")

        blocks_results_C = self.compute_blocks_with_gaps(seriesC, min_block=2, gap_between=None, log_widget=self.log_teamC_1)

        for block_res in blocks_results_C:
            block_values = block_res.get("block_values", [])
            lin_pred = block_res.get("linear_rebound", 0)
            peak_pred = block_res.get("peak_envelope", 0)
            count = block_res.get("count", 0)
            remaining = block_res.get("remaining_values", [])
            avg_remaining = block_res.get("average_remaining", 0)

            self.write_log(
                self.log_teamC_1,
                f"LR ⚽={lin_pred:.3f} | PE ⚽={peak_pred:.3f}\n"
                f"AVG(now): {avg_remaining:.3f}\n"
                #f"VALUES(now): {remaining}\n"
            )

        # Consecutive 0 & 1...
        self.display_consecutive_stats(seriesC, self.log_teamC_1)        
    def prev_csv(self):
        if self.current_index>0:
            self.current_index-=1
            self.loaded_csv_path=self.csv_log[self.current_index]
            self.load_csv()
            self.run_prediction()

    def next_csv(self):
        if self.current_index<len(self.csv_log)-1:
            self.current_index+=1
            self.loaded_csv_path=self.csv_log[self.current_index]
            self.load_csv()
            self.run_prediction()

    def on_close(self):
        self.destroy()
    
if __name__ == "__main__":

    os.makedirs(EXPORT_DIR, exist_ok=True)
    app = FlashscoreApp()
    app.mainloop()
