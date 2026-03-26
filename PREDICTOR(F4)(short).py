#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import traceback,csv,os,shutil,subprocess,sys,math
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import pandas as pd
import tkinter as tk
from tkinter import ttk,messagebox,simpledialog
from collections import Counter
from itertools import combinations

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

def is_number(v)->bool:
    try:int(v);return True
    except: return False

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

def read_numeric_after_marker(column:list,marker:str)->list:
    start=next((i+1 for i,v in enumerate(column)
        if str(v).strip().upper()==marker),None)
    if start is None:return []
    return [int(v) for v in column[start:] if is_number(v)]

class FlashscoreApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PREDICTOR(F4)")
        self.geometry("1400x1000")
        self.teamA_scores:List[str]=[]
        self.teamC_scores:List[str]=[]
        self.motif_next_distance=0

        self.motif_configs=[
            ("0= OVER",2,30),("0= OVER",3,30),("0= OVER",4,30),("0= OVER",5,30),
            ("0= OVER",6,30),("0= OVER",2,60),("(-)= OVER",3,60),("(-)= OVER",4,60),
            ("(-)= OVER",5,60),("(-)= OVER",6,60),("(-)= OVER",2,10),("(-)= OVER",2,20),
            ("(-)= OVER",2,40),("(-)= OVER",2,50),("(-)= OVER",2,70),
        ]

        self.motif_configs2=[
            ("1.B",2,3),("2.T",3,4),("3.T",4,5),("4.T",5,6),("5.L",3,6),
            ("6.B",6,7),("7.T",7,8),("8.T",8,9),("9.T",9,10),("0.T",10,11),
        ]

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
            
    def motif_engine_complete2(self, series, tol=0.15):
        results = {}
        if len(series) < 10: return results

        deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
        motif_lengths = getattr(self, "motif_lengths_override", [3, 4, 5, 6])

        def is_proportional(a, b, tol=0.15):
            ratios = [x / y for x, y in zip(a, b) if y != 0]
            if not ratios: return False
            r0 = sum(ratios) / len(ratios)
            return all(abs(r - r0) <= tol * abs(r0) for r in ratios)

        used_values = set()  
        used_motifs = set()  

        for L in motif_lengths:
            if len(deltas) <= L: continue
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
                count_equal = count_prop = 0
                predicted_value = None

                for i in range(len(deltas) - L - 1):
                    window = [1 if d > 0 else -1 if d < 0 else 0 for d in deltas[i:i+L]] if label=="_SIGNES" else deltas[i:i+L]
                    if window == motif:
                        count_equal += 1
                        predicted_value = series[i + L + 1]
                    elif is_proportional(window, motif, tol):
                        count_prop += 1
                        predicted_value = series[i + L + 1]

                if predicted_value is not None and predicted_value not in used_values:
                    key = f"{label}_L{L}"
                    results[key] = {"value": predicted_value, "equal": count_equal, "prop": count_prop, "L": L}
                    used_values.add(predicted_value)
                    used_motifs.add(motif_tuple)

        if not results:
            results["__REPLI"] = {"value": series[-1] + deltas[-1], "equal": 0, "prop": 0, "L": None}

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
        results = {}; n=len(series)
        def is_proportional(a,b,tol=0.15):
            ratios=[x/y for x,y in zip(a,b) if y!=0]
            if not ratios: return False
            r0=sum(ratios)/len(ratios)
            return all(abs(r-r0)<=tol*abs(r0) for r in ratios)
        def cr2_difference(next_values,last_value,prev=None):
            first=next_values[0]; cr_val=first-last_value
            ratio_str=f"{first/prev[0]:.2f}" if prev and prev[0]!=0 else f"({first})"
            desc=f"Cr=Δ (ratio={ratio_str})"
            if prev: desc+=f" ΔR={int(round(first-prev[0]))}"
            return last_value+cr_val, desc
        for L in (3,4,5):
            if n<L+2: continue
            motif_C=series[-L:]; last_value=motif_C[-1]
            equal_next=[]; prop_next_values=[]
            for i in range(n-L-1):
                motif_H=series[i:i+L]; next_val=series[i+L]
                if motif_H==motif_C: equal_next.append(next_val)
                elif is_proportional(motif_H,motif_C,tol): prop_next_values.append(next_val)
            if equal_next:
                prediction_value=equal_next[-1]; cr2_used=False; cr2_value=None; cr2_description=""
            elif len(prop_next_values)>=2:
                prediction_value,cr2_description=cr2_difference(prop_next_values,last_value)
                cr2_used=True; cr2_value=prediction_value-last_value
            else: continue
            results[f"L{L}"]={"value":prediction_value,"Cr2":cr2_used,"cr2_value":cr2_value,
                               "cr2_description":cr2_description,"equal":len(equal_next),
                               "prop":len(prop_next_values),"L":L}
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

    def detect_target_motif_prediction_disorder(self,series,motif_length,block_size,motif_type):
        if motif_type is None: motif_type="croissant"
        if len(series)<block_size: return {"motif_cible":None,"similar_blocks":[],"prediction":None}
        bloc_cible=series[-block_size:]; bloc15=bloc_cible[:-1]; motif_main=[]
        if motif_type in ["croissant","decroissant"]:
            for start in range(len(bloc15)-motif_length+1):
                candidate=bloc15[start:start+motif_length]
                if motif_type=="croissant" and all(candidate[i]<candidate[i+1] for i in range(motif_length-1)):
                    motif_main=candidate; break
                elif motif_type=="decroissant" and all(candidate[i]>candidate[i+1] for i in range(motif_length-1)):
                    motif_main=candidate; break
        elif motif_type=="proportionnel":
            for start in range(len(bloc15)-motif_length+1):
                candidate=bloc15[start:start+motif_length]; diffs=[candidate[i+1]-candidate[i] for i in range(len(candidate)-1)]
                if all(diffs[i]*diffs[0]>0 for i in range(1,len(diffs))):
                    ratios=[diffs[i]/diffs[0] for i in range(1,len(diffs))]
                    if all(abs(r-1.0)<1e-6 for r in ratios): motif_main=candidate; break
        if not motif_main: return {"motif_cible":None,"similar_blocks":[],"prediction":None}
        similar_blocks=[]; delta_cible=[motif_main[i+1]-motif_main[i] for i in range(len(motif_main)-1)]
        for i in range(len(series)-block_size):
            bloc=series[i:i+block_size]; bloc15_histo=bloc[:-1]
            var_values=bloc15_histo[-motif_length:]; delta_bloc=[var_values[j+1]-var_values[j] for j in range(len(var_values)-1)]
            if motif_type=="proportionnel":
                if all(d!=0 for d in delta_cible) and all(d!=0 for d in delta_bloc):
                    ratios=[delta_bloc[j]/delta_cible[j] for j in range(len(delta_cible))]
                    if max(ratios)-min(ratios)<1e-6: similar_blocks.append(bloc)
            elif delta_bloc==delta_cible: similar_blocks.append(bloc)
        if similar_blocks:
            followers=[bloc[-1] for bloc in similar_blocks]; n=len(followers)
            if n%2==1: center=n//2; weights=[i-center for i in range(n)]
            else: mid_left=n//2-1; mid_right=n//2; weights=[i-mid_left if i<=mid_left else i-mid_right for i in range(n)]
            evolution=sum(followers[i]*weights[i] for i in range(n))/n
            prediction=motif_main[-1]+evolution
        else: prediction=None
        return {"motif_cible":motif_main,"similar_blocks":similar_blocks,"prediction":prediction}

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

    def display_recent_averages(self,series,log_widget):
        series_data=series[:-1];lengths=[5,10,15];avg=[]
        for L in lengths:
            avg.append(f"Last{L}={sum(series_data[-L:])/L:.2f}" if len(series_data)>=L else f"Last{L}=N/A")
        self.write_log(log_widget,f"\n📊 LAST MEANS : {' | '.join(avg)}\n")

    def fully_static_method_pattern0(self, series, motif_length, block_size, motif_next_distance=0):
        series_data = series[:-1]
        base_pred = series[-1]
        n = len(series_data)
        last_bloc = series_data[-block_size:]
        max_val = max(last_bloc)
        min_val = min(last_bloc)

        target_motif = last_bloc[:motif_length]

        def weighted_linear_regression(values):
            n_vals = len(values)
            if n_vals < 2: return sum(values)
            x = list(range(n_vals))
            y = values
            w = [n_vals - i for i in range(n_vals)]
            sum_w = sum(w)
            sum_wx = sum(w[i] * x[i] for i in range(n_vals))
            sum_wy = sum(w[i] * y[i] for i in range(n_vals))
            sum_wxx = sum(w[i] * x[i] * x[i] for i in range(n_vals))
            sum_wxy = sum(w[i] * x[i] * y[i] for i in range(n_vals))
            denom = sum_w * sum_wxx - sum_wx * sum_wx
            if denom == 0: return sum(values)
            slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
            intercept = (sum_wy - slope * sum_wx) / sum_w
            return intercept + slope * n_vals

        def remove_farthest_occurrence(value):
            idx = [i for i, v in enumerate(last_bloc) if v == value]
            if not idx: return None
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
            if i + block_size < n:
                search_zones.append((i, sim_bloc, series_data[i + block_size]))

        store = {f"pred_results{i}": [] for i in range(1, 8)}

        def ordered_match(sim_bloc, motif):
            pos = 0
            for v in sim_bloc:
                if pos < len(motif) and v == motif[pos]:
                    pos += 1
            return pos == len(motif)

        def delta_calc(i, corres_next):
            return max(0, corres_next - i)

        for i, sim_bloc, corres_next in search_zones:
            delta = delta_calc(i, corres_next)
            matched = False

            if ordered_match(sim_bloc, target_motif):
                store["pred_results1"].append(delta)
                matched = True

            if not matched:
                sum_target = sum(target_motif)
                ratio = sum(sim_bloc) / sum_target if sum_target != 0 else 1
                if ratio > 1 or (0 < ratio < 1):
                    store["pred_results2"].append(delta)
                    matched = True

            if motif_excl_max and ordered_match(sim_bloc, motif_excl_max):
                store["pred_results4"].append(delta)
            if motif_excl_min and ordered_match(sim_bloc, motif_excl_min):
                store["pred_results5"].append(delta)
            if motif_excl_max and ordered_match(sim_bloc, motif_excl_max) and delta not in store["pred_results4"]:
                store["pred_results6"].append(delta)
            if motif_excl_min and ordered_match(sim_bloc, motif_excl_min) and delta not in store["pred_results5"]:
                store["pred_results7"].append(delta)

        pred_results3 = store["pred_results1"] + store["pred_results2"]
        pred_results12 = pred_results3 + store["pred_results4"] + store["pred_results5"] + store["pred_results6"] + store["pred_results7"]
        combined_map = {3: pred_results3, 12: pred_results12}

        def stats(lst):
            if not lst:
                return {"prediction": 0, "count": 0}
            return {"prediction": weighted_linear_regression(lst), "count": len(lst)}

        final_results = {}
        for i in range(1, 8):
            final_results[f"pred_results{i}"] = stats(store[f"pred_results{i}"])
        for k, v in combined_map.items():
            final_results[f"pred_results{k}"] = stats(v)

        return final_results

    def fully_static_method_pattern1(self, series, motif_length, block_size, motif_next_distance, used_global=None):
        if used_global is None:
            used_global = set()

        series_data = series[:-1]
        n = len(series_data)
        last_bloc = series_data[-block_size:]
        base_pred = series[-1]

        target_motif = last_bloc[:motif_length]

        sim_blocs_list = []
        corres_next_list = []
        for i in range(0, n - block_size, block_size):
            sim_blocs_list.append(series_data[i:i + block_size])
            corres_next_list.append(series_data[i + block_size:i + block_size + 1])

        pr1, pr2, pr4, pr5 = [], [], [], []
        pr3, pr6, pr7 = [], [], []

        used_patterns_12 = set()
        used_patterns_45 = set()

        def check(sim, target):
            for i in range(len(sim) - len(target) + 1):
                if sim[i:i+len(target)] == target:
                    return True
            return False

        target_mean = sum(target_motif) / motif_length if motif_length > 0 else 0

        for i, sim in enumerate(sim_blocs_list):
            corres = corres_next_list[i][0] if corres_next_list[i] else None
            if corres is None:
                continue

            motif_id = tuple(sim)

            if motif_id in used_global:
                continue

            d = abs(corres - base_pred)

            if check(sim, target_motif):
                if d == motif_next_distance:
                    pr1.append((motif_id, d))
                    used_patterns_12.add(motif_id)
                    used_global.add(motif_id)
            else:
                sim_mean = sum(sim) / len(sim) if len(sim) > 0 else 0
                ratio = sim_mean / target_mean if target_mean != 0 else 1

                if ratio > 1 or (0 < ratio < 1):
                    pr2.append((motif_id, d))
                    used_patterns_12.add(motif_id)
                    used_global.add(motif_id)

        for sim in sim_blocs_list:
            motif_id = tuple(sim)

            if motif_id in used_global:
                continue

            if max(sim) > target_mean:
                pr4.append((motif_id, max(sim)))
                used_patterns_45.add(motif_id)
                used_global.add(motif_id)
            elif min(sim) < target_mean:
                pr5.append((motif_id, min(sim)))
                used_patterns_45.add(motif_id)
                used_global.add(motif_id)

        candidate_mx2, candidate_mn2 = [], []

        for sim in sim_blocs_list:
            motif_id = tuple(sim)

            if motif_id in used_global:
                continue

            if motif_id in used_patterns_45:
                continue

            if max(sim) > target_mean:
                candidate_mx2.append((motif_id, max(sim)))
            elif min(sim) < target_mean:
                candidate_mn2.append((motif_id, min(sim)))

        pr6 = [d for _, d in candidate_mx2]
        pr7 = [d for _, d in candidate_mn2]

        pr3 = [d for _, d in pr1 + pr2]

        def make_result(arr):
            count = len(arr)
            s = sum(arr)
            pred = s / count if count > 0 else 0
            return {"sum": s, "count": count, "prediction": pred}

        return {
            "pred_results1": make_result([d for _, d in pr1]),
            "pred_results2": make_result([d for _, d in pr2]),
            "pred_results3": make_result(pr3),
            "pred_results4": make_result([d for _, d in pr4]),
            "pred_results5": make_result([d for _, d in pr5]),
            "pred_results6": make_result(pr6),
            "pred_results7": make_result(pr7),
        }

    def fully_static_method_pattern11(self, series, motif_length, block_size, motif_next_distance, used_global=None):
        if used_global is None:
            used_global = set()

        series_data = series[:-1]
        n = len(series_data)
        if n < block_size:
            return {f"pred_results{i}": {"sum": 0, "count": 0, "prediction": 0} for i in range(1, 8)}

        last_bloc = series_data[-block_size:]
        base_pred = series[-1]

        target_motif = last_bloc[:motif_length]

        sim_blocs_list = []
        corres_next_list = []
        for i in range(0, n - block_size, block_size):
            sim_blocs_list.append(series_data[i:i + block_size])
            corres_next_list.append(series_data[i + block_size:i + block_size + 1])

        pr1, pr2, pr4, pr5 = [], [], [], []
        used_patterns_12 = set()
        used_patterns_45 = set()

        def check(sim, target):
            for i in range(len(sim) - len(target) + 1):
                if sim[i:i+len(target)] == target:
                    return True
            return False

        target_mean = sum(target_motif) / motif_length if motif_length > 0 else 0

        for i, sim in enumerate(sim_blocs_list):
            corres = corres_next_list[i][0] if corres_next_list[i] else None
            if corres is None:
                continue

            motif_id = tuple(sim)
            if motif_id in used_global:
                continue

            d = abs(corres - base_pred)

            if check(sim, target_motif):
                if d == motif_next_distance:
                    pr1.append(d)
                    used_patterns_12.add(motif_id)
                    used_global.add(motif_id)
            else:
                sim_mean = sum(sim) / len(sim) if len(sim) > 0 else 0
                ratio = sim_mean / target_mean if target_mean != 0 else 1
                if ratio != 1:
                    pr2.append(sim_mean) 
                    used_patterns_12.add(motif_id)
                    used_global.add(motif_id)

        for sim in sim_blocs_list:
            motif_id = tuple(sim)
            if motif_id in used_global or motif_id in used_patterns_12:
                continue

            if max(sim) > target_mean:
                pr4.append(max(sim))
                used_patterns_45.add(motif_id)
                used_global.add(motif_id)
            elif min(sim) < target_mean:
                pr5.append(min(sim))
                used_patterns_45.add(motif_id)
                used_global.add(motif_id)

        candidate_mx2 = [max(sim) for sim in sim_blocs_list
                         if tuple(sim) not in used_global and tuple(sim) not in used_patterns_45 and max(sim) > target_mean]
        candidate_mn2 = [min(sim) for sim in sim_blocs_list
                         if tuple(sim) not in used_global and tuple(sim) not in used_patterns_45 and min(sim) < target_mean]

        pr3 = pr1 + pr2
        pr6 = candidate_mx2
        pr7 = candidate_mn2

        def make_result(arr, use_mean=False):
            count = len(arr)
            s = int(round(sum(arr))) if arr else 0
            pred = int(round(sum(arr) / count)) if use_mean and count > 0 else s
            return {"sum": s, "count": count, "prediction": pred}

        return {
            "pred_results1": make_result(pr1),
            "pred_results2": make_result(pr2, use_mean=True),
            "pred_results3": make_result(pr3),
            "pred_results4": make_result(pr4),
            "pred_results5": make_result(pr5),
            "pred_results6": make_result(pr6),
            "pred_results7": make_result(pr7),
        }

    def fully_static_method_pattern2(self, series, motif_length, block_size, motif_next_distance=0):
        series_data = series[:-1]
        n = len(series_data)
        last_bloc = series_data[-block_size:]
        min_val, max_val = min(last_bloc), max(last_bloc)

        target_motif = last_bloc[:motif_length]

        def remove_far(value):
            idx = [i for i, v in enumerate(last_bloc) if v == value]
            if not idx: return None
            motif = target_motif.copy()
            if value in motif:
                motif.remove(value)
                return motif
            return None

        motif_excl_max = remove_far(max_val)
        motif_excl_min = remove_far(min_val)

        search = []
        for i in range(0, n - block_size, block_size):
            sim = series_data[i:i + block_size]
            if i + block_size < n:
                search.append((i, sim, series_data[i + block_size]))

        store = {f"pred_results{i}": [] for i in range(1, 8)}

        def ordered(sim, motif):
            pos = 0
            for v in sim:
                if pos < len(motif) and v == motif[pos]:
                    pos += 1
            return pos == len(motif)

        def delta(i, nxt):
            return max(0, nxt - i)

        for i, sim, nxt in search:
            d = delta(i, nxt)
            matched = False

            if ordered(sim, target_motif):
                store["pred_results1"].append(d)
                matched = True

            if not matched:
                ratio = sum(sim) / sum(target_motif) if sum(target_motif) != 0 else 1
                if ratio > 1 or (0 < ratio < 1):
                    store["pred_results2"].append(d)
                    matched = True

            if motif_excl_max and ordered(sim, motif_excl_max):
                store["pred_results4"].append(d)
            if motif_excl_min and ordered(sim, motif_excl_min):
                store["pred_results5"].append(d)
            if motif_excl_max and ordered(sim, motif_excl_max) and d not in store["pred_results4"]:
                store["pred_results6"].append(d)
            if motif_excl_min and ordered(sim, motif_excl_min) and d not in store["pred_results5"]:
                store["pred_results7"].append(d)

        def stats(lst):
            if not lst:
                return {"linear_rebound": 0, "peak_envelope": 0, "count": 0}
            lin = self.linear_rebound_prediction(lst)
            peak = self.peak_envelope_linear_prediction(lst)
            return {"linear_rebound": lin["prediction"], "peak_envelope": peak["prediction"], "count": len(lst)}

        final = {}
        for i in range(1, 8):
            final[f"pred_results{i}"] = stats(store[f"pred_results{i}"])

        combined = store["pred_results1"] + store["pred_results2"] + store["pred_results4"] + store["pred_results5"] + store["pred_results6"] + store["pred_results7"]
        final["pred_results12"] = stats(combined)

        return final

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

    def peak_envelope_linear_prediction(self, series, min_peaks=4):
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

    def display_consecutive_stats(self,series,log_widget):
        data=series[:-1]
        if not data:return
        last=data[-1];count=0;i=len(data)-1
        while i>=0 and data[i]==last:count+=1;i-=1
        zero_last=count if last==0 else 0
        one_last=count if last==1 else 0
        self.write_log(log_widget,f"Most recent: 0={zero_last}* | 1={one_last}*\n")

        last15=data[-15:] if len(data)>=15 else data
        max0=max1=cur0=cur1=0
        for v in last15:
            if v==0:cur0+=1;max0=max(max0,cur0);cur1=0
            elif v==1:cur1+=1;max1=max(max1,cur1);cur0=0
            else:cur0=cur1=0
        self.write_log(log_widget,f"In 15 last: 0={max0}* | 1={max1}*\n")
        
    def compute_average_remaining(self, series):
        data = series[:-1]
        if not data:
            return 0
        return sum(data) / len(data)

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

    def display_median_extrema_means(self,series,log_widget,mode="min"):
        data=series[:-1]
        if not data:return
        configs=[(7,15),(7,27),(7,51),(13,27),(13,51),(13,99),(25,51),(25,99)]

        if mode=="min":
            self.write_log(log_widget,"\n📊 Median minima 'N values mean' in 'N lasts':\n");func=min
        elif mode=="max":
            self.write_log(log_widget,"\n📊 Median maxima 'N values mean' in 'N lasts':\n");func=max
        else:raise ValueError("mode must be 'min' or 'max'")

        for mean_size,zone_size in configs:
            if len(data)<zone_size:
                self.write_log(log_widget,f"{mean_size} in {zone_size} : N/A\n");continue
            zone=data[-zone_size:]
            val=func(zone);idx=zone.index(val)
            half=mean_size//2
            start=max(0,idx-half);end=start+mean_size
            if end>len(zone):end=len(zone);start=end-mean_size
            subset=zone[start:end]
            if len(subset)==mean_size:
                avg=sum(subset)/mean_size
                self.write_log(log_widget,f"{mean_size} in {zone_size} : {avg:.2f}\n")
            else:self.write_log(log_widget,f"{mean_size} in {zone_size} : N/A\n")

    def compute_blocks_with_gaps(self,series,min_block=2,gap_between=None,log_widget=None):
        data=series[:-1];n=len(data);i=n-1
        results=[];block_avgs=[]
        while i>=min_block:
            target=data[i];found=False
            for j in range(i-min_block,-1,-1):
                block=data[j:i+1]
                if len(block)>=min_block+1 and sum(block)/len(block)==target:
                    block_avgs.append(target);i=j-1;found=True;break
            if not found:i-=1

        if block_avgs:
            lin=self.linear_rebound_prediction(block_avgs)
            peak=self.peak_envelope_linear_prediction(block_avgs)
            start=n-len(block_avgs)
            remaining=data[start:] if start<n else []
            avg_rem=sum(remaining)/len(remaining) if remaining else 0
            results.append({
                "block_values":block_avgs,
                "linear_rebound":lin["prediction"],
                "peak_envelope":peak["prediction"],
                "count":lin["count"],
                "remaining_values":remaining,
                "average_remaining":avg_rem
            })
        return results

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
        self.display_motifs(seriesA,motifA,eA,pA,piA,mA,nextA,rebA,peakA,
            self.log_teamA_1,"motifA",self.linear_rebound,self.peak_envelope)

        motifC,dataC=resultC
        eC,pC,piC,mC,nextC,rebC,peakC=dataC
        self.display_motifs(seriesC,motifC,eC,pC,piC,mC,nextC,rebC,peakC,
            self.log_teamC_1,"motifC",self.linear_rebound,self.peak_envelope)

        return resultA,resultC

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

    def detect_over_probable(self, series, log_widget):
        data = series[:-1]  
        n = len(data)
        if n < 3:
            return

        windows = [3, 15, 30, 45]
        avg_results = []

        for w in windows:
            if n < w:
                continue
            last_w = data[-w:]
            avg_w = sum(last_w) / w
            avg_results.append(avg_w)

        start = 45
        while start < n:
            end = min(start + 45, n)
            window_data = data[-end:-start] if start != 0 else data[-end:]
            if window_data:
                avg_window = sum(window_data) / len(window_data)
                avg_results.append(avg_window)
            start += 45

        trend_window = min(10, n)
        recent = data[-trend_window:]
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        trend_score = sum(d for d in diffs if d < 0)  
        extreme_fall = trend_score <= -0.5 * trend_window 

        overall_avg = sum(avg_results) / len(avg_results) if avg_results else 0
        if overall_avg < 0.25 or extreme_fall:
            self.write_log(log_widget, "⚽= LOT GOALS(PROB)\n")

    def check_over_probability_hybrid(self, series, log_widget):
        data = series[:-1]  
        n = len(data)
        if n < 3:
            return

        windows = [3, 15, 30, 45]
        results = []

        for w in windows:
            if n < w:
                continue
            last_w = data[-w:]
            avg_w = sum(last_w) / w
            results.append(avg_w)

        start = 45
        while start < n:
            end = min(start + 45, n)
            window_data = data[-end:-start] if start != 0 else data[-end:]
            if window_data:
                avg_window = sum(window_data) / len(window_data)
                results.append(avg_window)
            start += 45

        overall_score = sum(results) / len(results) if results else 0

        last = data[-1]
        prev1 = data[-2] if n >= 2 else 0
        prev2 = data[-3] if n >= 3 else 0
        cond_last = last == 0 and prev1 < 3 and prev2 < 3

        threshold = 0.25
        if overall_score < threshold or cond_last:
            self.write_log(log_widget, "⚽= LOT GOALS(HYBRID)\n")

    def compute_over_under_full(self, series, log_widget):
        data = series[:-1]
        n = len(data)

        if n < 5:
            return

        last_value = data[-1]

        def avg_last(k):
            return sum(data[-k:]) / k if n >= k else None

        avg5  = avg_last(5)
        avg10 = avg_last(10)
        avg15 = avg_last(15)
        avg20 = avg_last(20)
        avg25 = avg_last(25)

        count = 0
        if avg5  is not None and last_value > avg5:
            count += -2
        if avg10 is not None and last_value > avg10:
            count += -1
        if avg15 is not None and last_value > avg15:
            count += 0
        if avg20 is not None and last_value > avg20:
            count += 1
        if avg25 is not None and last_value > avg25:
            count += 2

        count2 = 0
        if avg5  is not None and last_value < avg5:
            count2 += 2
        if avg10 is not None and last_value < avg10:
            count2 += 1
        if avg15 is not None and last_value < avg15:
            count2 += 0
        if avg20 is not None and last_value < avg20:
            count2 += -1
        if avg25 is not None and last_value < avg25:
            count2 += -2

        countT = count + count2

        def label_with_value(v):
            if v > 0:
                return f"({v}) ⚽ + de "
            elif v < 0:
                return f"({v}) ⚽ - de "
            else:
                return f"({v}) ⚽ = à "

        def label_simple(v):
            if v > 0:
                return "⚽ + de "
            elif v < 0:
                return "⚽ - de "
            else:
                return "⚽ = à "

        self.write_log(
            log_widget,
            f"\n📊 OVER/ UNDER/ NEUTRAL:\n"
            f"{label_with_value(countT)}{last_value} or LAST MEANS\n"
        )

        self.write_log(
            log_widget,
            f"{label_simple(countT)}{last_value} or LAST MEANS\n"
        )

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
        log1=tk.Text(c1,height=15,width=60);log1.pack(side="left",fill="both",expand=True)
        sb1=ttk.Scrollbar(c1,orient="vertical",command=log1.yview);sb1.pack(side="right",fill="y")
        log1.config(yscrollcommand=sb1.set)
        setattr(self,f"log_team{team}",log1)

        c2=ttk.Frame(frame);c2.pack(fill="both",expand=True)
        log2=tk.Text(c2,height=35,width=60);log2.pack(side="left",fill="both",expand=True)
        sb2=ttk.Scrollbar(c2,orient="vertical",command=log2.yview);sb2.pack(side="right",fill="y")
        log2.config(yscrollcommand=sb2.set)
        setattr(self,f"log_team{team}_1",log2)

    def _build_log_frame(self,parent,controls):
        container=ttk.Frame(parent)
        container.grid(row=0,column=2,sticky="nsew",padx=(6,2))
        container.rowconfigure(3,weight=1)
        container.columnconfigure(0,weight=1)

        btns=ttk.Frame(container);btns.grid(row=0,column=0,sticky="ew",pady=(0,4))
        ttk.Button(btns,text="HIDE",command=self.hide_selected_csv).pack(side="left",padx=4)
        ttk.Button(btns,text="ARCHIVE",command=self.archive_selected_csv).pack(side="left",padx=4)

        ttk.Button(container,text="CLEAR LOG",
            command=self.reset_csv_log).grid(row=1,column=0,sticky="w",pady=(0,4))

        nav=ttk.Frame(container)
        nav.grid(row=2,column=0,sticky="w",pady=(0,4))
        ttk.Button(nav,text="PREV.",command=self.prev_csv).pack(side="left",padx=2)
        ttk.Label(nav,text="<").pack(side="left",padx=(0,6))
        ttk.Label(nav,text=">").pack(side="left",padx=(6,0))
        ttk.Button(nav,text="NEXT.",command=self.next_csv).pack(side="left",padx=2)

        self.log_csv=tk.Listbox(container,selectmode="extended",height=25)
        self.log_csv.grid(row=3,column=0,sticky="nsew")

        sb=ttk.Scrollbar(container,command=self.log_csv.yview)
        sb.grid(row=3,column=1,sticky="ns")
        self.log_csv.config(yscrollcommand=sb.set)

        self.log_csv.bind("<<ListboxSelect>>",self.on_csv_select)
        self.refresh_csv_log()

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

    def build_prediction_table(self,results_per_column):
        row_keys=[("Id","pred_results1"),("Pr","pred_results2"),("Mx","pred_results4"),
                  ("Mn","pred_results5"),("Mx2","pred_results6"),("Mn2","pred_results7")]

        num_cols = len(results_per_column)
        headers = [" "] + [str(i+1) for i in range(num_cols)]

        cellw = 3
        fmt = lambda x: f"{str(x):>{cellw}}"

        def build_row(vals):
            return "|".join(fmt(v) for v in vals) + "|"

        lines = [build_row(headers)]
        lines.append("-" * len(lines[0]))

        for label, key in row_keys:
            row = [label]
            for col in results_per_column:
                count = col[key]["count"]
                pred = col[key]["prediction"]
                if count == 0:
                    row.append("X") 
                else:
                    row.append(classify_value(pred))

            lines.append(build_row(row))

        return "\n".join(lines)

    def build_result_table(self, all_results):
        rows = ["1Id", "2Pr", "4Mx", "5Mn"]
        key_map = {
            "1Id": "pred_results1",
            "2Pr": "pred_results2",
            "4Mx": "pred_results4",
            "5Mn": "pred_results5"
        }

        cols = [cfg[0] for cfg in self.motif_configs2]
        col_w = 3
        cell = lambda v: str(v).center(col_w)
        lines = []

        header = "".ljust(col_w) + "".join("|" + cell(c) for c in cols) + "|"
        lines.append(header)

        for r in rows:
            row = r.ljust(col_w)
            for title in cols:
                res = all_results.get(title, {})
                key = key_map.get(r)
                if not key or key not in res:
                    v = ""
                else:
                    entry = res.get(key, {})
                    count = entry.get("count", 0)
                    prediction = entry.get("prediction", 0)
                    prediction_rounded = int(round(prediction)) if count > 0 else "X"
                    v = prediction_rounded
                row += "|" + cell(v)
            row += "|"
            lines.append(row)
        return "\n".join(lines)


    def display_motifs(self, series, motif, e_list, prop_superieur, prop_inferieur,
                       m_list, next_vals, rebounds, peaks, target, label_prefix,
                       linear_rebound_func, peak_envelope_func):
        lin = linear_rebound_func(next_vals)
        peak = peak_envelope_func(next_vals)
        avg_rem = self.compute_average_remaining(next_vals)

        text = (
            f"LR1 ⚽={lin:.2f} | PE1 ⚽={peak:.2f}\n"
            f"CORRECT SCORE ⚽= {avg_rem:.2f}\n"
        )

        if hasattr(target, "insert"):
            target.insert(tk.END, text)
        else:
            target.write(text)

    def write_log(self,target,text):
        if hasattr(target,"insert"):target.insert(tk.END,text)
        else:target.write(text)

    def log_motif_table(self, target, motif_configs, results):
        col_width = 12

        def write(text):
            if hasattr(target, "insert"):
                target.insert(tk.END, text)
            else:
                target.write(text)

        header = " " * 3
        for i, (title, _, _) in enumerate(motif_configs, 1):
            header += f"|{str(i)+'°)':^{col_width}}"
        write(header + "|\n")

        separator = "-" * (3 + (col_width + 1) * len(motif_configs) + 1)
        write(separator + "\n")

        labels = ["C", "D", "P"]
        keys = ["croissant", "decroissant", "proportionnel"]

        for label, key in zip(labels, keys):
            row = f"{label:<3}"
            for res in results[key]:
                if res["motif_cible"] is None or res["prediction"] is None:
                    value = "X"
                else:
                    value = f"{res['prediction']:.1f}({len(res['similar_blocks'])})"
                row += f"|{value:^{col_width}}"
            write(row + "|\n")

    def run_benchmark(self):
        if not self.csv_loaded:
            return
        filepath = "Benchmark_F4.txt"
        file_exists = os.path.isfile(filepath)

        seriesA = read_numeric_after_marker(self.teamA_scores, "A")
        seriesC = read_numeric_after_marker(self.teamC_scores, "C")
        if not seriesA or not seriesC:
            return

        resultA, resultC = self.fully_static_method_with_patterns(seriesA, seriesC)

        with open(filepath, "w", encoding="utf-8") as f:  
            f.write("=========== BENCHMARK ===========\n\n")

            def log(text):
                f.write(text + "\n")

            teamA_score = getattr(self, "entry_teamA_score").get().strip()
            teamC_score = getattr(self, "entry_teamC_score").get().strip()

            def process_series(series, result, label, team_score):
                log(f"\nEQUIPE {label}")
                log(f"(L'équipe {label}) à marquée {team_score} but(s)")
                log(f"===> 📊 METHODE 1 <===")
                used_global = set()
                res1 = [self.fully_static_method_pattern1(series, ml, blocN, self.motif_next_distance, used_global)
                        for _, ml, blocN in self.motif_configs]
                log(self.build_prediction_table(res1))

                log(f"\n===> GOAL PROB 1 <===")
                motif_configs = [
                    ("1°) ML=2 / Bloc=31", 2, 31),
                    ("2°) ML=3 / Bloc=61", 3, 61),
                    ("3°) ML=4 / Bloc=91", 4, 91),
                    ("4°) ML=6 / Bloc=91", 6, 91),
                ]
                results = {"croissant": [], "decroissant": [], "proportionnel": []}
                for title, ml, bloc in motif_configs:
                    results["croissant"].append(self.detect_target_motif_prediction_disorder(series[:-1], ml, bloc, "croissant"))
                    results["decroissant"].append(self.detect_target_motif_prediction_disorder(series[:-1], ml, bloc, "decroissant"))
                    results["proportionnel"].append(self.detect_target_motif_prediction_disorder(series[:-1], ml, bloc, "proportionnel"))
                self.log_motif_table(f, motif_configs, results)

                log(f"\n===> GOAL PROB 2 <===")
                used_global = set()
                res2 = {t: self.fully_static_method_pattern1(series, ml, blocN, self.motif_next_distance, used_global)
                        for t, ml, blocN in self.motif_configs2}
                log(self.build_result_table(res2))

                log(f"\n===> 📊 METHODE 3 <===")
                lin_Id, lin_Pr, peak_Id, peak_Pr = [], [], [], []
                for _, ml, blocN in self.motif_configs:
                    r = self.fully_static_method_pattern2(series, ml, blocN, self.motif_next_distance)
                    lin_Id.append(f"{r['pred_results1']['linear_rebound']:.1f}")
                    lin_Pr.append(f"{r['pred_results2']['linear_rebound']:.1f}")
                    peak_Id.append(f"{r['pred_results1']['peak_envelope']:.1f}")
                    peak_Pr.append(f"{r['pred_results2']['peak_envelope']:.1f}")

                def row(l, v): return f"{l:<2}|" + "|".join(f"{x:^2}" for x in v) + "|"

                log("Rebond linéaires:")
                log(row("Id", lin_Id))
                log(row("Pr", lin_Pr))
                log("\nPeak envelope:")
                log(row("Id", peak_Id))
                log(row("Pr", peak_Pr))

                log(f"\n=========== CORRECT SCORE SET 2 ===========")
                results_complete = self.motif_engine_complete2(series)
                log("\n📊 ACCURACY 2:")
                for k, v in results_complete.items():
                    log(f"⚽={v['value']:<3} | {k:<2} | equal={v['equal']:<3} | prop={v['prop']:<3}")

                divergence = self.decide_divergence2(results_complete)
                log(f"⚽ could be: {divergence}.")

                log("\n📊 ACCURACY:")
                results_targeted = self.motif_engine_targeted_with_cr2(series, tol=0.15)
                if results_targeted:
                    for k, v in results_targeted.items():
                        log(f"⚽={v['value']:<3} | {k:<2} | equal={v['equal']:<3} | prop={v['prop']:<3}")

                divergenceT = self.decide_divergence_targeted(results_targeted)
                log(f"⚽ could be: {divergenceT}.")

                lin = self.linear_rebound_prediction(series)
                peak = self.peak_envelope_linear_prediction(series)
                avg_rem = self.compute_average_remaining(series)

                log("\n📊 🎯OVER LINEAR REBOUND🎯")
                log(f"LR ⚽={lin['prediction']:.3f} | PE ⚽={peak['prediction']:.3f}")
                log(f"CORRECT SCORE ⚽= {avg_rem:.3f}")

                motif, data = result
                e, p, pi, m, next_v, reb, peak_v = data
                self.display_motifs(series, motif, e, p, pi, m, next_v, reb, peak_v, f, "motif", self.linear_rebound, self.peak_envelope)

                log("\n📊 🎯UNDER LINEAR REBOUND🎯")
                for b in self.compute_blocks_with_gaps(series, 2, None, f):
                    log(f"LR ⚽={b.get('linear_rebound',0):.3f} | PE ⚽={b.get('peak_envelope',0):.3f} | CORRECT SCORE ⚽={b.get('average_remaining',0):.3f}")

                self.compute_over_under_full(series, f)
                self.display_recent_averages(series, f)
                self.display_median_extrema_means(series, f, "min")
                self.display_median_extrema_means(series, f, "max")

                log("\n📊 GLOBAL LOT / FEW GOALS PROB:")
                zero_seq, zero_signal, last_zero = self.analyze_zero_pattern(series)
                if zero_seq:
                    log("X" if len(zero_seq)<3 else f"⚽= {zero_signal} {last_zero}")

                self.detect_over_probable(series, f)
                self.check_over_probability_hybrid(series, f)
                log("\n📊 Consecutives 0 & 1:")
                self.display_consecutive_stats(series, f)

            process_series(seriesA, resultA, "A", teamA_score)
            process_series(seriesC, resultC, "C", teamC_score)

            messagebox.showinfo("Benchmark", "Benchmark terminé ✔")

    def extract_last_means(self, series):
        data = series[:-1]
        last5 = sum(data[-5:]) / 5 if len(data) >= 5 else 0
        last10 = sum(data[-10:]) / 10 if len(data) >= 10 else 0
        last15 = sum(data[-15:]) / 15 if len(data) >= 15 else 0
        return last5, last10, last15


    def extract_median_extrema(self, series, mode="min"):
        data = series[:-1]
        configs = [(7,15),(7,27),(7,51),(13,27),(13,51),(13,99),(25,51),(25,99)]

        values = []
        for mean_size, zone_size in configs:
            if len(data) < zone_size:
                continue
            zone = data[-zone_size:]
            val = min(zone) if mode == "min" else max(zone)
            idx = zone.index(val)

            half = mean_size // 2
            start = max(0, idx - half)
            end = start + mean_size
            if end > len(zone):
                end = len(zone)
                start = end - mean_size

            subset = zone[start:end]
            if len(subset) == mean_size:
                values.append(sum(subset) / mean_size)

        return values if values else [0]


    def extract_over_under(self, series):
        data = series[:-1]
        if len(data) < 5:
            return "neutre"

        last = data[-1]
        avg = sum(data[-10:]) / min(10, len(data))

        if last > avg:
            return "+ de"
        elif last < avg:
            return "- de"
        else:
            return "à 0"

    def predict_score(self, data):
        last5 = data["last5"]
        last10 = data["last10"]
        last15 = data["last15"]

        median_min = min(data["median_min_values"])
        median_max = max(data["median_max_values"])

        lr_cs = data["lr_correct_score"]
        pe_cs = data["pe_correct_score"]

        over_under = data["over_under"]

        accuracy = data["accuracy"]
        accuracy2 = data["accuracy2"]

        zone_min = min(last5, last10, last15, median_min)
        zone_max = max(last5, last10, last15, median_max)

        mean_last = (last5 + last10 + last15) / 3

        score_continu = (lr_cs + pe_cs + mean_last) / 3

        if "à 0" in over_under:
            score_continu -= 0.3
        elif "+ de" in over_under:
            score_continu += 0.3

        score_final = round(score_continu)

        if score_final < zone_min:
            score_final = int(zone_min)

        if score_final > zone_max:
            score_final = int(zone_max)

        best_acc = max(accuracy, key=lambda x: (x["equal"], x["prop"]))

        if abs(best_acc["score"] - score_final) == 1:
            score_final = best_acc["score"]

        return score_final

    def run_prediction(self,motif_length=2):
        seriesA=read_numeric_after_marker(self.teamA_scores,"A")
        seriesC=read_numeric_after_marker(self.teamC_scores,"C")

        if not seriesA or not seriesC:
            self.log_csv.insert(tk.END,"CSV vide – arrêt.")
            self.log_csv.see(tk.END)
            return

        resultA,resultC=self.fully_static_method_with_patterns(seriesA,seriesC)

        for log in (self.log_teamA,self.log_teamA_1,self.log_teamC,self.log_teamC_1):
            log.delete("1.0",tk.END)

        def write_method1(series,log):
            self.write_log(log,"===> 📊 METHODE 1 <===\n")
            used_global = set()
            res = [self.fully_static_method_pattern1(series, ml, blocN, self.motif_next_distance, used_global)
                   for _, ml, blocN in self.motif_configs]

            self.write_log(log,self.build_prediction_table(res)+"\n")

        write_method1(seriesA,self.log_teamA)
        write_method1(seriesC,self.log_teamC)

        def write_method2(series,log):
            self.write_log(log, "\n===> GOAL PROB 1 <===\n")
            motif_configs = [            
                ("1°) ML=2 / Bloc=31", 2, 31),
                ("2°) ML=3 / Bloc=61", 3, 61),
                ("3°) ML=4 / Bloc=91", 4, 91),
                ("4°) ML=6 / Bloc=91", 6, 91),
            ]

            results = {
                "croissant": [],
                "decroissant": [],
                "proportionnel": []
            }

            for title, ml, bloc in motif_configs:
                res_croissant = self.detect_target_motif_prediction_disorder(
                    series[:-1], motif_length=ml, block_size=bloc, motif_type="croissant"
                )
                results["croissant"].append(res_croissant)

                res_decroissant = self.detect_target_motif_prediction_disorder(
                    series[:-1], motif_length=ml, block_size=bloc, motif_type="decroissant"
                )
                results["decroissant"].append(res_decroissant)

                res_prop = self.detect_target_motif_prediction_disorder(
                    series[:-1], motif_length=ml, block_size=bloc, motif_type="proportionnel"
                )
                results["proportionnel"].append(res_prop)

            self.log_motif_table(log, motif_configs, results)

        write_method2(seriesA,self.log_teamA)
        write_method2(seriesC,self.log_teamC)
        
        def write_method3(series, log):
            self.write_log(log, "\n===> GOAL PROB 2 <===\n")

            used_global = set() 
            res = {}
            for t, ml, blocN in self.motif_configs2:
                res[t] = self.fully_static_method_pattern1(series, ml, blocN, self.motif_next_distance, used_global)

            self.write_log(log,self.build_result_table(res)+"\n")

        write_method3(seriesA,self.log_teamA)
        write_method3(seriesC,self.log_teamC)
        
        def write_method4(series,log):
            self.write_log(log,"\n===> 📊 METHODE 3 <===\n")
            lin_Id,lin_Pr,peak_Id,peak_Pr=[],[],[],[]
            for _,ml,blocN in self.motif_configs:
                r=self.fully_static_method_pattern2(series,ml,blocN,self.motif_next_distance)
                lin_Id.append(f"{r['pred_results1']['linear_rebound']:.1f}")
                lin_Pr.append(f"{r['pred_results2']['linear_rebound']:.1f}")
                peak_Id.append(f"{r['pred_results1']['peak_envelope']:.1f}")
                peak_Pr.append(f"{r['pred_results2']['peak_envelope']:.1f}")

            def row(l,v):return f"{l:<2}|"+ "|".join(f"{x:^2}" for x in v)+"|"

            self.write_log(log,"Rebond linéaires:\n")
            self.write_log(log,""+row("Id",lin_Id))
            self.write_log(log,"\n"+row("Pr",lin_Pr))
            self.write_log(log,"\n")
            self.write_log(log,"\nPeak envelope:\n")
            self.write_log(log,""+row("Id",peak_Id))
            self.write_log(log,"\n"+ row("Pr", peak_Pr) + "\n\n")
            
        write_method4(seriesA,self.log_teamA)
        write_method4(seriesC,self.log_teamC)

        def write_lower(series,result,log):
            self.write_log(log,"=========== CORRECT SCORE SET 2 ===========\n")

            # ====== ICI ======
            try:
                # Préparer les données nécessaires pour predict_score
                lin = self.linear_rebound_prediction(series)
                peak = self.peak_envelope_linear_prediction(series)
                avg_rem = self.compute_average_remaining(series)

                results_complete = self.motif_engine_complete2(series)
                results_targeted = self.motif_engine_targeted_with_cr2(series, tol=0.15)

                last5, last10, last15 = self.extract_last_means(series)
                median_min = self.extract_median_extrema(series, "min")
                median_max = self.extract_median_extrema(series, "max")
                over_under = self.extract_over_under(series)

                acc = []
                if results_targeted:
                    for v in results_targeted.values():
                        acc.append({
                            "score": v["value"],
                            "equal": v["equal"],
                            "prop": v["prop"]
                        })

                acc2 = []
                for v in results_complete.values():
                    acc2.append({
                        "score": v["value"],
                        "equal": v["equal"],
                        "prop": v["prop"]
                    })

                data = {
                    "last5": last5,
                    "last10": last10,
                    "last15": last15,
                    "median_min_values": median_min,
                    "median_max_values": median_max,
                    "lr_correct_score": lin["prediction"],
                    "pe_correct_score": peak["prediction"],
                    "over_under": over_under,
                    "accuracy": acc,
                    "accuracy2": acc2
                }

                final_score = self.predict_score(data)
                self.write_log(log, f"⚽ PREDICTED SCORE = {final_score}\n")
            except Exception as e:
                self.write_log(log, f"[MODEL ERROR] {e}\n")
            # ==================


                
            results_complete = self.motif_engine_complete2(series)
            self.write_log(log, "\n📊 ACCURACY 2:\n")
            for k, v in results_complete.items():
                self.write_log(
                    log,
                    f"⚽={v['value']:<3} | {k:<2} | equal={v['equal']:<3} | prop={v['prop']:<3}\n"
                )
            
            divergence = self.decide_divergence2(results_complete)
            self.write_log(log,f"⚽ could be: {divergence}.\n")
            
            self.write_log(log, "\n📊 ACCURACY:\n")
            results_targeted = self.motif_engine_targeted_with_cr2(series, tol=0.15)
            if results_targeted:
                for k, v in results_targeted.items():
                    self.write_log(
                        log,
                        f"⚽={v['value']:<3} | {k:<2} | equal={v['equal']:<3} | prop={v['prop']:<3}\n"
                    )

            divergenceT = self.decide_divergence_targeted(results_targeted)

            self.write_log(log,f"⚽ could be: {divergenceT}.\n")

            lin = self.linear_rebound_prediction(series)
            peak = self.peak_envelope_linear_prediction(series)
            avg_rem = self.compute_average_remaining(series)

            self.write_log(log, f"\n📊 🎯OVER LINEAR REBOUND🎯\n")
            self.write_log(log,
                f"LR ⚽={lin['prediction']:.3f} | PE ⚽={peak['prediction']:.3f}\n"
                f"CORRECT SCORE ⚽= {avg_rem:.3f}\n"
            )

            motif,data=result
            e,p,pi,m,next_v,reb,peak_v=data
            self.display_motifs(series,motif,e,p,pi,m,next_v,reb,peak_v,log,"motif",self.linear_rebound,self.peak_envelope)
            
            self.write_log(log,"\n📊 🎯UNDER LINEAR REBOUND🎯\n")
            for b in self.compute_blocks_with_gaps(series,2,None,log):
                self.write_log(log,
                    f"LR ⚽={b.get('linear_rebound',0):.3f} | PE ⚽={b.get('peak_envelope',0):.3f}\n"
                    f"CORRECT SCORE ⚽= {b.get('average_remaining',0):.3f}\n"
                )

            self.compute_over_under_full(series, log)

            self.display_recent_averages(series,log)
                
            self.display_median_extrema_means(series,log,"min")
            self.display_median_extrema_means(series,log,"max")

            self.write_log(log,"\n📊 GLOBAL LOT / FEW GOALS PROB:\n")

            zero_seq,zero_signal,last_zero=self.analyze_zero_pattern(series)
            if zero_seq:
                if len(zero_seq)<3:self.write_log(log,"X\n")
                else:self.write_log(log,f"⚽= {zero_signal} {last_zero}\n")

            self.detect_over_probable(series, log)
            
            self.check_over_probability_hybrid(series, log)

            self.write_log(log,"\n📊 Consecutives 0 & 1:\n")
            self.display_consecutive_stats(series,log)



        write_lower(seriesA,resultA,self.log_teamA_1)
        write_lower(seriesC,resultC,self.log_teamC_1)
            
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
