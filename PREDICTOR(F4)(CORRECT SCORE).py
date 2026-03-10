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

DEFAULT_SCORES = {}
EXPORT_DIR = "export_csv" # proposer un chemin configurable

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
        self.title("PREDICTOR(F4)(CP1_2)")
        self.geometry("1200x1000")
        self.teamA_scores: List[str] = []
        self.teamC_scores: List[str] = []
        self.motif_next_distance = 0
        
        self.motif_configs = [            
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
            ("66.TREND", 11, 12), #
            ("77.LIMIT", 6, 12), #

            ("SHORT", 3, 15), #
            ("WIDE", 3, 30), #
            #("3 >", 2, 45), #
            #("4 >", 5, 15), #
            ("WIDE 2", 5, 30), #
            #("6 >", 5, 45), #
            #("7 >", 7, 15), #
            ("WIDE 3", 7, 30), #
            #("9 >", 7, 45), #
            ("WIDE L", 3, 60), #
            #("11 >", 5, 60), #
            ("WIDE XL", 7, 60), #
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
        index = self.log_csv.nearest(event.y) # <== X en cas de bug
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

    def detect_cross_inversion(self, results):
        found_pairs = set()
        differences = []

        for i in range(len(results)):
            b1, d1 = results[i]

            for j in range(i + 1, len(results)):
                b2, d2 = results[j]

                if b1 == d2 and d1 == b2:
                    key = tuple(sorted((b1, d1)))

                    if key not in found_pairs:
                        found_pairs.add(key)
                        diff = abs(b1 - d1)
                        differences.append(diff)

        return differences

    def get_prediction_from_difference(self, scores: List[int]) -> List[tuple[float, float]]:
        nCount = len(scores)
        if nCount < 12:
            return []

        nbSeries = nCount // 12
        total1 = 0.0
        results = []

        for i in range(1, nbSeries + 1):
            endRow = nCount - (i - 1) * 12
            start = endRow - 11 

            serie = scores[start : start + 12]
            serie = [int(v) if v is not None else 0 for v in serie]

            sum11Current = sum(serie[:11])
            resultB = total1 - sum11Current

            total2 = sum(serie)
            resultD = total2 - total1

            results.append((resultB, resultD))
            total1 = total2

        return results

    def compute_prediction(self, scores: List[int]) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []

        n = len(scores)
        if n < 12:
            return results

        total1 = 0.0

        for i in range(11, n):
            start = i - 11
            serie = scores[start:i+1]

            total2 = sum(serie)
            sum11 = sum(serie[:11])

            resultB = total1 - sum11
            resultD = total2 - total1

            results.append({
                "B": float(resultB),
                "D": float(resultD)
            })

            total1 = total2

        return results

    def fully_static_method_pattern(self, series, motif_length, block_size, motif_next_distance):
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

    def add_motif_next_button(self, controls):
        def modify_distance():
            new_val = simpledialog.askinteger(
                "MOTIF<>NEXT",
                f"Distance admise actuelle: {self.motif_next_distance}",
                minvalue=0
            )
            if new_val is not None:
                self.motif_next_distance = new_val

        ttk.Button(controls, text="MOTIF<>NEXT", command=modify_distance)\
            .grid(row=0, column=5, padx=4)

    def fully_static_method_with_patterns(self, seriesA, seriesC):
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

            for i in range(len(data) - L):
                block = data[i:i+L]
                if block == motif:
                    e_list.append(i)
                elif is_proportional(motif, block):

                    ratio = block[0] / motif[0]
                    if ratio > 1:
                        p_super_list.append(i)
                    elif 0 < ratio < 1:
                        p_infer_list.append(i)

            m_list = list(set(e_list + p_super_list + p_infer_list))
            return e_list, p_super_list, p_infer_list, m_list

        motifA = detect_target_motif(seriesA)
        if motifA:
            e_list, prop_superieur, prop_inferieur, m_list = classify(seriesA, motifA)
            resultA = (motifA, (e_list, prop_superieur, prop_inferieur, m_list))
        else:
            resultA = (None, ([], [], [], []))

        motifC = detect_target_motif(seriesC)
        if motifC:
            e_list, prop_superieur, prop_inferieur, m_list = classify(seriesC, motifC)
            resultC = (motifC, (e_list, prop_superieur, prop_inferieur, m_list))
        else:
            resultC = (None, ([], [], [], []))

        return resultA, resultC

    def compute_blocks_with_gaps(self, series, min_block=2, gap_between=None):
        n = len(series)
        i = n - 1
        results = []

        most_recent_mean_end = None

        while i >= min_block:
            target = series[i]
            found_block = False

            for j in range(i - min_block, -1, -1):
                block = series[j:i+1]

                if len(block) >= (min_block + 1) and sum(block)/len(block) == target:

                    if most_recent_mean_end is None:
                        most_recent_mean_end = i

                    results.append({
                        "type": "moyenne",
                        "values": block,
                        "average": target
                    })

                    next_index = i + 1

                    if next_index < n:
                        if gap_between is None:
                            unused = series[next_index:]
                        else:
                            unused = series[next_index: next_index + gap_between]

                        if unused:
                            results.append({
                                "type": "non_utilise",
                                "values": unused
                            })

                    i = j - 1
                    found_block = True
                    break

            if not found_block:
                i -= 1

        if most_recent_mean_end is not None:
            remaining = series[most_recent_mean_end + 1 : -1]

            if remaining:
                results.append({
                    "type": "reste_final",
                    "values": remaining
                })

                if len(remaining) > 1:
                    avg_remaining = sum(remaining[1:]) / len(remaining[1:])
                else:
                    avg_remaining = None  

                results.append({
                    "type": "moyenne_reste_final",
                    "values": remaining,
                    "average": avg_remaining
                })

        return results

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

        self.entry_motif_next = ttk.Entry(controls, width=5)
        self.entry_motif_next.grid(row=0, column=6, padx=2)
        self.entry_motif_next.insert(0, str(self.motif_next_distance))

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=8, pady=8)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.columnconfigure(2, weight=1)

        self._build_team_frame(mid, 0, "A")
        self._build_team_frame(mid, 1, "C")
        self._build_log_frame(mid, controls)

    def on_motif_next_clicked(self):
        try:
            val = int(self.entry_motif_next.get())
            self.motif_next_distance = max(0, val)
            self.entry_motif_next.delete(0, tk.END)
            self.entry_motif_next.insert(0, str(self.motif_next_distance))
        except Exception:
            messagebox.showwarning("Valeur invalide", "Veuillez entrer un entier >= 0.")

    def _build_team_frame(self, parent, col, team):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, sticky="nsew", padx=4)

        line = ttk.Frame(frame)
        line.pack(anchor="w", fill="x")

        entry = ttk.Entry(line, width=53)
        entry.pack(side="left")
        setattr(self, f"entry_team{team}", entry)

        entry_score = ttk.Entry(line, width=5)
        entry_score.pack(side="left", padx=(6, 0))
        setattr(self, f"entry_team{team}_score", entry_score)

        entry_score.bind(
            "<FocusOut>",
            lambda e, t=team: self.save_team_score_to_csv(t)
        )

        log1 = tk.Text(frame, height=25, width=60)
        log1.pack()
        setattr(self, f"log_team{team}", log1)

        log2 = tk.Text(frame, height=25, width=60)
        log2.pack()
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

    def get_indices_to_show(self, lst):
        n = len(lst)
        if n < 5:
            
            return list(range(n))
        elif n % 2 == 0 and n > 5:
    
            mid1 = n // 2 - 1
            mid2 = n // 2
            return [0, 1, mid1, mid2, n-2, n-1]
        elif n % 2 == 1 and n > 6:

            mid = n // 2
            return [0, 1, mid-1, mid, mid+1, n-2, n-1]
        else:

            return list(range(n))

    def display_motifs(self, series, motif, e_list, prop_superieur, prop_inferieur, m_list, log_widget, label_prefix):
        if motif is None:
            return

        def next_val(idx):
            if idx + len(motif) < len(series):
                return series[idx + len(motif)]
            return "N/A"

        if e_list:
            log_widget.insert(tk.END, "IDENTIQUE\n")
            for j, idx in enumerate(e_list[:4]):
                log_widget.insert(tk.END, f"motif{j+1} = {next_val(idx)}\n")

            if len(e_list) >= 2:
                base = next_val(e_list[0])
                comp = lambda a, b: a > b
                sign_main = "+"
                word = "OVER"

                if comp(next_val(e_list[1]), base):
                    log_widget.insert(tk.END, f"{word}\n")

                if len(e_list) >= 3:
                    v3 = next_val(e_list[2])
                    if comp(v3, base):
                        log_widget.insert(tk.END, f"{word} {sign_main}\n")
                    if comp(v3, base) and comp(v3, next_val(e_list[1])):
                        log_widget.insert(tk.END, f"{word} {sign_main}{sign_main}\n")

                if len(e_list) >= 4:
                    remaining = e_list[3:]
                    half = len(remaining) // 2
                    count1 = count2 = 0
                    for k, idx in enumerate(remaining):
                        if comp(next_val(idx), base):
                            if k < half:
                                count1 += 1
                            else:
                                count2 += 1
                    if count1 > 0:
                        log_widget.insert(tk.END, f"{word} COUNT= {sign_main*2}{count1}\n")
                    if count2 > 0:
                        log_widget.insert(tk.END, f"{word} COUNT2= {sign_main}{count2}\n")

        if prop_superieur:
            log_widget.insert(tk.END, "\nPROPORTIONNEL SUP\n")
            for j, idx in enumerate(prop_superieur[:4]):
                log_widget.insert(tk.END, f"motif{j+1} = {next_val(idx)}\n")

            if len(prop_superieur) >= 4:
                base = next_val(prop_superieur[0])
                count1 = count2 = 0
                remaining = prop_superieur[3:]
                half = len(remaining)//2
                for k, idx in enumerate(remaining):
                    if next_val(idx) > base:
                        if k < half:
                            count1 += 1
                        else:
                            count2 += 1
                if count1 > 0:
                    log_widget.insert(tk.END, f"OVER COUNT= ++{count1}\n")
                if count2 > 0:
                    log_widget.insert(tk.END, f"OVER COUNT2= +{count2}\n")

        if prop_inferieur:
            log_widget.insert(tk.END, "\nPROPORTIONNEL INF\n")
            for j, idx in enumerate(prop_inferieur[:4]):
                log_widget.insert(tk.END, f"motif{j+1} = {next_val(idx)}\n")

            if len(prop_inferieur) >= 4:
                base = next_val(prop_inferieur[0])
                count1 = count2 = 0
                remaining = prop_inferieur[3:]
                half = len(remaining)//2
                for k, idx in enumerate(remaining):
                    if next_val(idx) < base:
                        if k < half:
                            count1 += 1
                        else:
                            count2 += 1
                if count1 > 0:
                    log_widget.insert(tk.END, f"UNDER COUNT= --{count1}\n")
                if count2 > 0:
                    log_widget.insert(tk.END, f"UNDER COUNT2= -{count2}\n")                
# (BENCHMARK)
    def run_prediction(self, motif_length=2):
        
        full_series_A = read_numeric_after_marker(self.teamA_scores, "A")
        full_series_C = read_numeric_after_marker(self.teamC_scores, "C")

        seriesA = full_series_A
        seriesC = full_series_C

        if not seriesA or not seriesC:
            msg = "Séries A ou C vides – arrêt de la prédiction."
            self.log_csv.insert(tk.END, "[WARNING] " + msg)
            self.log_csv.see(tk.END)
            return

        for log in (self.log_teamA, self.log_teamA_1, self.log_teamC, self.log_teamC_1):
            log.delete("1.0", tk.END)

        start_A = next((i+1 for i,v in enumerate(self.teamA_scores) if str(v).strip().upper()=="A"), 0)
        start_C = next((i+1 for i,v in enumerate(self.teamC_scores) if str(v).strip().upper()=="C"), 0)
        
        resultsA = self.compute_prediction(seriesA)
        resultsC = self.compute_prediction(seriesC)
       
        (resultA, (eA, pA, piA, mA)), (resultC, (eC, pC, piC, mC)) = self.fully_static_method_with_patterns(seriesA, seriesC)

        # Example: take first config
        title, ml, blocN = self.motif_configs[0]

        resA = self.fully_static_method_pattern(
            seriesA,
            motif_length=ml,
            block_size=blocN,
            motif_next_distance=self.motif_next_distance
        )

        resC = self.fully_static_method_pattern(
            seriesC,
            motif_length=ml,
            block_size=blocN,
            motif_next_distance=self.motif_next_distance
        )

        # ====== log ======
        
        # === CP SET A ===
        self.log_teamA.insert(tk.END, "📊 CP(+prop sup & inf)\n")

        for r in resultsA[-6:]:
            #b = r[0]
            #d = r[2]
            b = r["B"]
            d = r["D"]
            total = b + d

            self.log_teamA.insert(
                tk.END,
                f"{b:8.2f} | {d:8.2f} | {total:8.2f}\n"
            )

        # ===== ZERO PATTERN A =====
        zero_seq_A, zero_signal_A, last_zero_A = self.analyze_zero_pattern(seriesA)
        if zero_seq_A:
            self.log_teamA.insert(tk.END, "\n📊 ZERO PATTERN\n")

            if len(zero_seq_A) < 3:
                self.log_teamA.insert(tk.END, "Comparaison impossible (<3 blocs)\n")
            else:
                if zero_signal_A == "Under":
                    self.log_teamA.insert(
                        tk.END,
                        f"⚽= Under {last_zero_A}\n"
                    )
                elif zero_signal_A == "Over":
                    self.log_teamA.insert(
                        tk.END,
                        f"⚽= Over {last_zero_A}\n"
                    )

        # ===== DIFFÉRENCES CROISÉES A =====
        diff_results_A = self.get_prediction_from_difference(seriesA)
        cross_A = self.detect_cross_inversion(diff_results_A)

        if cross_A:
            self.log_teamA.insert(tk.END, "\n📊 DIFF CROISÉES\n")
            for d in cross_A:
                self.log_teamA.insert(tk.END, f"⚽={d}\n")

        # ===== DIFFÉRENCES COMPLET =====
        if diff_results_A:
            self.log_teamA.insert(tk.END, "\n📊 DIFF DIRECTES: D= Line 10/Count= 3\n")
            resultB_list, resultD_list = map(list, zip(*diff_results_A))
            zipped_list = list(zip(resultB_list, resultD_list))
            display_list = zipped_list[1:]
            display_list = list(reversed(display_list))
            results_list = [
                {
                    "B": b,
                    "D": d,
                    "diff_abs": abs(abs(b) - abs(d)),
                    "diff": abs(b - d)
                }
                for b, d in display_list
            ]

            if not results_list:
                self.log_teamA.insert(tk.END, "⚠️ SERIE TROP COURTE\n")
            else:
                min_B = min(res["B"] for res in results_list)
                max_B = max(res["B"] for res in results_list)

                widths = {
                    k: max(len(f"{res[k]:.2f}") for res in results_list)
                    for k in ["B", "D", "diff_abs", "diff"]
                }

                for res in results_list:
                    symbol = "(-)" if res["B"] == min_B else "(+)" if res["B"] == max_B else ""
                    
                diff_counts = Counter(res['diff'] for res in results_list)

            for i, res in enumerate(results_list, start=1):
                symbol = "(-)" if res["B"] == min_B else "(+)" if res["B"] == max_B else ""
                count = diff_counts[res['diff']]

                line = (
                    f"B={res['B']:{widths['B']}.2f} | "
                    f"D={res['D']:{widths['D']}.2f} | "
                    f"⚽={res['diff_abs']:{widths['diff_abs']}.2f} | "
                    f"DIFF={res['diff']:{widths['diff']}.2f} | "
                    f"COUNT={count} | "
                    f"{symbol}\n"
                )

                if 8 <= i <= 12:
                    line = line.replace(" | ", " > ")

                self.log_teamA.insert(tk.END, line)
            
        # --- fully_static_method_pattern ---
        for title, ml, blocN in self.motif_configs:
            self.log_teamA.insert(tk.END, f"\n===== {title} =====\n")

            # Appel à la méthode adaptée
            results = self.fully_static_method_pattern(
                seriesA,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            # Labels pour les différents types de résultats
            label_map = {
                "pred_results1": "1._____IDENTIQUE",
                "pred_results2": "2.PROPORTIONNELS",
                "pred_results3": "3.___________1+2",
                "pred_results4": "4.______SS MN/MX",
                "pred_results5": "5.MULTI SS MN/MX",
                "pred_results6": "6._________1+2+4",
                "pred_results7": "7._________1+2+5"
            }

            # Affichage détaillé de chaque type
            for key, label in label_map.items():
                count = results[key]['count']
                sum_val = results[key]['sum']

                if count == 0:
                    ball_display = "⛔"
                else:
                    ball_display = sum_val

                self.log_teamA.insert(
                    tk.END,
                    f"  {label}, ⚽={ball_display} | X={count}\n"
                )

        # === CP SET C ===
        self.log_teamC.insert(tk.END, "📊 CP(+prop sup & inf)\n")

        for r in resultsC[-6:]:
            #b = r[0]
            #d = r[2]
            b = r["B"]
            d = r["D"]
            total = b + d
            
            self.log_teamC.insert(
                tk.END,
                f"{b:8.2f} | {d:8.2f} | {total:8.2f}\n"
            )
        
        # ===== ZERO PATTERN C =====
        zero_seq_C, zero_signal_C, last_zero_C = self.analyze_zero_pattern(seriesC)

        if zero_seq_C:
            self.log_teamC.insert(tk.END, "\n📊 ZERO PATTERN\n")

            if len(zero_seq_C) < 3:
                self.log_teamC.insert(tk.END, "Comparaison impossible (<3 blocs)\n")
            else:
                if zero_signal_C == "Under":
                    self.log_teamC.insert(
                        tk.END,
                        f"⚽= Under {last_zero_C}\n"
                    )
                elif zero_signal_C == "Over":
                    self.log_teamC.insert(
                        tk.END,
                        f"⚽= Over {last_zero_C}\n"
                    )

        # ===== DIFFÉRENCES CROISÉES C =====
        diff_results_C = self.get_prediction_from_difference(seriesC)
        cross_C = self.detect_cross_inversion(diff_results_C)

        if cross_C:
            self.log_teamC.insert(tk.END, "\n📊 DIFF CROISÉES\n")
            for d in cross_C:
                self.log_teamC.insert(tk.END, f"⚽={d}\n")

        # ===== DIFFÉRENCES COMPLET =====
        if diff_results_C:
            self.log_teamC.insert(tk.END, "\n📊 DIFF DIRECTES: D= Line 10/Count= 3\n")
            resultB_list, resultD_list = map(list, zip(*diff_results_C))
            zipped_list = list(zip(resultB_list, resultD_list))
            display_list = zipped_list[1:]
            display_list = list(reversed(display_list))
            results_list = [
                {
                    "B": b,
                    "D": d,
                    "diff_abs": abs(abs(b) - abs(d)),
                    "diff": abs(b - d)
                }
                for b, d in display_list
            ]

            if not results_list:
                self.log_teamC.insert(tk.END, "⚠️ SERIE TROP COURTE\n")
            else:
                min_B = min(res["B"] for res in results_list)
                max_B = max(res["B"] for res in results_list)

                widths = {
                    k: max(len(f"{res[k]:.2f}") for res in results_list)
                    for k in ["B", "D", "diff_abs", "diff"]
                }

                for res in results_list:
                    symbol = "(-)" if res["B"] == min_B else "(+)" if res["B"] == max_B else ""
                                    
                diff_counts = Counter(res['diff'] for res in results_list)

            for i, res in enumerate(results_list, start=1):
                symbol = "(-)" if res["B"] == min_B else "(+)" if res["B"] == max_B else ""
                count = diff_counts[res['diff']]

                line = (
                    f"B={res['B']:{widths['B']}.2f} | "
                    f"D={res['D']:{widths['D']}.2f} | "
                    f"⚽={res['diff_abs']:{widths['diff_abs']}.2f} | "
                    f"DIFF={res['diff']:{widths['diff']}.2f} | "
                    f"COUNT={count} | "
                    f"{symbol}\n"
                )

                if 8 <= i <= 12:
                    line = line.replace(" | ", " > ")

                self.log_teamC.insert(tk.END, line)
            
        # --- fully_static_method_pattern ---
        for title, ml, blocN in self.motif_configs:
            self.log_teamC.insert(tk.END, f"\n===== {title} =====\n")

            # Appel à la méthode adaptée
            results = self.fully_static_method_pattern(
                seriesC,
                motif_length=ml,
                block_size=blocN,
                motif_next_distance=self.motif_next_distance
            )

            # Labels pour les différents types de résultats
            label_map = {
                "pred_results1": "1._____IDENTIQUE",
                "pred_results2": "2.PROPORTIONNELS",
                "pred_results3": "3.___________1+2",
                "pred_results4": "4.______SS MN/MX",
                "pred_results5": "5.MULTI SS MN/MX",
                "pred_results6": "6._________1+2+4",
                "pred_results7": "7._________1+2+5"
            }
            
            # Affichage détaillé de chaque type
            for key, label in label_map.items():
                count = results[key]['count']
                sum_val = results[key]['sum']

                if count == 0:
                    ball_display = "⛔"
                else:
                    ball_display = sum_val

                self.log_teamC.insert(
                    tk.END,
                    f"  {label}, ⚽={ball_display} | X={count}\n"
                )

        # ====== log_1 ======

        # ====== ANALYSE MOTIFS STATIQUES AVANCES ======
        self.display_motifs(seriesA, resultA, eA, pA, piA, mA, self.log_teamA_1, "motifA")

        # ===== BLOCS MOYENNES + GAP =====
        self.log_teamA_1.insert(tk.END, "\n📊 BLOCS MOYENNES\n")
        blocksA = self.compute_blocks_with_gaps(seriesA, min_block=2, gap_between=1)

        types = set(r["type"] for r in blocksA)
        indices_par_type = {}

        for t in types:
            blocs_type = [r for r in blocksA if r["type"] == t]
            indices_par_type[t] = self.get_indices_to_show(blocs_type)

        compteurs = {t: 0 for t in types}

        for r in blocksA:
            t = r["type"]
            idx = compteurs[t]
            if idx in indices_par_type[t]:
                if t == "reste_final":
                    self.log_teamA_1.insert(
                        tk.END,
                        f"\nReste en cours: {r['values']}\n"
                    )
                elif t == "moyenne_reste_final":
                    self.log_teamA_1.insert(
                        tk.END,
                        f"📈 Moyenne bloc en cours: {r['average']:.2f}\n"
                        if r['average'] is not None
                        else "📈 Moyenne bloc en cours: N/A\n"
                    )
                elif t == "moyenne":
                    self.log_teamA_1.insert(tk.END, f"Moyenne bloc = {r['average']}\n")
                elif t == "non_utilise":
                    self.log_teamA_1.insert(tk.END, f"Éléments non utilisés : {r['values']}\n")
            compteurs[t] += 1

        # ====== ANALYSE MOTIFS STATIQUES AVANCES ======
        self.display_motifs(seriesC, resultC, eC, pC, piC, mC, self.log_teamC_1, "motifC")

        # ===== BLOCS MOYENNES + GAP =====
        self.log_teamC_1.insert(tk.END, "\n📊 BLOCS MOYENNES\n")
        blocksC = self.compute_blocks_with_gaps(seriesC, min_block=2, gap_between=1)

        types = set(r["type"] for r in blocksC)
        indices_par_type = {}

        for t in types:
            blocs_type = [r for r in blocksC if r["type"] == t]
            indices_par_type[t] = self.get_indices_to_show(blocs_type)

        compteurs = {t: 0 for t in types}

        for r in blocksC:
            t = r["type"]
            idx = compteurs[t]
            if idx in indices_par_type[t]:
                if t == "reste_final":
                    self.log_teamC_1.insert(
                        tk.END,
                        f"\nReste en cours: {r['values']}\n"
                    )
                elif t == "moyenne_reste_final":
                    self.log_teamC_1.insert(
                        tk.END,
                        f"📈 Moyenne bloc en cours: {r['average']:.2f}\n"
                        if r['average'] is not None
                        else "📈 Moyenne bloc en cours: N/A\n"
                    )
                elif t == "moyenne":
                    self.log_teamC_1.insert(tk.END, f"Moyenne bloc = {r['average']}\n")
                elif t == "non_utilise":
                    self.log_teamC_1.insert(tk.END, f"Éléments non utilisés : {r['values']}\n")
            compteurs[t] += 1

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
