import os
import pandas as pd
import numpy as np
from tkinter import Tk, Frame, Button, Label, filedialog, Text, END, Listbox, Scrollbar, Entry, StringVar
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, peak_prominences, hilbert
import pywt
from sklearn.cross_decomposition import PLSRegression
from PyEMD import EMD

FEATURE_WEIGHTS = {
    "Lin_Arround7Last": 0.10441411353265306,
    "Peak_Arround7Last": 0.1522755424521934
}

PLAN_FEATURE_NAMES = [
    "Lin_Arround7Last","Peak_Arround7Last"
]

FEATURE_WEIGHTS2 = {
    "mean": 0.6,
    "freq_ge2": 0.3,
    "stability": 0.1
}

PLAN_FEATURE_NAMES2 = ["mean", "freq_ge2", "stability"]

WINDOW_SIZES = [1, 2]
WINDOW_SIZES2 = [1, 4]

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def read_numeric_after_marker(column: list, marker: str) -> list:
    start = next((i+1 for i, v in enumerate(column)
                  if str(v).strip().upper() == marker), None)
    if start is None:
        return []
    return [int(v) for v in column[start:] if is_number(v)]

def hilbert_envelope(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def emd_envelope(signal, mode='first'):
    emd = EMD()
    IMFs = emd(signal)
    if IMFs.shape[0] == 0:
        return signal
    if mode == 'first':
        return np.abs(hilbert(IMFs[0]))
    else:
        return np.abs(hilbert(IMFs).sum(axis=0))

def wavelet_envelope(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA = coeffs[0]  # approximation à la plus basse fréquence
    return np.abs(pywt.upcoef('a', cA, wavelet, level=level, take=len(signal)))

def peak_filtered_envelope(signal, distance=1):
    peaks, _ = find_peaks(signal, distance=distance)
    if len(peaks) == 0:
        return np.zeros_like(signal)
    env = np.interp(np.arange(len(signal)), peaks, signal[peaks])
    return env

def compute_features(series, window_size, n_occurrences=None):
    features_dict = arround_last_features(series, window_size, n_occurrences)

    # Liste ordonnée des features à utiliser
    feature_keys = [
        "Lin", "Peak", "HilbertMax", "HilbertMean",
        "EMDMax", "EMDMean", "WaveletMax", "WaveletMean",
        "PeakFilteredMax", "PeakFilteredMean"
    ]

    features_array = np.array([features_dict[k] for k in feature_keys])
    return features_array

# Exemple pour régression multiple
def fit_regressions(X, y):
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'PLS': PLSRegression(n_components=min(X.shape[1], 2))
    }
    fitted_models = {}
    for name, model in models.items():
        model.fit(X, y)
        fitted_models[name] = model
    return fitted_models

def get_non_overlapping_future_segments(series, window_size):
    series = np.array(series)

    if len(series) < window_size + 2:
        return np.array([])

    last_val = series[-(window_size + 1)]
    all_indices = np.where(series == last_val)[0]

    # NON CHEVAUCHEMENT
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

def arround_last_features_pattern(series, window_size):
    FEATURE_LIST = [
        "Lin","Peak",
        "HilbertMax","HilbertMean",
        "EMDMax","EMDMean",
        "WaveletMax","WaveletMean",
        "PeakFilteredMax","PeakFilteredMean"
    ]

    series = np.array(series)
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

    # motif exact = dernier segment
    ref_segment = segments[-1]

    mask = np.all(segments == ref_segment, axis=1)

    segments = segments[mask]
    targets = targets[mask]

    if len(segments) == 0:
        return {k: "X" for k in FEATURE_LIST}

    # LIN / PEAK
    y = targets.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)

    lin_coef = LinearRegression().fit(x, y).coef_[0][0]
    peak_env = y.max() - y.min()

    # HILBERT
    hilbert_env = np.abs(hilbert(segments, axis=1))
    hilbert_max = np.max(hilbert_env)
    hilbert_mean = hilbert_env.mean()

    # EMD approx
    detrended = segments - segments.mean(axis=1, keepdims=True)
    emd_env = np.abs(hilbert(detrended, axis=1))
    emd_max = np.max(emd_env)
    emd_mean = emd_env.mean()

    # WAVELET
    wav_envs = np.array([
        np.abs(pywt.upcoef('a', pywt.wavedec(seg,'db4',1)[0], 'db4', level=1, take=len(seg)))
        for seg in segments
    ])
    wav_max = np.max(wav_envs)
    wav_mean = wav_envs.mean()

    # PEAK FILTER
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

def arround_last_features(series, window_size, n_occurrences=None):
    segments, targets = get_non_overlapping_future_segments(series, window_size)

    if len(segments) == 0:
        return {
            "Lin": 0, "Peak": 0,
            "HilbertMax": 0, "HilbertMean": 0,
            "EMDMax": 0, "EMDMean": 0,
            "WaveletMax": 0, "WaveletMean": 0,
            "PeakFilteredMax": 0, "PeakFilteredMean": 0
        }

    # Limiter occurrences si demandé
    if n_occurrences is not None:
        segments = segments[-n_occurrences:]
        targets = targets[-n_occurrences:]

    # --- LIN / PEAK sur targets ---
    y = targets.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)

    lin_coef = LinearRegression().fit(x, y).coef_[0][0]
    peak_env = y.max() - y.min()

    # --- HILBERT ---
    hilbert_env = np.abs(hilbert(segments, axis=1))
    hilbert_max = np.max(hilbert_env)
    hilbert_mean = hilbert_env.mean()

    # --- EMD approx ---
    detrended = segments - segments.mean(axis=1, keepdims=True)
    emd_env = np.abs(hilbert(detrended, axis=1))
    emd_max = np.max(emd_env)
    emd_mean = emd_env.mean()

    # --- WAVELET ---
    wav_envs = np.array([
        np.abs(pywt.upcoef('a', pywt.wavedec(seg, 'db4', 1)[0], 'db4', level=1, take=len(seg)))
        for seg in segments
    ])
    wav_max = np.max(wav_envs)
    wav_mean = wav_envs.mean()

    # --- PEAK FILTERED ---
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

def arround_last_features_correctifs(s, window_size, n_occurrences=None):
    segments, targets = get_non_overlapping_future_segments(s, window_size)

    if len(targets) == 0:
        return 0, 0

    if n_occurrences is not None:
        targets = targets[-n_occurrences:]

    y = targets.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression().fit(x, y)
    lin_coef = model.coef_[0][0]

    peak_env = y.max() - y.min()

    peak_factor = min(1.0, window_size / 3)
    peak_env *= peak_factor

    var_segment = np.var(segments)
    lin_coef *= (1 + var_segment)

    return lin_coef, peak_env

# --- Nouveau modèle linéaire basé sur résultats pré-calculés ---
def predict_from_arround_last(series):
    """
    Extrait les 5 features principales autour des dernières occurrences
    et applique un modèle linéaire avec les coefficients connus.
    """
    window_map = {
        'Lin_Arround7Last': 3,
        'Peak_Arround7Last': 3,
        'Lin_Arround11Last': 5,
        'Peak_Arround11Last': 5,
        'Lin_Arround15Last': 7,
        'Peak_Arround15Last': 7
    }

    # Extraire les features
    features = {}
    for name, w in window_map.items():
        lin, peak = arround_last_features_correctifs(series, window_size=w, n_occurrences=7)
        if 'Lin' in name:
            features[name] = lin
        else:
            features[name] = peak

    # Coefficients linéaires appris
    coefs = {
        'Peak_Arround15Last': 0.8760847129389883,
        'Peak_Arround7Last': -0.3850933042814028,
        'Peak_Arround11Last': -0.37884174555140293,
        'Lin_Arround15Last': 3.526328493609997,
        'Lin_Arround7Last': -2.5027887320743583
    }

    # Sélection des features utilisées (top 5)
    selected_features = ['Peak_Arround15Last', 'Peak_Arround7Last', 'Peak_Arround11Last',
                         'Lin_Arround15Last', 'Lin_Arround7Last']

    # Prédiction linéaire
    pred = 0
    for f in selected_features:
        pred += features.get(f, 0) * coefs[f]

    return pred

def compute_features2(series, window_size):
    s = np.array(series)

    if len(s) < window_size:
        window = s
    else:
        window = s[-window_size:]

    # Moyenne locale (signal principal)
    mean_val = np.mean(window)

    # Fréquence de buts >= 2 (capacité à gagner)
    freq_ge2 = np.sum(window >= 2) / len(window)

    # Stabilité (inverse de variance)
    var = np.var(window)
    stability = 1 / (1 + var)

    return np.array([
        mean_val,
        freq_ge2,
        stability
    ])

def weighted_prediction(features_array, weights_dict, feature_names):
    pred = 0.0
    for i, fname in enumerate(feature_names):
        if fname in weights_dict:
            pred += features_array[i] * weights_dict[fname]
    return pred

class PredictorGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Prediction Scores exacts")
        self.match_names = []

        frame_buttons = Frame(self.root)
        frame_buttons.pack(pady=5)

        btn_load = Button(frame_buttons, text="CHARGER UN DOSSIER", command=self.load_folder)
        btn_load.pack(side="left", padx=5)

        btn_prev = Button(frame_buttons, text="Prev.", command=self.prev_csv)
        btn_prev.pack(side="left", padx=5)

        btn_next = Button(frame_buttons, text="Next.", command=self.next_csv)
        btn_next.pack(side="left", padx=5)

        # ---- Frame horizontal pour 3 logs ----
        frame_logs = Frame(self.root)
        frame_logs.pack(fill="both", expand=True)

        self.log = Text(frame_logs, height=10, width=180)  # log général
        self.log.pack(side="top", fill="x", expand=False)

        # --- BARRE DE RECHERCHE ---
        self.search_var = StringVar()

        self.search_entry = Entry(frame_logs, textvariable=self.search_var)
        self.search_entry.pack(fill="x")

        # événement saisie
        self.search_var.trace_add("write", self.on_search)

        # --- LISTE DES MATCHS ---
        frame_list = Frame(frame_logs)
        frame_list.pack(fill="x")

        self.match_list = Listbox(frame_list, height=6)
        self.match_list.pack(side="left", fill="x", expand=True)

        scrollbar = Scrollbar(frame_list, orient="vertical")
        scrollbar.config(command=self.match_list.yview)
        scrollbar.pack(side="right", fill="y")

        self.match_list.config(yscrollcommand=scrollbar.set)

        # événement clic
        self.match_list.bind("<<ListboxSelect>>", self.on_select_match)

        self.log1 = Text(frame_logs, height=40, width=60)  # log PATTERN
        self.log1.pack(side="left", fill="both", expand=True)

        self.log2 = Text(frame_logs, height=40, width=60)  # log VERSION 1
        self.log2.pack(side="left", fill="both", expand=True)

        self.log3 = Text(frame_logs, height=40, width=60)  # log VERSION 2
        self.log3.pack(side="left", fill="both", expand=True)

        self.csv_files = []
        self.current_index = -1
        self.data = None

        self.X_all = []
        self.y_all = []
        self.model = self.create_model()
        self.scaler = StandardScaler()

    def create_model(self):
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('svr', SVR())
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=2
        )

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # --- CHARGEMENT DES CSV ---
            self.csv_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".csv")
            ]
            self.csv_files.sort()

            # --- RESET LISTE ---
            self.match_list.delete(0, END)
            self.match_names = []

            # --- REMPLISSAGE LISTBOX ---
            for path in self.csv_files:
                try:
                    data = pd.read_csv(path, header=None)
                    teamA_name = data.iloc[322, 0]
                    teamC_name = data.iloc[322, 2]
                except:
                    teamA_name = "A"
                    teamC_name = "C"

                display_name = f"{teamA_name} VS {teamC_name}"
                self.match_names.append(display_name)
                self.match_list.insert(END, display_name)

            # --- AUTO LOAD PREMIER CSV ---
            if self.csv_files:
                self.current_index = 0
                self.load_csv(self.csv_files[self.current_index])

                # sélection visuelle du premier élément
                self.match_list.selection_clear(0, END)
                self.match_list.selection_set(0)
                self.match_list.activate(0)

    def load_csv(self, path):
        self.data = pd.read_csv(path, header=None)

        # --- EXTRACTION NOM FICHIER ---
        filename = os.path.basename(path)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split("_")

        heure = parts[0] if len(parts) > 0 else "N/A"
        sport = parts[1] if len(parts) > 1 else "N/A"
        nb_scores = parts[2] if len(parts) > 2 else "N/A"

        # --- NOMS DES ÉQUIPES (ligne 323 → index 322) ---
        try:
            teamA_name = self.data.iloc[322, 0]
            teamC_name = self.data.iloc[322, 2]
        except IndexError:
            teamA_name = "N/A"
            teamC_name = "N/A"

        # --- LOG ---
        self.log.delete('1.0', END)
        self.log.insert(END, f"Fichier : {filename}\n")
        self.log.insert(END, f"Heure : {heure}\n")
        self.log.insert(END, f"Sport : {sport}\n")
        self.log.insert(END, f"Scores historique : {nb_scores}\n")
        self.log.insert(END, f"Teams → A: {teamA_name} | C: {teamC_name}\n")

        # --- EXTRACTION COLONNES BRUTES ---
        self.teamA_scores = self.data.iloc[:, 0].tolist()
        self.teamC_scores = self.data.iloc[:, 2].tolist()

        # --- EXTRACTION APRÈS MARQUEURS ---
        seriesA = read_numeric_after_marker(self.teamA_scores, "A")
        seriesC = read_numeric_after_marker(self.teamC_scores, "C")

        # --- FALLBACK si vide ---
        if len(seriesA) == 0:
            seriesA = self.data.iloc[1:321, 0].values

        if len(seriesC) == 0:
            seriesC = self.data.iloc[1:321, 2].values

        # --- FORMAT FINAL ---
        self.team1_series = np.array(seriesA, dtype=float)
        self.team2_series = np.array(seriesC, dtype=float)

        self.X_all = []
        self.y_all = []

        for w in WINDOW_SIZES:
            featA_values = compute_features(self.team1_series, window_size=w)
            featC_values = compute_features(self.team2_series, window_size=w)

            self.X_all.append(featA_values)
            self.y_all.append(self.team1_series[-1])

            self.X_all.append(featC_values)
            self.y_all.append(self.team2_series[-1])

        X_train = np.array(self.X_all)
        y_train = np.array(self.y_all)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        self.model.fit(X_train_scaled, y_train)

        self.predict()

    def on_select_match(self, event):
        selection = self.match_list.curselection()
        if not selection:
            return

        index = selection[0]
        self.current_index = index
        self.load_csv(self.csv_files[index])

    def on_search(self, *args):
        query = self.search_var.get().lower()

        # reset sélection
        self.match_list.selection_clear(0, END)

        if not query:
            return

        first_match_index = None

        for i, name in enumerate(self.match_names):
            if query in name.lower():
                self.match_list.selection_set(i)

                if first_match_index is None:
                    first_match_index = i

        # scroll vers premier résultat
        if first_match_index is not None:
            self.match_list.see(first_match_index)
            self.match_list.activate(first_match_index)

    def predict_mean_recent2(self, recent_n=5):
        if self.data is None:
            return None, None, 1, 1

        def compute_pred(series):
            preds = []

            for w in [3,5,7]:
                segments, targets = get_non_overlapping_future_segments(series, w)

                if len(targets) == 0:
                    continue

                # limiter au recent_n derniers patterns
                if len(targets) >= recent_n:
                    segments = segments[-recent_n:]
                    targets = targets[-recent_n:]

                # --- moyenne globale des targets ---
                mean_val = np.mean(targets)

                # --- tendance ---
                x = np.arange(len(targets)).reshape(-1,1)
                model = LinearRegression().fit(x, targets)
                trend = model.predict(np.array([[len(targets)]]))[0] - targets[-1]

                # --- booster (récent vs global) ---
                global_mean = np.mean(series)
                booster = mean_val / global_mean if global_mean != 0 else 1

                # --- peak sur targets ---
                peak_env = np.max(targets) - np.min(targets)

                pred = (mean_val * booster) + trend + peak_env * 0.1

                preds.append(pred)

            if len(preds) == 0:
                return 0, 1

            return np.mean(preds), booster

        pred_team1, booster_team1 = compute_pred(self.team1_series)
        pred_team2, booster_team2 = compute_pred(self.team2_series)

        return pred_team1, pred_team2, booster_team1, booster_team2

    def predict_score(self, series, window_sizes=[3,5,7]):
        preds = []

        for w in window_sizes:
            segments, targets = get_non_overlapping_future_segments(series, w)

            if len(targets) == 0:
                continue

            # --- FEATURES calculées sur segments ---
            # moyenne des segments
            mean_val = np.mean(segments)

            # fréquence des valeurs >= 2 dans segments
            freq_ge2 = np.sum(segments >= 2) / segments.size

            # stabilité (inverse variance des segments)
            var = np.var(segments)
            stability = 1 / (1 + var)

            features = np.array([
                mean_val,
                freq_ge2,
                stability
            ])

            pred = weighted_prediction(features, FEATURE_WEIGHTS2, PLAN_FEATURE_NAMES2)

            # on peut aussi combiner avec la moyenne des targets
            pred = (pred + np.mean(targets)) / 2

            preds.append(pred)

        if len(preds) == 0:
            return 0

        return np.mean(preds)

    def predict(self):
        if self.data is None:
            self.log.insert('end', "Aucun CSV chargé.\n")
            return

        try:
            col_e1 = self.data.iloc[0, 4]  
            col_g1 = self.data.iloc[0, 6]  
            self.log.insert('end', f"Résultat → ⚽= {col_e1} - {col_g1}\n")
        except IndexError:
            self.log.insert('end', "Colonnes E ou G manquantes.\n")

        self.log.insert('end', f"Derniers scores → ⚽= A: {self.team1_series[-1]} || C: {self.team2_series[-1]}\n")
    
        # --- PATTERN ---
        self.log1.delete('1.0', END)
        self.log1.insert('end', f"\nPATTERN\n")
        for window_size in WINDOW_SIZES:
            featuresA = arround_last_features_pattern(self.team1_series, window_size)
            featuresC = arround_last_features_pattern(self.team2_series, window_size)
            self.log1.insert('end', f"\n=== Window size: {window_size} ===\n")

            # Filtrage spécifique par window_size
            if window_size == 1:
                keys_to_show = ["Peak"]
            elif window_size == 2:
                keys_to_show = ["HilbertMax"]
            else:
                keys_to_show = []

            for k in keys_to_show:
                self.log1.insert('end', f"{k:<18} → ⚽= A: {featuresA[k]:>6.2f} | C: {featuresC[k]:>6.2f}\n")

        # --- VERSION 1 ---
        self.log2.delete('1.0', END)
        self.log2.insert('end', f"\nVERSION 1\n")
        for window_size in WINDOW_SIZES2:
            featuresA = arround_last_features(self.team1_series, window_size)
            featuresC = arround_last_features(self.team2_series, window_size)
            self.log2.insert('end', f"\n=== Window size: {window_size} ===\n")

            # Filtrage spécifique par window_size
            if window_size == 4:
                keys_to_show = ["Peak"]
            elif window_size == 1:
                keys_to_show = ["HilbertMax"]
            else:
                keys_to_show = []

            for k in keys_to_show:
                self.log2.insert('end', f"{k:<18} → ⚽= A: {featuresA[k]:>6.2f} | C: {featuresC[k]:>6.2f}\n")

        # VERSION 3, 4, 5 (une seule fois)
        self.log3.delete('1.0', END)  # <-- ajouté pour vider le log avant affichage
        predA = self.predict_score(self.team1_series)
        predC = self.predict_score(self.team2_series)
        self.log3.insert('end', f"V3 → ⚽= A: {predA:.2f} | C: {predC:.2f}\n")

        predA2, predC2, boosterA, boosterC = self.predict_mean_recent2(recent_n=5)
        self.log3.insert('end', f"V4 → ⚽= A: {predA2:.2f} | C: {predC2:.2f} | B. A: {boosterA:.2f} | B. C: {boosterC:.2f}\n")

        predA5 = predict_from_arround_last(self.team1_series)
        predC5 = predict_from_arround_last(self.team2_series)
        self.log3.insert('end', f"V5 → ⚽= A: {predA5:.2f} | C: {predC5:.2f}\n")


    def prev_csv(self):
        if self.csv_files and self.current_index > 0:
            self.current_index -= 1
            self.load_csv(self.csv_files[self.current_index])

    def next_csv(self):
        if self.csv_files and self.current_index < len(self.csv_files) - 1:
            self.current_index += 1
            self.load_csv(self.csv_files[self.current_index])

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PredictorGUI()
    app.run()
