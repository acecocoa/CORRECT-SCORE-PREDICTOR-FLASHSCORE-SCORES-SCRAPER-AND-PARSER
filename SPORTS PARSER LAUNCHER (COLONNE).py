import customtkinter as ctk
import subprocess
import threading
import os
import sys

# ================== CONFIG ==================
BASE_DIR = "/home/annick/SCRIPTS"
VENV_DIR = os.path.join(BASE_DIR, "venv_flashscore")
PYTHON_VENV = os.path.join(VENV_DIR, "bin", "python3")

SCRIPT_0 = "FLASHSCORE TODAY SCRAPER.py"
SCRIPT_10 = "FLASHSCORE TODAY SCRAPER(LIMIT).py"
SCRIPT_1 = "FLASHSCORE LEAGUE SCRAPER.py"
SCRIPT_11 = "FLASHSCORE MATCH SCRAPER.py"
SCRIPT_2 = "PREDICTOR(F4)(short).py"
SCRIPT_4 = "PREDICTOR_DRAW_CS.py"

REQUIREMENTS = [
    "pip",
    "selenium",
    "beautifulsoup4",
    "pandas",
    "numpy",
    "lxml",
    "requests",
    "webdriver-manager",
    "scipy",
    "send2trash"
]

process = None
# ============================================


def update_status(text, color="white"):
    status_label.configure(text=text, text_color=color)


def setup_venv():
    if not os.path.exists(VENV_DIR):
        update_status("Création de l'environnement virtuel...", "orange")
        subprocess.run(["python3", "-m", "venv", VENV_DIR], cwd=BASE_DIR)

    update_status("Installation des dépendances...", "orange")
    PIP_VENV = os.path.join(VENV_DIR, "bin", "pip")

    subprocess.run([PIP_VENV, "install", "--upgrade", "pip"])
    subprocess.run([PIP_VENV, "install"] + REQUIREMENTS)

processes = {}  # dictionnaire pour stocker un processus par script

def run_script(script_name):
    if script_name in processes and processes[script_name] is not None:
        update_status(f"{script_name} est déjà en cours", "red")
        return

    def task():
        try:
            update_status("Initialisation...", "orange")
            setup_venv()

            update_status(f"Lancement : {script_name}", "green")
            proc = subprocess.Popen(
                [PYTHON_VENV, script_name],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            processes[script_name] = proc

            for line in proc.stdout:
                update_status(f"[{script_name}] {line.strip()}", "white")

            proc.wait()
            update_status(f"{script_name} terminé", "cyan")

        except Exception as e:
            update_status(f"Erreur ({script_name}) : {e}", "red")

        finally:
            processes[script_name] = None

    threading.Thread(target=task, daemon=True).start()


def stop_script(script_name=None):
    """Arrêter un script précis ou tous les scripts si script_name=None"""
    if script_name:
        proc = processes.get(script_name)
        if proc:
            proc.terminate()
            processes[script_name] = None
            update_status(f"{script_name} arrêté", "red")
    else:
        for key, proc in processes.items():
            if proc:
                proc.terminate()
                processes[key] = None
        update_status("Tous les scripts arrêtés", "red")


# ================== GUI ==================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("FLASHSCORE SCRAPER LAUNCHER")
app.geometry("425x1000") # H max= x1000

title = ctk.CTkLabel(
    app,
    text="FLASHSCORE SCORES SCRAPER",
    font=("Arial", 28, "bold")
)
title.pack(pady=30)

btn0 = ctk.CTkButton(
    app,
    text="TODAY SCRAPER",
    height=60,
    font=("Arial", 18),
    command=lambda: run_script(SCRIPT_0)
)
btn0.pack(pady=20)

btn10 = ctk.CTkButton(
    app,
    text="TODAY SCRAPER(LIMIT)",
    height=60,
    font=("Arial", 18),
    command=lambda: run_script(SCRIPT_10)
)
btn10.pack(pady=20)

btn1 = ctk.CTkButton(
    app,
    text="LEAGUES SCRAPER",
    height=60,
    font=("Arial", 18),
    command=lambda: run_script(SCRIPT_1)
)
btn1.pack(pady=20)

btn11 = ctk.CTkButton(
    app,
    text="MATCH SCRAPER",
    height=60,
    font=("Arial", 18),
    command=lambda: run_script(SCRIPT_11)
)
btn11.pack(pady=20)

title = ctk.CTkLabel(
    app,
    text="PREDICTORS",
    font=("Arial", 28, "bold")
)
title.pack(pady=30)

btn2 = ctk.CTkButton(
    app,
    text="PREDICTOR F4", 
    height=60,
    font=("Arial", 18),
    command=lambda: run_script(SCRIPT_2)
)
btn2.pack(pady=20)

btn4 = ctk.CTkButton(
    app,
    text="PREDICTOR DRAW_CS", 
    height=60,
    font=("Arial", 18),
    command=lambda: run_script(SCRIPT_4)
)
btn4.pack(pady=20)

stop_btn = ctk.CTkButton(
    app,
    text="STOP SCRIPT",
    height=50,
    fg_color="red",
    hover_color="#aa0000",
    font=("Arial", 16),
    command=stop_script
)
stop_btn.pack(pady=40)

status_label = ctk.CTkLabel(
    app,
    text="Aucun script en cours",
    font=("Arial", 16)
)
status_label.pack(pady=20)

app.mainloop()
