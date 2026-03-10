#!/bin/bash

BASE_DIR="/home/annick/SCRIPTS"
VENV_DIR="$BASE_DIR/venv_flashscore"
PYTHON="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

cd "$BASE_DIR" || exit 1

# 1️⃣ Création du venv s'il n'existe pas
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Création du venv"
    python3 -m venv "$VENV_DIR"
fi

# 2️⃣ Activation du venv
source "$VENV_DIR/bin/activate"

# 3️⃣ Upgrade pip
"$PIP" install --upgrade pip

# 4️⃣ Installation dépendances (une seule fois si déjà installées)
"$PIP" install \
    customtkinter \
    selenium \
    beautifulsoup4 \
    pandas \
    numpy \
    lxml \
    requests \
    webdriver-manager \
    scipy \
    send2trash \
    playwright 

# 5️⃣ INSTALLATION DES NAVIGATEURS PLAYWRIGHT (OBLIGATOIRE)
"$PYTHON" -m playwright install firefox

# 6️⃣ Lancement de la GUI
#"$PYTHON" "SPORTS PARSER LAUNCHER(GRILLE).py"
"$PYTHON" "SPORTS PARSER LAUNCHER (COLONNE).py"
