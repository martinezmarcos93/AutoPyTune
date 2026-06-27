# -*- mode: python ; coding: utf-8 -*-
"""
Empaquetado one-folder de AutoPyTune para correr SIN instalar dependencias.
Incluye el núcleo (GUI + motor de audio + karaoke) y EXCLUYE el preprocesamiento
pesado (torch/demucs/whisper/gemini), que ya se hizo y no se necesita para grabar.
Build:  .venv\\Scripts\\pyinstaller AutoPyTune.spec --noconfirm
Salida: dist/AutoPyTune/AutoPyTune.exe  (+ se le copia data/ al lado)
"""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("librosa", "soundfile", "sounddevice", "pedalboard", "psola",
            "soxr", "audioread", "lazy_loader", "numba", "llvmlite"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

excludes = [
    "torch", "torchaudio", "demucs", "faster_whisper", "ctranslate2",
    "google", "transformers", "onnxruntime", "julius", "openunmix",
    "matplotlib", "IPython", "tkinter", "tensorflow", "yt_dlp", "truststore",
]

a = Analysis(
    ["src/gui.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AutoPyTune",
    console=False,           # app de ventana: sin consola al doble-click.
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="AutoPyTune")
