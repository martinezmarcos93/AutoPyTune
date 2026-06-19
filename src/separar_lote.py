"""
Separación por lotes de voz / instrumental con Demucs (htdemucs).

A diferencia de `audio_engine.separar_pistas` (que devuelve mono porque el
pipeline de afinación trabaja en mono), este script conserva el **estéreo**:
los instrumentales son para usarse como pista musical real, donde el estéreo
importa.

Carga el audio con librosa (no con torchaudio) para evitar la dependencia de
torchcodec que rompe el CLI de Demucs en torchaudio >= 2.11.

Uso:
    python src/separar_lote.py
    python src/separar_lote.py "data/00_originales"   # carpeta a procesar
"""

import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
from demucs import pretrained
from demucs.apply import apply_model

RAIZ = Path(__file__).resolve().parent.parent
DIR_ORIGINALES = RAIZ / "data" / "00_originales"
DIR_INSTRUMENTALES = RAIZ / "data" / "01_instrumentales"
DIR_VOCES = RAIZ / "data" / "04_voces_extraidas"

EXTENSIONES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def _preparar_audio(ruta, canales, model_sr):
    """Carga estéreo a la frecuencia del modelo y normaliza como espera Demucs."""
    audio, _ = librosa.load(str(ruta), sr=model_sr, mono=False)
    if audio.ndim == 1:                       # mono -> duplicar a estéreo
        audio = np.stack([audio, audio])
    if audio.shape[0] == 1:
        audio = np.repeat(audio, canales, axis=0)
    audio = audio[:canales]                   # recortar si trae más canales

    wav = torch.from_numpy(audio).float()
    ref = wav.mean(0)
    desv = ref.std() + 1e-8
    wav = (wav - ref.mean()) / desv
    return wav, ref.mean(), desv


def separar_archivo(ruta, modelo, device):
    """
    Devuelve (voz, instrumental) en estéreo (np.ndarray shape [2, n]).
    El instrumental es la suma de todas las fuentes menos 'vocals'.
    """
    model_sr = modelo.samplerate
    canales = modelo.audio_channels
    wav, media, desv = _preparar_audio(ruta, canales, model_sr)

    with torch.no_grad():
        fuentes = apply_model(
            modelo, wav[None], device=device, shifts=1, split=True, overlap=0.25
        )[0]
    fuentes = fuentes * desv + media          # deshacer normalización

    nombres = list(modelo.sources)            # ['drums','bass','other','vocals']
    idx_voz = nombres.index("vocals")
    voz = fuentes[idx_voz].cpu().numpy()
    instrumental = sum(
        fuentes[i].cpu().numpy() for i in range(len(nombres)) if i != idx_voz
    )
    return voz.astype(np.float32), instrumental.astype(np.float32), model_sr


def separar_lote(dir_entrada=DIR_ORIGINALES):
    DIR_INSTRUMENTALES.mkdir(parents=True, exist_ok=True)
    DIR_VOCES.mkdir(parents=True, exist_ok=True)

    archivos = sorted(
        p for p in Path(dir_entrada).iterdir()
        if p.suffix.lower() in EXTENSIONES
    )
    if not archivos:
        print(f"No hay audios en {dir_entrada}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}  |  archivos: {len(archivos)}")
    print("Cargando modelo htdemucs...")
    modelo = pretrained.get_model("htdemucs")
    modelo.to(device).eval()

    for i, ruta in enumerate(archivos, 1):
        nombre = ruta.stem
        salida_inst = DIR_INSTRUMENTALES / f"{nombre}_instrumental.wav"
        salida_voz = DIR_VOCES / f"{nombre}_voz_ia.wav"
        if salida_inst.exists() and salida_voz.exists():
            print(f"[{i}/{len(archivos)}] {nombre}  (ya existe, salto)")
            continue

        print(f"[{i}/{len(archivos)}] separando: {nombre} ...", flush=True)
        voz, instrumental, sr = separar_archivo(ruta, modelo, device)
        # soundfile espera [n, canales]
        sf.write(salida_inst, instrumental.T, sr)
        sf.write(salida_voz, voz.T, sr)
        print(f"    -> {salida_inst.name}  +  {salida_voz.name}", flush=True)

    print("\nListo. Instrumentales en data/01_instrumentales/, "
          "voces en data/04_voces_extraidas/")


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else DIR_ORIGINALES
    separar_lote(entrada)
