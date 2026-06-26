"""
Etapa 2 (timing) — forced alignment del texto exacto contra la voz IA.

Toma el texto perfecto de `data/07_karaoke/NN-Tema.txt` (la verdad, ver
`src/compilar_letras.py`) y lo alinea contra la voz IA aislada con el modelo
multilingüe MMS_FA de torchaudio (CTC forced alignment). Devuelve el tiempo de
cada palabra y escribe `data/07_karaoke/NN-Tema.json` ({palabra, inicio, fin}),
que es lo que consume el karaoke de la GUI.

El texto MANDA: la alineación es monótona, así que respeta las repeticiones. Se
intercala el token estrella `*` entre palabras para que los tramos instrumentales
y silencios se absorban en `*` y no estiren palabras reales sobre ellos.

Sin dependencias nuevas: torch/torchaudio ya están (los usa demucs). El modelo
MMS (~1.18 GB) se descarga la 1ra vez al cache de torch.

Uso:
  .venv\\Scripts\\python src/alinear_forzado.py            # los 11
  .venv\\Scripts\\python src/alinear_forzado.py 1 10       # solo algunos
"""

import os
import sys
import json
import time
import shutil
import unicodedata

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import librosa
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_VOCES = os.path.join(ROOT, "data", "04_voces_extraidas")
DIR_KARAOKE = os.path.join(ROOT, "data", "07_karaoke")
SR = bundle.sample_rate  # 16000
DESCARTAR = (".asr.txt", ".difflib.txt", ".gemini.txt")


def _ruta_voz(num):
    for f in os.listdir(DIR_VOCES):
        if f.lower().endswith(".wav") and f.startswith(f"{num:02d}-") and "_voz_ia" in f:
            return os.path.join(DIR_VOCES, f)
    return None


def _base(num):
    for f in os.listdir(DIR_KARAOKE):
        if f.startswith(f"{num:02d}-") and f.endswith(".txt") and not f.endswith(DESCARTAR):
            return f[:-4]
    return None


def _normalizar(palabra, validos):
    """Romaniza a los chars del diccionario MMS (sin tildes, minúsculas, ñ->n)."""
    base = unicodedata.normalize("NFD", palabra)
    base = "".join(c for c in base if unicodedata.category(c) != "Mn")
    base = base.lower().replace("ñ", "n")
    return "".join(c for c in base if c in validos)


def _palabras_del_texto(base):
    ruta = os.path.join(DIR_KARAOKE, base + ".txt")
    palabras = []
    for linea in open(ruta, encoding="utf-8").read().splitlines():
        palabras.extend(linea.split())
    return palabras


def alinear_tema(num, model, tokenizer, aligner, validos):
    ruta = _ruta_voz(num)
    base = _base(num)
    if not ruta or not base:
        print(f"[X] Falta voz o .txt del tema {num}, salto.")
        return
    display, norm = [], []
    for p in _palabras_del_texto(base):
        n = _normalizar(p, validos)
        if n:
            display.append(p)
            norm.append(n)

    audio, _ = librosa.load(ruta, sr=SR, mono=True)
    pico = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if pico > 0:
        audio = audio * (0.95 / pico)
    wav = torch.from_numpy(audio).unsqueeze(0)
    dur = audio.shape[0] / SR

    seq = ["*"]
    for n in norm:
        seq.append(n)
        seq.append("*")

    t0 = time.time()
    with torch.inference_mode():
        emission, _ = model(wav)
        spans = aligner(emission[0], tokenizer(seq))
    seg_por_frame = dur / emission.size(1)

    palabras = []
    for i, pal in enumerate(display):
        sp = spans[2 * i + 1]
        palabras.append({
            "palabra": pal,
            "inicio": round(sp[0].start * seg_por_frame, 3),
            "fin": round(sp[-1].end * seg_por_frame, 3),
        })

    ruta_json = os.path.join(DIR_KARAOKE, base + ".json")
    respaldo = os.path.join(DIR_KARAOKE, base + ".difflib.json")
    if os.path.exists(ruta_json) and not os.path.exists(respaldo):
        shutil.copy2(ruta_json, respaldo)
    json.dump(palabras, open(ruta_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[OK] {base}.json — {len(palabras)} palabras, "
          f"{dur:.0f}s audio, alineado en {time.time()-t0:.0f}s")


def main():
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 12))
    print(f"[i] Cargando modelo MMS_FA (1ra vez descarga ~1.18 GB)...")
    model = bundle.get_model()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    validos = set(bundle.get_dict().keys())
    print(f"[i] Temas a alinear: {nums}\n")
    for num in nums:
        alinear_tema(num, model, tokenizer, aligner, validos)


if __name__ == "__main__":
    main()
