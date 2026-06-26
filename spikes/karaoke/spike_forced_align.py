"""
SPIKE (desechable) — Etapa 2: timing por palabra con forced alignment.

Valida el punto riesgoso: ¿torchaudio MMS_FA (forced alignment, multilingüe, CTC)
alinea el TEXTO EXACTO ya conocido (data/07_karaoke/NN-Tema.txt) contra la voz IA
aislada y devuelve timestamps por palabra fiables, respetando repeticiones y
aguantando el canto/gritos? Sin dependencias nuevas (torch/torchaudio ya están por
demucs). El texto manda; la alineación solo le pone tiempos.

Uso:
  .venv\\Scripts\\python spikes/karaoke/spike_forced_align.py 1   # tema 1 (default)
"""

import os
import sys
import json
import time
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_VOCES = os.path.join(ROOT, "data", "04_voces_extraidas")
DIR_KARAOKE = os.path.join(ROOT, "data", "07_karaoke")
DIR_SALIDA = os.path.join(ROOT, "spikes", "karaoke", "salida")
SR = bundle.sample_rate  # 16000


def _ruta_voz(num):
    for f in os.listdir(DIR_VOCES):
        if f.lower().endswith(".wav") and f.startswith(f"{num:02d}-") and "_voz_ia" in f:
            return os.path.join(DIR_VOCES, f)
    return None


def _base(num):
    descartar = (".asr.txt", ".difflib.txt", ".gemini.txt")
    for f in os.listdir(DIR_KARAOKE):
        if f.startswith(f"{num:02d}-") and f.endswith(".txt") and not f.endswith(descartar):
            return f[:-4]
    return None


def _normalizar(palabra, validos):
    """Romaniza a los chars del diccionario MMS (sin acentos, minúsculas)."""
    base = unicodedata.normalize("NFD", palabra)
    base = "".join(c for c in base if unicodedata.category(c) != "Mn")  # sin tildes
    base = base.lower().replace("ñ", "n")
    return "".join(c for c in base if c in validos)


def main():
    num = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
    ruta = _ruta_voz(num)
    base = _base(num)
    if not ruta or not base:
        print(f"[X] Falta voz o .txt del tema {num}")
        return
    lineas = [l.strip() for l in open(os.path.join(DIR_KARAOKE, base + ".txt"),
                                      encoding="utf-8").read().splitlines() if l.strip()]
    palabras_txt = []
    for l in lineas:
        palabras_txt.extend(l.split())

    print(f"[i] Tema {num}: {base}")
    print(f"[i] Palabras del texto: {len(palabras_txt)}")

    print("[i] Cargando modelo MMS_FA...")
    model = bundle.get_model()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    validos = set(bundle.get_dict().keys())

    # Texto normalizado para alinear (guardando el mapeo a la palabra de display).
    display, norm = [], []
    for p in palabras_txt:
        n = _normalizar(p, validos)
        if n:
            display.append(p)
            norm.append(n)
    print(f"[i] Palabras alineables (no vacías al normalizar): {len(norm)}")

    audio, _ = librosa.load(ruta, sr=SR, mono=True)
    pico = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if pico > 0:
        audio = audio * (0.95 / pico)
    wav = torch.from_numpy(audio).unsqueeze(0)
    dur = audio.shape[0] / SR
    print(f"[i] Audio {dur:.0f}s — alineando (CPU, puede tardar)...")

    # Intercalar el token estrella '*' entre palabras: absorbe los tramos
    # instrumentales/silencios para que no se "estiren" palabras sobre ellos.
    seq = ["*"]
    for n in norm:
        seq.append(n)
        seq.append("*")

    t0 = time.time()
    with torch.inference_mode():
        emission, _ = model(wav)
        spans = aligner(emission[0], tokenizer(seq))
    num_frames = emission.size(1)
    seg_por_frame = dur / num_frames
    print(f"[i] Alineación en {time.time()-t0:.0f}s ({num_frames} frames)")

    # Las palabras reales están en posiciones impares (1,3,5,...); las pares son '*'.
    palabras = []
    for i, pal in enumerate(display):
        sp = spans[2 * i + 1]
        ini = round(sp[0].start * seg_por_frame, 3)
        fin = round(sp[-1].end * seg_por_frame, 3)
        palabras.append({"palabra": pal, "inicio": ini, "fin": fin})

    os.makedirs(DIR_SALIDA, exist_ok=True)
    out = os.path.join(DIR_SALIDA, f"{num:02d}_align.json")
    json.dump(palabras, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n--- primeras 12 palabras ---")
    for p in palabras[:12]:
        print(f"   {p['inicio']:7.2f}–{p['fin']:6.2f}  {p['palabra']}")
    print("--- últimas 6 ---")
    for p in palabras[-6:]:
        print(f"   {p['inicio']:7.2f}–{p['fin']:6.2f}  {p['palabra']}")

    monot = all(palabras[i]["inicio"] <= palabras[i+1]["inicio"] + 1e-6
                for i in range(len(palabras)-1))
    print(f"\n[i] 1ra palabra a {palabras[0]['inicio']:.2f}s | "
          f"última fin {palabras[-1]['fin']:.2f}s | dur {dur:.2f}s")
    print(f"[i] Monótono (sin retrocesos): {monot}")
    print(f"[i] Guardado: {out}")


if __name__ == "__main__":
    main()
