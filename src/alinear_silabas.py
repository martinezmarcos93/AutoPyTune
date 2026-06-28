"""
v3 del estudio rítmico — timing REAL de la sílaba tónica (forced alignment a nivel
de carácter con MMS_FA).

`src/alinear_forzado.py` ya alinea el texto exacto contra la voz IA con MMS_FA y
obtiene spans por CARÁCTER, pero los colapsa a inicio/fin de PALABRA. Acá reusamos
el mismo modelo (ya descargado, sin dependencias nuevas) y conservamos el tiempo de
cada carácter para ubicar el ataque de la VOCAL TÓNICA de cada palabra. Eso elimina
la estimación equiespaciada de la v2 (que se rompía en melismas).

Salida (sidecar nuevo, no toca nada): `data/07_karaoke/NN-Tema.silabas.json`
  [{palabra, inicio, fin, silabas, silaba_tonica, t_tonica}]
que `src/analizar_ritmo.py` consume con prioridad sobre la estimación.

~110s/tema en CPU (mismo costo que alinear_forzado). Uso:
  .venv\\Scripts\\python src/alinear_silabas.py            # los 11
  .venv\\Scripts\\python src/alinear_silabas.py 1 2 7      # validar algunos
"""

import os
import sys
import json
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import librosa
import torch

import alinear_forzado as af   # modelo MMS, _normalizar, _ruta_voz, _base, rutas
import analizar_ritmo as ar    # _tonica_vocal_pos (silabificador del español)

DIR_KARAOKE = af.DIR_KARAOKE
SR = af.SR                     # 16000 (lo exige MMS)


def _indice_tonica_normalizado(palabra, validos):
    """Índice (en la palabra NORMALIZADA al dict MMS) de la vocal tónica.
    Devuelve (idx_norm | None, idx_silaba, n_silabas)."""
    pos_vocal, idx_sil, n_sil, limpia = ar._tonica_vocal_pos(palabra)
    if pos_vocal is None:
        return None, idx_sil, n_sil
    # Reproducir la normalización de af._normalizar carácter a carácter, mapeando
    # la posición en `limpia` a la posición en la cadena normalizada (la que el
    # alineador tokenizó). Solo los caracteres descartados corren el índice.
    idx_norm = None
    cuenta = 0
    for p, c in enumerate(limpia):
        base = unicodedata.normalize("NFD", c)
        base = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
        base = base.lower().replace("ñ", "n")
        base = "".join(ch for ch in base if ch in validos)
        if not base:
            continue
        if p == pos_vocal:
            idx_norm = cuenta
        cuenta += len(base)
    return idx_norm, idx_sil, n_sil


def alinear_tema(num, model, tokenizer, aligner, validos):
    ruta = af._ruta_voz(num)
    base = af._base(num)
    if not ruta or not base:
        print(f"{num:>2}  (falta voz o .txt, salto)")
        return
    display, norm = [], []
    for p in af._palabras_del_texto(base):
        n = af._normalizar(p, validos)
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
    spf = dur / emission.size(1)

    salida = []
    sin_tonica = 0
    for i, palabra in enumerate(display):
        char_spans = spans[2 * i + 1]          # spans por carácter de la palabra i
        inicio = char_spans[0].start * spf
        fin = char_spans[-1].end * spf
        idx_norm, idx_sil, n_sil = _indice_tonica_normalizado(palabra, validos)
        if idx_norm is not None and 0 <= idx_norm < len(char_spans):
            t_ton = char_spans[idx_norm].start * spf    # ataque real de la vocal tónica
        else:
            t_ton = inicio + (idx_sil / n_sil) * (fin - inicio) if n_sil else inicio
            sin_tonica += 1
        salida.append({
            "palabra": palabra,
            "inicio": round(float(inicio), 3),
            "fin": round(float(fin), 3),
            "silabas": n_sil,
            "silaba_tonica": idx_sil + 1,
            "t_tonica": round(float(t_ton), 3),
        })

    ruta_json = os.path.join(DIR_KARAOKE, base + ".silabas.json")
    json.dump(salida, open(ruta_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{num:>2}  {base}.silabas.json — {len(salida)} palabras "
          f"({sin_tonica} sin tónica mapeada), {dur:.0f}s, {time.time()-t0:.0f}s")


def main():
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 12))
    print("[i] Cargando modelo MMS_FA (ya en cache de torch)...")
    model = af.bundle.get_model()
    tokenizer = af.bundle.get_tokenizer()
    aligner = af.bundle.get_aligner()
    validos = set(af.bundle.get_dict().keys())
    print(f"[i] Temas a alinear (sílaba): {nums}\n")
    for num in nums:
        alinear_tema(num, model, tokenizer, aligner, validos)
    print("\n[OK] Sidecars data/07_karaoke/NN-Tema.silabas.json escritos.")
    print("[i] Ahora: .venv\\Scripts\\python src/analizar_ritmo.py  (usará la tónica REAL)")


if __name__ == "__main__":
    main()
