"""
SPIKE (desechable) — detección automática de tempo de los 11 instrumentales.

Pregunta que valida: ¿librosa detecta el BPM de estos instrumentales de forma
confiable, y puede decir si el tempo es CONSTANTE o VARIABLE? Marcos midió Abzu a
mano = 128 BPM; este spike se valida contra eso (si da 64 o 256 es "error de
octava", típico de la detección por autocorrelación → se reporta para que Marcos
lo confirme de oído).

No toca el código del proyecto ni data/. Solo lee data/01_instrumentales/ e imprime
un cuadro. Salida opcional a spikes/tempo/salida/tempo.json (gitignored).

Uso:
  .venv\\Scripts\\python spikes/tempo/spike_tempo.py            # los 11
  .venv\\Scripts\\python spikes/tempo/spike_tempo.py 1 10       # algunos
"""

import os
import sys
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import librosa

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_INSTR = os.path.join(ROOT, "data", "01_instrumentales")
DIR_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida")
SR = 22050  # estándar de análisis de librosa; suficiente para tempo


def _ruta_instrumental(num):
    for f in sorted(os.listdir(DIR_INSTR)):
        if f.lower().endswith(".wav") and f.startswith(f"{num:02d}-"):
            return os.path.join(DIR_INSTR, f)
    return None


def _candidatos_octava(bpm):
    """BPM en mitad/doble — la detección suele equivocarse de octava rítmica."""
    return sorted({round(bpm / 2, 1), round(bpm, 1), round(bpm * 2, 1)})


def analizar_tema(num):
    ruta = _ruta_instrumental(num)
    if not ruta:
        return None
    nombre = os.path.basename(ruta)

    t0 = time.time()
    y, sr = librosa.load(ruta, sr=SR, mono=True)
    dur = len(y) / sr

    # Envolvente de onsets (percusión/ataques) = base para el tempo.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # (1) Tempo GLOBAL + posiciones de los beats.
    tempo_global, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_global = float(np.atleast_1d(tempo_global)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # (2) CONSTANTE vs VARIABLE — dos miradas:
    #   a) estabilidad de los intervalos entre beats detectados.
    bpm_inst = 60.0 / np.diff(beat_times) if len(beat_times) > 2 else np.array([])
    if bpm_inst.size:
        bpm_med = float(np.median(bpm_inst))
        cv = float(np.std(bpm_inst) / np.mean(bpm_inst))  # coef. de variación
    else:
        bpm_med, cv = tempo_global, float("nan")

    #   b) tempo dinámico por ventanas (estimación independiente de los beats).
    try:
        tempo_din = librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr, aggregate=None
        )
        tempo_din = np.atleast_1d(tempo_din).astype(float)
        din_p10, din_p90 = (float(np.percentile(tempo_din, 10)),
                            float(np.percentile(tempo_din, 90)))
        din_unicos = len(set(np.round(tempo_din).tolist()))
    except Exception as e:
        din_p10 = din_p90 = float("nan")
        din_unicos = -1

    # Veredicto heurístico de estabilidad (lo confirma el oído de Marcos).
    if not np.isnan(cv):
        if cv < 0.04:
            veredicto = "CONSTANTE"
        elif cv < 0.10:
            veredicto = "casi constante"
        else:
            veredicto = "VARIABLE (revisar)"
    else:
        veredicto = "?"

    return {
        "num": num,
        "nombre": nombre,
        "dur_s": round(dur, 1),
        "bpm_global": round(tempo_global, 1),
        "bpm_mediano_beats": round(bpm_med, 1),
        "cv_intervalos": round(cv, 4) if not np.isnan(cv) else None,
        "veredicto": veredicto,
        "bpm_dinamico_p10_p90": [round(din_p10, 1), round(din_p90, 1)],
        "candidatos_octava": _candidatos_octava(tempo_global),
        "n_beats": int(len(beat_times)),
        "primer_beat_s": round(float(beat_times[0]), 3) if len(beat_times) else None,
        "calc_s": round(time.time() - t0, 1),
    }


def main():
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 12))
    resultados = []
    print(f"[i] Analizando tempo de los temas {nums}\n")
    print(f"{'#':>2}  {'BPM glob':>8}  {'BPM med':>7}  {'CV':>6}  "
          f"{'veredicto':<16}  {'octava cand.':<20}  {'1er beat':>8}  tema")
    print("-" * 100)
    for num in nums:
        r = analizar_tema(num)
        if not r:
            print(f"{num:>2}  (sin instrumental, salto)")
            continue
        resultados.append(r)
        print(f"{r['num']:>2}  {r['bpm_global']:>8}  {r['bpm_mediano_beats']:>7}  "
              f"{str(r['cv_intervalos']):>6}  {r['veredicto']:<16}  "
              f"{str(r['candidatos_octava']):<20}  {str(r['primer_beat_s']):>8}  "
              f"{r['nombre']}")

    os.makedirs(DIR_SALIDA, exist_ok=True)
    ruta_json = os.path.join(DIR_SALIDA, "tempo.json")
    json.dump(resultados, open(ruta_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[OK] Detalle en {ruta_json}")
    print("[i] VALIDAR: Abzu (01) debería dar ~128. Si da ~64 o ~256 es error de "
          "octava (mirar 'octava cand.').")


if __name__ == "__main__":
    main()
