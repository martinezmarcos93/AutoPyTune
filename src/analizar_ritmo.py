"""
Incremento 1 (estudio) — análisis rítmico: grilla métrica ⨯ acentos. **v3**

Mide, por tema, si las SÍLABAS TÓNICAS de la versión Suno caen sobre los tiempos
fuertes del compás (el concepto de Marcos: el acento lingüístico coincide con el
acento métrico → suena natural). Cruza:
  - la GRILLA de beats del instrumental (BPM + fase del downbeat, vía librosa), y
  - el tiempo de la sílaba tónica de cada palabra cantada.

Fuente del tiempo de la tónica, por prioridad:
  1. SIDECAR `data/07_karaoke/NN-Tema.silabas.json` (timing de sílaba REAL, sacado del
     forced alignment a nivel de carácter — ver `src/alinear_silabas.py`). ← v3.
  2. Estimación: reparte las sílabas equiespaciadas en `[inicio, fin]` de la palabra
     (timing por palabra de `NN-Tema.json`). ← v2, fallback. Es impreciso en melismas.

Mejoras vigentes: filtra outliers del forced alignment (zonas sin beats / colas
sostenidas), separa palabras de CONTENIDO de las ÁTONAS, downbeat por banda grave.

SOLO LECTURA. Escribe un archivo NUEVO por tema: `data/07_karaoke/NN-Tema.grid.json`.

Uso:
  .venv\\Scripts\\python src/analizar_ritmo.py            # los 11
  .venv\\Scripts\\python src/analizar_ritmo.py 1 5        # algunos
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_INSTR = os.path.join(ROOT, "data", "01_instrumentales")
DIR_KARAOKE = os.path.join(ROOT, "data", "07_karaoke")
SR = 22050
PULSOS_POR_COMPAS = 4          # 4/4
FUERTES = (0, 2)               # tiempos fuertes en 4/4 (0 = el más fuerte)
TOL_FRAC = 1 / 4               # tolerancia de "en grilla": ±1/4 de pulso
FUERA_GRILLA_FRAC = 0.6        # |desfase| > 0.6 de pulso = zona sin beats / outlier
N_MELS_GRAVES = 16            # bandas mel bajas (~bombo) para estimar la fase
DESCARTAR = (".asr.json", ".difflib.json", ".grid.json", ".gemini.json",
             ".silabas.json")

# Override de fase del downbeat por tema (lo fija el oído de Marcos cuando la
# detección automática falle). Clave = número de tema, valor = fase 0..3.
DOWNBEAT_OVERRIDE = {}

# Palabras átonas del español (no llevan acento prosódico): no compiten por caer
# en tiempo fuerte. Con tilde diacrítica (él, sí, tú, mí, sé) SON tónicas → no van.
ATONAS = {
    "el", "la", "los", "las", "lo", "un", "una", "unos", "unas",
    "a", "ante", "bajo", "con", "contra", "de", "del", "al", "desde", "en",
    "entre", "hacia", "hasta", "para", "por", "según", "sin", "so", "sobre", "tras",
    "y", "e", "o", "u", "ni", "que", "pero", "sino", "mas", "si", "como",
    "cuando", "donde", "mientras", "porque", "pues", "aunque",
    "me", "te", "se", "nos", "os", "le", "les",
    "mi", "tu", "su", "mis", "tus", "sus",
}

VOC = "aeiouáéíóúü"
VOC_FUERTES = "aeoáéó"
VOC_DEBIL_ACENT = "íú"
ACENTOS = "áéíóú"


def _ruta_instrumental(num):
    for f in sorted(os.listdir(DIR_INSTR)):
        if f.lower().endswith(".wav") and f.startswith(f"{num:02d}-"):
            return os.path.join(DIR_INSTR, f)
    return None


def _base_karaoke(num):
    for f in sorted(os.listdir(DIR_KARAOKE)):
        if (f.startswith(f"{num:02d}-") and f.endswith(".json")
                and not f.endswith(DESCARTAR)):
            return f[:-5]
    return None


def _limpiar(palabra):
    """Minúsculas, conserva tildes/ñ, descarta puntuación y dígitos."""
    return "".join(c for c in palabra.lower() if c in VOC or c.isalpha())


def _nucleos(limpia):
    """Lista de (nucleo_str, pos_inicial_en_limpia) resolviendo diptongos/hiatos.
    Cada núcleo = una sílaba."""
    nucleos = []
    i, n = 0, len(limpia)
    while i < n:
        if limpia[i] in VOC:
            j = i
            while j < n and limpia[j] in VOC:
                j += 1
            grupo = limpia[i:j]
            ini_nuc = 0
            for k in range(1, len(grupo)):
                a, b = grupo[k - 1], grupo[k]
                hiato = (a in VOC_FUERTES and b in VOC_FUERTES) \
                    or a in VOC_DEBIL_ACENT or b in VOC_DEBIL_ACENT
                if hiato:
                    nucleos.append((grupo[ini_nuc:k], i + ini_nuc))
                    ini_nuc = k
            nucleos.append((grupo[ini_nuc:], i + ini_nuc))
            i = j
        else:
            i += 1
    return nucleos


def _indice_tonica(limpia):
    """(indice_silaba_tonica, n_silabas, nucleos) por acento escrito o regla."""
    nucleos = _nucleos(limpia)
    n = len(nucleos)
    if n == 0:
        return 0, 1, nucleos
    for idx, (nuc, _pos) in enumerate(nucleos):
        if any(c in ACENTOS for c in nuc):
            return idx, n, nucleos
    if n == 1:
        return 0, 1, nucleos
    ultima = limpia[-1]
    if ultima in "aeiou" or ultima in "ns":
        return n - 2, n, nucleos
    return n - 1, n, nucleos


def _tonica_vocal_pos(palabra):
    """(pos_char_de_la_vocal_tonica_en_limpia | None, idx_tonica, n_silabas, limpia)."""
    limpia = _limpiar(palabra)
    idx, n, nucleos = _indice_tonica(limpia)
    if not nucleos:
        return None, 0, 1, limpia
    return nucleos[idx][1], idx, n, limpia


def _es_atona(palabra):
    return _limpiar(palabra) in ATONAS


def _cargar_items(base):
    """Items por palabra con el tiempo de su tónica. Prefiere el sidecar real."""
    ruta_sil = os.path.join(DIR_KARAOKE, base + ".silabas.json")
    if os.path.exists(ruta_sil):
        datos = json.load(open(ruta_sil, encoding="utf-8"))
        items = [{
            "palabra": d["palabra"], "inicio": float(d["inicio"]),
            "fin": float(d["fin"]), "t_ton": float(d["t_tonica"]),
            "silabas": d.get("silabas", 1), "silaba_tonica": d.get("silaba_tonica", 1),
        } for d in datos]
        return items, "tónica REAL (MMS char-level)"
    datos = json.load(open(os.path.join(DIR_KARAOKE, base + ".json"), encoding="utf-8"))
    items = []
    for p in datos:
        ini, fin = float(p["inicio"]), float(p["fin"])
        _pos, idx, n, _lim = _tonica_vocal_pos(p["palabra"])
        t_ton = ini + (idx / n) * (fin - ini) if n else ini
        items.append({"palabra": p["palabra"], "inicio": ini, "fin": fin,
                      "t_ton": t_ton, "silabas": n, "silaba_tonica": idx + 1})
    return items, "tónica estimada (equiespaciada)"


def _detectar_grilla(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return bpm, beat_times


def _fase_downbeat(y, sr, beat_times):
    """Fase 0..3 cuyo onset GRAVE (bombo) es más fuerte = el '1' del compás."""
    if len(beat_times) < PULSOS_POR_COMPAS:
        return 0
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    graves = librosa.power_to_db(mel[:N_MELS_GRAVES])
    onset_grave = librosa.onset.onset_strength(S=graves, sr=sr)
    env_times = librosa.times_like(onset_grave, sr=sr)
    fuerza = np.interp(beat_times, env_times, onset_grave)
    mejor_fase, mejor_energia = 0, -1e9
    for fase in range(PULSOS_POR_COMPAS):
        idx = [i for i in range(len(beat_times))
               if (i - fase) % PULSOS_POR_COMPAS == 0]
        energia = float(np.median(fuerza[idx])) if idx else -1e9
        if energia > mejor_energia:
            mejor_energia, mejor_fase = energia, fase
    return mejor_fase


def analizar_tema(num):
    ruta_wav = _ruta_instrumental(num)
    base = _base_karaoke(num)
    if not ruta_wav or not base:
        print(f"{num:>2}  (falta instrumental o .json, salto)")
        return None
    items, fuente = _cargar_items(base)
    if not items:
        print(f"{num:>2}  ({base} sin palabras, salto)")
        return None

    t0 = time.time()
    y, sr = librosa.load(ruta_wav, sr=SR, mono=True)
    bpm, beat_times = _detectar_grilla(y, sr)
    if len(beat_times) < 2:
        print(f"{num:>2}  (no se detectaron beats, salto)")
        return None
    periodo = 60.0 / bpm
    tol = periodo * TOL_FRAC
    limite_fuera = periodo * FUERA_GRILLA_FRAC
    fase_grave = _fase_downbeat(y, sr, beat_times)   # detector independiente (bombo)

    # Datos por palabra independientes de la fase del downbeat
    base_pal = []
    for it in items:
        t_ton = it["t_ton"]
        j = int(np.argmin(np.abs(beat_times - t_ton)))
        desfase = float(t_ton - beat_times[j])
        atona = _es_atona(it["palabra"])
        fuera = abs(desfase) > limite_fuera
        base_pal.append((it, j, desfase, atona, fuera))

    # Elegir la fase del downbeat (el '1' del compás)
    if num in DOWNBEAT_OVERRIDE:
        fase, origen_fase = DOWNBEAT_OVERRIDE[num], "override"
    else:
        # Concepto ya validado de forma independiente → para construir la grilla de
        # producción se toma la fase que MAXIMIZA tónicas de contenido en tiempo
        # fuerte (no es circular: ya se validó con el oído en Abzu).
        med_j = [j for (it, j, d, at, fu) in base_pal if not at and not fu]

        def _alineadas(ph):
            return sum(1 for j in med_j if (j - ph) % PULSOS_POR_COMPAS in FUERTES)
        fase = max(range(PULSOS_POR_COMPAS),
                   key=lambda ph: (_alineadas(ph), ph == fase_grave))
        origen_fase = "auto-align"
    discrepa_grave = (fase != fase_grave)

    detalle = []
    for it, j, desfase, atona, fuera in base_pal:
        pos = (j - fase) % PULSOS_POR_COMPAS
        es_fuerte = bool((pos in FUERTES) and (abs(desfase) <= tol) and not fuera)
        detalle.append({
            "palabra": it["palabra"],
            "inicio": round(it["inicio"], 3),
            "fin": round(it["fin"], 3),
            "t_tonica": round(it["t_ton"], 3),
            "silabas": it["silabas"],
            "silaba_tonica": it["silaba_tonica"],
            "atona": atona,
            "beat": j,
            "pos_compas": pos + 1,
            "desfase_ms": round(desfase * 1000, 1),
            "fuera_de_grilla": fuera,
            "en_tiempo_fuerte": es_fuerte,
        })

    medibles = [d for d in detalle if not d["atona"] and not d["fuera_de_grilla"]]
    descartadas_fuera = sum(1 for d in detalle if d["fuera_de_grilla"])
    conteo_pos = [0, 0, 0, 0]
    en_fuerte = 0
    desfases = []
    for d in medibles:
        conteo_pos[d["pos_compas"] - 1] += 1
        if d["en_tiempo_fuerte"]:
            en_fuerte += 1
        desfases.append(abs(d["desfase_ms"]))
    n_med = len(medibles)
    fuerte_pos = conteo_pos[0] + conteo_pos[2]
    total_pos = sum(conteo_pos) or 1
    pct_fuerte_pos = 100.0 * fuerte_pos / total_pos
    pct_en_tol = 100.0 * en_fuerte / n_med if n_med else 0.0
    desfase_medio_ms = float(np.mean(desfases)) if desfases else 0.0
    peores = sorted(medibles, key=lambda d: abs(d["desfase_ms"]), reverse=True)[:5]

    salida = {
        "tema": base,
        "fuente_tonica": fuente,
        "bpm": round(bpm, 1),
        "periodo_beat_s": round(periodo, 3),
        "fase_downbeat": fase,
        "fase_downbeat_grave": fase_grave,
        "origen_fase": origen_fase,
        "discrepa_deteccion_grave": discrepa_grave,
        "compas": "4/4",
        "tolerancia_ms": round(tol * 1000, 1),
        "n_palabras": len(detalle),
        "n_contenido_medibles": n_med,
        "n_descartadas_fuera_grilla": descartadas_fuera,
        "pct_tonicas_en_pulso_fuerte": round(pct_fuerte_pos, 1),
        "pct_en_tiempo_fuerte_estricto": round(pct_en_tol, 1),
        "desfase_medio_ms": round(desfase_medio_ms, 1),
        "distribucion_posicion": {f"t{i+1}": conteo_pos[i] for i in range(4)},
        "beats": [round(float(b), 3) for b in beat_times],
        "palabras": detalle,
    }
    json.dump(salida, open(os.path.join(DIR_KARAOKE, base + ".grid.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)

    etiqueta = "REAL" if "REAL" in fuente else "est."
    marca = " ⚠fase≠grave" if discrepa_grave and origen_fase != "override" else ""
    print(f"{num:>2}  {bpm:>5.1f}bpm  fase={fase}({origen_fase})  "
          f"fuerte={pct_fuerte_pos:>4.0f}%  estricto={pct_en_tol:>4.0f}%  "
          f"desf~{desfase_medio_ms:>4.0f}ms  "
          f"pos[{conteo_pos[0]},{conteo_pos[1]},{conteo_pos[2]},{conteo_pos[3]}]  "
          f"n={n_med:>3}(-{descartadas_fuera})  [{etiqueta}]{marca}  {base}  ({time.time()-t0:.0f}s)")
    return salida, peores


def main():
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 12))
    print(f"[i] Análisis rítmico v3 de los temas {nums}")
    print("    fuerte%  = sílabas tónicas (de contenido) cuyo beat más cercano es FUERTE (t1/t3)")
    print("    estricto% = además dentro de ±1/4 de pulso del beat · [REAL]=tónica del MMS, [est.]=estimada\n")
    todos = []
    for num in nums:
        r = analizar_tema(num)
        if r:
            todos.append(r)

    if todos:
        prom = float(np.mean([s["pct_tonicas_en_pulso_fuerte"] for s, _ in todos]))
        print(f"\n[i] Promedio tónicas en pulso fuerte: {prom:.1f}% (azar = 50%)")
        disc = [s["tema"] for s, _ in todos if s["discrepa_deteccion_grave"]
                and s["origen_fase"] != "override"]
        if disc:
            print(f"[⚠] Fase auto ≠ detector grave en: {', '.join(disc)}")
            print("    → CHEQUEAR DE OÍDO el downbeat; fijar en DOWNBEAT_OVERRIDE si hace falta.")
        print("[i] Peores desfases por tema (tónicas de contenido, para escuchar):")
        for salida, peores in todos:
            ej = ", ".join(
                f"'{d['palabra']}'@{d['t_tonica']}s({d['desfase_ms']:+.0f}ms)"
                for d in peores[:3])
            print(f"    {salida['tema']}: {ej}")
    print(f"\n[OK] Escrito data/07_karaoke/NN-Tema.grid.json por tema.")


if __name__ == "__main__":
    main()
