"""
Etapa 1 (texto exacto) — FUENTE DE VERDAD definitiva.

Marcos transcribió a mano, escuchando las canciones, el archivo
`LETRAS CON VERSOS REPETIDOS FINAL.txt`: cada tema con sus versos y repeticiones
reales. Es la verdad última; toda diferencia se corrige contra él. Este script lo
parsea y escribe los `data/07_karaoke/NN-Tema.txt` canónicos del karaoke.

Transformaciones (el texto se respeta VERBATIM, incluidas mayúsculas/puntuación):
  - Expande la taquigrafía de repeticiones `... xN` / `... XN` -> N líneas reales
    (el karaoke necesita cada repetición explícita para alinear su tiempo).
  - Quita líneas en blanco (separadores de estrofa) y los encabezados de tema.

Respalda el `.txt` previo (Gemini) en `NN-Tema.gemini.txt` la 1ra vez. NO toca el
`.json` (timestamps): la Etapa 2 (forced alignment) lo regenera desde este texto.

Uso:
  .venv\\Scripts\\python src/compilar_letras.py            # los 11
  .venv\\Scripts\\python src/compilar_letras.py 2 7        # solo algunos
"""

import os
import re
import sys
import shutil

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_SALIDA = os.path.join(ROOT, "data", "07_karaoke")
RUTA_FUENTE = os.path.join(DIR_SALIDA, "LETRAS CON VERSOS REPETIDOS FINAL.txt")

# Encabezado de tema: "1-Abzu", "5- El nacimiento del hombre", "10-Irkalla".
RE_ENCABEZADO = re.compile(r"^\s*(\d+)\s*-\s*(.+?)\s*$")
# Repetición al final de línea: "... x3", "... X2" (con espacio antes de la x).
RE_REPETICION = re.compile(r"^(.*\S)\s+[xX]\s*(\d+)\s*$")
# Líneas que son solo el índice del principio ("1.\tAbzu") -> punto, no guion.
RE_INDICE = re.compile(r"^\s*\d+\.\s")


def parsear_fuente(ruta):
    """Devuelve {numero: [lineas cantadas, repeticiones ya expandidas]}."""
    with open(ruta, encoding="utf-8") as f:
        crudo = f.read().splitlines()

    temas = {}
    num_actual = None
    for linea in crudo:
        if RE_INDICE.match(linea):
            continue
        m = RE_ENCABEZADO.match(linea)
        if m:
            num_actual = int(m.group(1))
            temas[num_actual] = []
            continue
        if num_actual is None:
            continue
        texto = linea.strip()
        if not texto:
            continue
        rep = RE_REPETICION.match(texto)
        if rep:
            base, veces = rep.group(1).strip(), int(rep.group(2))
            temas[num_actual].extend([base] * veces)
        else:
            temas[num_actual].append(texto)
    return temas


def _base_de(num):
    """Nombre base existente del tema en 07_karaoke (ej. '02-El Descenso de Inanna')."""
    prefijo = f"{num:02d}-"
    descartar = (".asr.txt", ".difflib.txt", ".gemini.txt")
    for f in os.listdir(DIR_SALIDA):
        if f.startswith(prefijo) and f.endswith(".txt") and not f.endswith(descartar):
            return f[:-4]
    return None


def compilar(nums=None):
    temas = parsear_fuente(RUTA_FUENTE)
    nums = nums or sorted(temas)
    for num in nums:
        if num not in temas:
            print(f"[X] El tema {num} no está en la fuente, salto.")
            continue
        base = _base_de(num)
        if not base:
            print(f"[X] No encontré el .txt existente del tema {num}, salto.")
            continue
        lineas = temas[num]
        ruta_txt = os.path.join(DIR_SALIDA, base + ".txt")

        respaldo = os.path.join(DIR_SALIDA, base + ".gemini.txt")
        if os.path.exists(ruta_txt) and not os.path.exists(respaldo):
            shutil.copy2(ruta_txt, respaldo)

        previas = 0
        if os.path.exists(respaldo):
            with open(respaldo, encoding="utf-8") as f:
                previas = len([x for x in f.read().splitlines() if x.strip()])

        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")
        print(f"[OK] {base}.txt — {len(lineas)} líneas "
              f"(Gemini tenía {previas})")


if __name__ == "__main__":
    pedidos = [int(a) for a in sys.argv[1:] if a.isdigit()]
    compilar(pedidos or None)
