"""
Reconciliación de letras para la sección Karaoke (pipeline OFFLINE).

Cruza la transcripción ASR (con repeticiones reales del canto, pero con errores)
contra la letra ORIGINAL correcta del PDF para producir, por tema, la letra final:
texto correcto en la secuencia cantada (con repeticiones), conservando los
timestamps por palabra.

Entradas (en data/07_karaoke/, gitignored):
  - NN-Tema.json / .txt          -> transcripción ASR (la genera src/alinear_letra.py).
  - LETRAS COMPLETAS ORIGINALES.txt -> letra original correcta de los 11 temas.

Salidas (sobreescribe, con respaldo del crudo):
  - NN-Tema.txt   -> letra original en la secuencia cantada (con repeticiones).
  - NN-Tema.json  -> palabras corregidas + (inicio, fin) interpolados.
  - NN-Tema.asr.{txt,json} -> respaldo del ASR crudo (se crea una sola vez).

Método A (ver docs/designs/DESIGN_20260625_reconciliacion-letras.md):
  cada segmento ASR se mapea al mejor tramo del original (ventana deslizante +
  SequenceMatcher sobre palabras normalizadas). Lo que no matchea es ad-lib real
  (se conserva) o basura (se marca [REVISAR] para el oido de Marcos).

Uso:
    .venv\\Scripts\\python src/reconciliar_letra.py            # todos los temas
    .venv\\Scripts\\python src/reconciliar_letra.py 2 1        # solo temas 2 y 1
"""

import os
import re
import sys
import json
import shutil
import unicodedata
from difflib import SequenceMatcher

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_SALIDA = os.path.join(ROOT, "data", "07_karaoke")
RUTA_ORIGINALES = os.path.join(DIR_SALIDA, "LETRAS COMPLETAS ORIGINALES.txt")

# Umbral de similitud para aceptar que un segmento ASR es un verso del original.
UMBRAL_MATCH = 0.55
# Por debajo de esto, un segmento sin match se trata como BASURA (no ad-lib).
UMBRAL_BASURA = 0.30


def _sin_acentos(texto):
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar(palabra):
    """Minuscula, sin acentos, solo alfanumerico (para comparar, no para mostrar)."""
    p = _sin_acentos(palabra.lower())
    return re.sub(r"[^a-z0-9ñ]", "", p)


def _tokens_norm(texto):
    return [t for t in (normalizar(w) for w in texto.split()) if t]


def versos_originales_por_tema():
    """Devuelve {numero: [verso, ...]} parseando LETRAS COMPLETAS ORIGINALES.txt.

    Las secciones arrancan con encabezados 'N-Titulo' al inicio de linea. La lista
    indice del principio ('1.  Abzu') usa punto, no guion, asi que no matchea.
    """
    if not os.path.exists(RUTA_ORIGINALES):
        return {}
    with open(RUTA_ORIGINALES, encoding="utf-8") as f:
        txt = f.read()
    marcas = list(re.finditer(r"(?m)^\s*(\d{1,2})-[^\n]*", txt))
    temas = {}
    for i, m in enumerate(marcas):
        num = int(m.group(1))
        ini = m.end()
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(txt)
        cuerpo = txt[ini:fin]
        versos = [l.strip() for l in cuerpo.splitlines() if l.strip()]
        temas[num] = versos
    return temas


def _segmentos_asr(ruta_txt, palabras_json):
    """Reconstruye los segmentos del ASR (lineas del txt) asociando a cada uno su
    rango temporal, consumiendo en orden las palabras del json.

    Devuelve [{texto, palabras:[{palabra,inicio,fin}], inicio, fin}, ...].
    """
    with open(ruta_txt, encoding="utf-8") as f:
        lineas = [l.strip() for l in f.read().splitlines() if l.strip()]

    segmentos = []
    idx = 0
    n = len(palabras_json)
    for linea in lineas:
        n_tok = len(linea.split())
        bloque = palabras_json[idx: idx + n_tok]
        idx += n_tok
        if bloque:
            inicio = bloque[0]["inicio"]
            fin = bloque[-1]["fin"]
        else:
            inicio = fin = None
        segmentos.append({"texto": linea, "palabras": bloque,
                          "inicio": inicio, "fin": fin})
    return segmentos


def _mejor_tramo(seg_tokens, orig_tokens):
    """Mejor ventana contigua de orig_tokens que matchea seg_tokens.

    Devuelve (ratio, ini, fin) sobre indices de orig_tokens. Permite reusar tramos
    (cada segmento se evalua contra todo el original) -> las repeticiones salen solas.
    """
    L = len(seg_tokens)
    if L == 0 or not orig_tokens:
        return 0.0, 0, 0
    mejor = (0.0, 0, 0)
    sm = SequenceMatcher()
    sm.set_seq2(seg_tokens)
    # Probar ventanas de tamaño 0.6L .. 1.6L en cada posicion del original.
    tam_min = max(1, int(L * 0.6))
    tam_max = min(len(orig_tokens), int(L * 1.6) + 1)
    for tam in range(tam_min, tam_max + 1):
        for ini in range(0, len(orig_tokens) - tam + 1):
            sm.set_seq1(orig_tokens[ini: ini + tam])
            r = sm.ratio()
            if r > mejor[0]:
                mejor = (r, ini, ini + tam)
    return mejor


def _interpolar_tiempos(palabras_destino, inicio, fin):
    """Reparte [inicio, fin] uniformemente entre las palabras_destino."""
    k = len(palabras_destino)
    if k == 0 or inicio is None or fin is None:
        return [{"palabra": p, "inicio": inicio, "fin": fin} for p in palabras_destino]
    dur = max(0.0, fin - inicio)
    paso = dur / k
    salida = []
    for i, p in enumerate(palabras_destino):
        t0 = round(inicio + paso * i, 3)
        t1 = round(inicio + paso * (i + 1), 3)
        salida.append({"palabra": p, "inicio": t0, "fin": t1})
    return salida


def _es_basura(texto):
    """Heuristica: el segmento es ruido/alucinacion del ASR, no letra real.

    Detecta: solo simbolos (sin palabras), la alucinacion 'Amara.org', y texto
    dominado por scripts NO latinos (cirilico, CJK, griego) que Whisper inventa
    cuando no entiende la voz cantada/gritada.
    """
    bajo = texto.lower()
    if "amara.org" in bajo or ("subtitul" in bajo and "amara" in bajo):
        return True
    toks = _tokens_norm(texto)
    if not toks:                      # solo simbolos (†, ❦, puntos, ...)
        return True
    # _sin_acentos descompone las tildes del español -> cualquier letra con
    # ord > 127 que quede es de otro alfabeto (cirilico/griego/CJK).
    plano = _sin_acentos(bajo)
    letras = [c for c in plano if c.isalpha()]
    if letras:
        no_latinas = sum(1 for c in letras if ord(c) > 127)
        if no_latinas / len(letras) > 0.15:
            return True
    return False


def reconciliar_tema(num, ruta_txt, ruta_json, versos):
    """Reconcilia un tema. Devuelve (segmentos_final, reporte)."""
    with open(ruta_json, encoding="utf-8") as f:
        palabras = json.load(f)
    segmentos = _segmentos_asr(ruta_txt, palabras)

    # Tokens normalizados del original (planos) + de donde viene cada token (verso).
    orig_tokens = []
    for v in versos:
        orig_tokens.extend(_tokens_norm(v))
    orig_palabras = []           # palabras originales "para mostrar"
    for v in versos:
        orig_palabras.extend(v.split())

    lineas_out = []
    palabras_out = []
    cubiertos = set()            # indices de orig_tokens que aparecieron (cobertura)
    flags = []                   # (n_linea, motivo, texto)

    for i, seg in enumerate(segmentos):
        seg_tokens = _tokens_norm(seg["texto"])
        ratio, ini, fin = _mejor_tramo(seg_tokens, orig_tokens)

        if ratio >= UMBRAL_MATCH:
            tramo = orig_palabras[ini:fin]
            texto_final = " ".join(tramo)
            lineas_out.append(texto_final)
            palabras_out.extend(_interpolar_tiempos(tramo, seg["inicio"], seg["fin"]))
            for j in range(ini, fin):
                cubiertos.add(j)
        elif _es_basura(seg["texto"]) and ratio < UMBRAL_BASURA:
            # Basura/alucinacion: se descarta del texto final, se reporta.
            flags.append((i + 1, f"BASURA descartada (sim {ratio:.2f})", seg["texto"]))
        else:
            # Ad-lib real o duda: se conserva tal cual y se marca para el oido de Marcos.
            lineas_out.append(f"{seg['texto']}    [REVISAR]")
            palabras_out.extend(seg["palabras"])
            flags.append((i + 1, f"AD-LIB/DUDA conservado (sim {ratio:.2f})", seg["texto"]))

    total_orig = len(orig_tokens)
    cobertura = (len(cubiertos) / total_orig) if total_orig else 0.0
    reporte = {
        "tema": num,
        "segmentos": len(segmentos),
        "cobertura": cobertura,
        "tokens_cubiertos": len(cubiertos),
        "tokens_original": total_orig,
        "flags": flags,
    }
    return lineas_out, palabras_out, reporte


def _respaldar_crudo(ruta_txt, ruta_json):
    """Copia el ASR crudo a .asr.* una sola vez (reversibilidad; data/ es gitignored)."""
    base_txt = ruta_txt[:-4] + ".asr.txt"
    base_json = ruta_json[:-5] + ".asr.json"
    if not os.path.exists(base_txt):
        shutil.copy2(ruta_txt, base_txt)
    if not os.path.exists(base_json):
        shutil.copy2(ruta_json, base_json)
    return base_txt, base_json


def _numero_de(nombre):
    m = re.match(r"\s*(\d{1,2})-", nombre)
    return int(m.group(1)) if m else None


def procesar_todos(numeros=None, sobreescribir=True):
    temas = versos_originales_por_tema()
    if not temas:
        print("[X] No pude parsear", RUTA_ORIGINALES)
        return

    # Temas con json de ASR (excluyendo respaldos .asr.json).
    jsons = sorted(f for f in os.listdir(DIR_SALIDA)
                   if f.lower().endswith(".json") and not f.endswith(".asr.json"))

    for nombre_json in jsons:
        base = nombre_json[:-5]
        num = _numero_de(base)
        if num is None or (numeros and num not in numeros):
            continue
        ruta_json = os.path.join(DIR_SALIDA, nombre_json)
        ruta_txt = os.path.join(DIR_SALIDA, base + ".txt")
        if not os.path.exists(ruta_txt):
            print(f"[!] Falta {base}.txt, salto.")
            continue
        if num not in temas:
            print(f"[!] No hay original para el tema {num}, salto.")
            continue

        # Respaldar el ASR crudo, y RECONCILIAR DESDE el crudo (re-runs idempotentes).
        asr_txt, asr_json = _respaldar_crudo(ruta_txt, ruta_json)
        lineas, palabras, rep = reconciliar_tema(num, asr_txt, asr_json, temas[num])

        print(f"\n[i] Tema {num} ({base}): {rep['segmentos']} segmentos, "
              f"cobertura {rep['cobertura']*100:.0f}% "
              f"({rep['tokens_cubiertos']}/{rep['tokens_original']} palabras del original)")
        for n_lin, motivo, texto in rep["flags"]:
            recorte = texto if len(texto) <= 60 else texto[:57] + "..."
            print(f"    L{n_lin:>2} {motivo}: {recorte}")

        if sobreescribir:
            with open(ruta_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(lineas) + "\n")
            with open(ruta_json, "w", encoding="utf-8") as f:
                json.dump(palabras, f, ensure_ascii=False, indent=2)
            print(f"    -> reconciliado: {base}.txt + .json "
                  f"(crudo respaldado en {os.path.basename(asr_txt)})")

    print("\n[i] Listo. Revisa los [REVISAR] y la cobertura de 01/06/07 (metodo A).")


if __name__ == "__main__":
    pedidos = {int(a) for a in sys.argv[1:] if a.isdigit()} or None
    procesar_todos(pedidos)
