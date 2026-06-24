"""
Alineación letra↔audio para la sección Karaoke (pipeline OFFLINE).

Transcribe cada voz IA extraída con `faster-whisper` (ASR con timestamps por
palabra) y deja dos salidas por tema en `data/07_karaoke/` (gitignored):

  - NN-Tema.txt  → la letra REAL cantada, con las repeticiones tal como suenan.
  - NN-Tema.json → lista de {palabra, inicio, fin} en segundos.

El ASR transcribe lo que se canta, así que las repeticiones de versos aparecen
solas (las letras "limpias" del PDF no las contemplan). La letra limpia se usa
solo como `initial_prompt` para sesgar la ortografía.

Decisiones validadas en el spike (`spikes/karaoke/spike_alinear.py`):
  - `vad_filter=False` (el VAD descarta la voz CANTADA por oírla como no-habla).
  - Normalizar el clip (la voz extraída viene bajita, pico ~0.2).
  - `truststore` para que la descarga del modelo no falle por el SSL del antivirus.

Uso:
    .venv\\Scripts\\python src/alinear_letra.py            # todos los temas
    .venv\\Scripts\\python src/alinear_letra.py 1 7        # solo temas 1 y 7
"""

import os
import re
import sys
import json

# La consola de Windows usa cp1252 y revienta al imprimir glifos como '✓'/'→'.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import soundfile as sf
import librosa

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_VOCES = os.path.join(ROOT, "data", "04_voces_extraidas")
DIR_LETRAS = os.path.join(ROOT, "data", "ZZZ-Letras")
DIR_SALIDA = os.path.join(ROOT, "data", "07_karaoke")
RUTA_LETRAS_TXT = os.path.join(DIR_LETRAS, "letras.txt")

SR = 16000
MODELO = "medium"          # 'small' es más rápido; 'medium' transcribe mejor
SUFIJO_VOZ = "_voz_ia"


def letras_por_seccion():
    """Devuelve {numero: texto_limpio} parseando data/ZZZ-Letras/letras.txt.

    Las secciones del PDF arrancan con encabezados tipo '1-Abzu', '2-El
    descenso de Inanna', etc. Se corta por esos marcadores.
    """
    if not os.path.exists(RUTA_LETRAS_TXT):
        return {}
    with open(RUTA_LETRAS_TXT, encoding="utf-8") as f:
        txt = f.read()
    # Posiciones de los encabezados 'N-...' al inicio de línea.
    marcas = list(re.finditer(r"(?m)^\s*(\d{1,2})-[^\n]+", txt))
    secciones = {}
    for i, m in enumerate(marcas):
        num = int(m.group(1))
        ini = m.end()
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(txt)
        cuerpo = txt[ini:fin]
        lineas = [l.strip() for l in cuerpo.splitlines() if l.strip()]
        lineas = [l for l in lineas if not l.isdigit()]   # nº de página suelto
        secciones[num] = " ".join(lineas)
    return secciones


def _clip_normalizado(ruta_voz):
    """Carga la voz a 16 kHz mono y la normaliza (la extraída viene bajita)."""
    audio, _ = librosa.load(ruta_voz, sr=SR, mono=True)
    pico = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if pico > 0:
        audio = audio * (0.95 / pico)
    return audio


def transcribir(modelo, audio, prompt=None, condicionar=True):
    """Transcribe un array (16 kHz) y devuelve (lineas, palabras).

    `condicionar` = condition_on_previous_text. En False evita que un segmento
    mal transcripto envenene a los siguientes (loops/colapso de Whisper).
    """
    segmentos, _ = modelo.transcribe(
        audio, language="es", word_timestamps=True,
        initial_prompt=prompt or None, vad_filter=False,
        condition_on_previous_text=condicionar,
    )
    lineas, palabras = [], []
    for seg in segmentos:
        lineas.append(seg.text.strip())
        for w in (seg.words or []):
            palabras.append({"palabra": w.word.strip(),
                             "inicio": round(w.start, 3),
                             "fin": round(w.end, 3)})
    return lineas, palabras


def alinear_tema(modelo, ruta_voz, prompt=None):
    """Procesa una voz y escribe NN-Tema.txt + NN-Tema.json. Devuelve la base."""
    base = os.path.basename(ruta_voz)
    base = base[: -len(".wav")] if base.lower().endswith(".wav") else base
    if base.endswith(SUFIJO_VOZ):
        base = base[: -len(SUFIJO_VOZ)]

    audio = _clip_normalizado(ruta_voz)
    lineas, palabras = transcribir(modelo, audio, prompt)

    # Red de seguridad: si salió degenerado (Whisper colapsa y devuelve basura;
    # ej.: muy pocas palabras pese a haber voz), reintentar sin prompt y sin
    # condicionar en el texto previo.
    dur = len(audio) / SR
    if dur > 30 and len(palabras) < dur / 10:
        print(f"    [!] '{base}' salió degenerado ({len(palabras)} palabras "
              f"en {dur:.0f}s); reintentando sin prompt y sin condicionar...")
        lineas, palabras = transcribir(modelo, audio, prompt=None,
                                       condicionar=False)

    os.makedirs(DIR_SALIDA, exist_ok=True)
    with open(os.path.join(DIR_SALIDA, base + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    with open(os.path.join(DIR_SALIDA, base + ".json"), "w", encoding="utf-8") as f:
        json.dump(palabras, f, ensure_ascii=False, indent=2)

    return base, len(palabras), len(lineas)


def _numero_de(nombre):
    m = re.match(r"\s*(\d{1,2})-", nombre)
    return int(m.group(1)) if m else None


def procesar_todos(numeros=None):
    """Transcribe todas las voces (o solo los números pedidos)."""
    if not os.path.isdir(DIR_VOCES):
        print("[X] No existe", DIR_VOCES)
        return
    voces = sorted(f for f in os.listdir(DIR_VOCES)
                   if f.lower().endswith(".wav") and SUFIJO_VOZ in f)
    secciones = letras_por_seccion()

    from faster_whisper import WhisperModel
    print(f"[i] Cargando modelo '{MODELO}' (descarga la 1ª vez)...")
    modelo = WhisperModel(MODELO, device="cpu", compute_type="int8")

    for nombre in voces:
        num = _numero_de(nombre)
        if numeros and num not in numeros:
            continue
        ruta = os.path.join(DIR_VOCES, nombre)
        prompt = secciones.get(num)
        print(f"[i] Transcribiendo {nombre} (tema {num})...")
        base, n_pal, n_lin = alinear_tema(modelo, ruta, prompt)
        print(f"    ✓ {base}: {n_pal} palabras, {n_lin} segmentos "
              f"-> data/07_karaoke/{base}.txt + .json")

    print("\n[i] Listo. Salidas en", DIR_SALIDA)


if __name__ == "__main__":
    pedidos = {int(a) for a in sys.argv[1:] if a.isdigit()} or None
    procesar_todos(pedidos)
