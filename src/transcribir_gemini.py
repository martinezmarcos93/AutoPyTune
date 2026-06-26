"""
Etapa 1 (texto exacto cantado) vía LLM multimodal — Gemini que ESCUCHA.

Reemplaza la reconciliación ASR+difflib para obtener la letra EXACTA cantada por
Suno (versos originales + repeticiones reales, en orden). Le da a Gemini la voz IA
aislada + la letra original como referencia de ortografía, y pide la transcripción
exacta en un paso. Diseño: docs/designs/DESIGN_20260626_gemini-letras.md.

Entradas:
  - data/04_voces_extraidas/NN-Tema_voz_ia.wav   (voz IA aislada)
  - data/07_karaoke/LETRAS COMPLETAS ORIGINALES.txt (vía reconciliar_letra)

Salida (en data/07_karaoke/, gitignored):
  - NN-Tema.txt        -> texto exacto cantado (sobreescribe el difflib previo).
  - NN-Tema.difflib.txt / .difflib.json -> respaldo del difflib (solo la 1ra vez).
  - El .json (timestamps) NO se toca: la Etapa 2 (forced alignment) lo regenera.

Uso:
  set GEMINI_API_KEY=...                         (Google AI Studio, free tier)
  .venv\\Scripts\\python src/transcribir_gemini.py 2        # un tema
  .venv\\Scripts\\python src/transcribir_gemini.py 1 2 3    # varios, en orden
"""

import os
import sys
import shutil
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import librosa
import soundfile as sf

try:
    import truststore                 # SSL del antivirus (MITM) -> trust store de Windows
    truststore.inject_into_ssl()
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from reconciliar_letra import versos_originales_por_tema  # noqa: E402

DIR_VOCES = os.path.join(ROOT, "data", "04_voces_extraidas")
DIR_SALIDA = os.path.join(ROOT, "data", "07_karaoke")
SR = 16000
MODELO = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT = """\
Sos un transcriptor experto de música. Te paso el audio de una voz cantada (una sola \
voz, aislada) y la LETRA ORIGINAL del tema como referencia.

Tu tarea: escribir EXACTAMENTE lo que se canta en el audio, en orden, INCLUYENDO todas \
las repeticiones de versos, frases o palabras tal como suenan, y los ad-libs/coritos \
reales (ej. "¡Vamos!", repeticiones tipo "Eternidad, eternidad").

Reglas estrictas:
- La letra original es la fuente de la verdad de las PALABRAS: usala para la ortografía, \
los acentos y los nombres propios correctos.
- NO agregues versos que no se cantan. NO "completes" con el original lo que no suena.
- NO borres repeticiones que SÍ se cantan: si un verso se canta 3 veces, escribilo 3 veces.
- Un verso por línea, en el orden en que se canta.
- Si una parte es ininteligible, escribí la mejor aproximación y marcá esa línea con [?].
- Devolvé SOLO la letra cantada, sin comentarios ni encabezados.

=== LETRA ORIGINAL (referencia de ortografía, SIN repeticiones) ===
{original}
=== FIN REFERENCIA ===
"""


def _clip_16k_norm(ruta):
    """Carga el wav a 16 kHz mono y lo normaliza al 95% del pico."""
    audio, _ = librosa.load(ruta, sr=SR, mono=True)
    pico = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if pico > 0:
        audio = audio * (0.95 / pico)
    return audio


def _ruta_voz(num):
    """Devuelve (ruta, base) de la voz IA del tema; base = nombre sin _voz_ia.wav."""
    for f in os.listdir(DIR_VOCES):
        if f.lower().endswith(".wav") and f.startswith(f"{num:02d}-") and "_voz_ia" in f:
            base = f[: f.lower().rindex("_voz_ia")]
            return os.path.join(DIR_VOCES, f), base
    return None, None


def _respaldar_difflib(ruta_txt, ruta_json):
    """Copia el .txt/.json difflib a .difflib.* la primera vez (no pisa respaldos)."""
    for ruta, suf in ((ruta_txt, ".difflib.txt"), (ruta_json, ".difflib.json")):
        if not os.path.exists(ruta):
            continue
        destino = ruta[: -len(os.path.splitext(ruta)[1])] + suf
        if not os.path.exists(destino):
            shutil.copy2(ruta, destino)
            print(f"    respaldo difflib -> {os.path.basename(destino)}")


def transcribir_tema(num, client, temas):
    """Transcribe un tema con Gemini y escribe data/07_karaoke/NN-Tema.txt."""
    if num not in temas:
        print(f"[X] No hay letra original para el tema {num}, salto.")
        return False
    ruta, base = _ruta_voz(num)
    if not ruta:
        print(f"[X] No encontré la voz IA del tema {num} en {DIR_VOCES}, salto.")
        return False

    print(f"[i] Tema {num}: {base}")
    audio = _clip_16k_norm(ruta)
    dur = len(audio) / SR
    print(f"[i] Duración {dur:.0f}s — clip 16k mono normalizado, subiendo a Gemini...")

    tmp = os.path.join(tempfile.gettempdir(), f"gemini_{num:02d}.wav")
    sf.write(tmp, audio, SR)
    archivo = client.files.upload(file=tmp)

    prompt = PROMPT.format(original="\n".join(temas[num]))
    print("[i] Pidiendo la transcripción (puede tardar)...")
    # thinking_budget=0: sin razonamiento -> evita que Gemini vuelque su cadena
    # de pensamiento dentro del texto (pasó con flash en 08-Fauce de ira).
    from google.genai import types
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    resp = client.models.generate_content(
        model=MODELO, contents=[prompt, archivo], config=config
    )
    texto = (resp.text or "").strip()

    ruta_txt = os.path.join(DIR_SALIDA, base + ".txt")
    ruta_json = os.path.join(DIR_SALIDA, base + ".json")
    _respaldar_difflib(ruta_txt, ruta_json)
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(texto + "\n")

    print("=" * 60)
    print(texto)
    print("=" * 60)
    print(f"[OK] {os.path.basename(ruta_txt)} — {len(texto.splitlines())} líneas "
          f"(original: {len(temas[num])} versos)\n")
    return True


def main():
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not nums:
        print("Uso: python src/transcribir_gemini.py NN [NN ...]")
        return
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("[X] Falta GEMINI_API_KEY. Conseguila en https://aistudio.google.com/apikey")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=key)
    temas = versos_originales_por_tema()
    print(f"[i] Modelo: {MODELO} — temas a procesar: {nums}\n")
    for num in nums:
        transcribir_tema(num, client, temas)


if __name__ == "__main__":
    main()
