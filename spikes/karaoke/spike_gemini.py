"""
SPIKE (desechable) — letra exacta cantada con Gemini (LLM multimodal que escucha).

Valida el punto riesgoso del DESIGN_20260626_gemini-letras.md: ¿Gemini transcribe la
voz IA aislada captando las REPETICIONES reales y usando la letra original como
referencia de ortografía, sin inventar versos no cantados?

Le da a Gemini:
  - el audio de la voz IA aislada (16 kHz mono normalizada),
  - la letra ORIGINAL del tema como referencia (fuente de la verdad de las palabras),
y le pide la letra EXACTA cantada, con repeticiones, en orden.

Requisitos:
  pip install google-genai --trusted-host pypi.org --trusted-host files.pythonhosted.org
  set GEMINI_API_KEY=...            (Google AI Studio, free tier alcanza)

Uso:
  .venv\\Scripts\\python spikes/karaoke/spike_gemini.py 2        # tema 2 (default)
  .venv\\Scripts\\python spikes/karaoke/spike_gemini.py 1 --modelos   # lista modelos
"""

import os
import sys
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from reconciliar_letra import versos_originales_por_tema  # noqa: E402

DIR_VOCES = os.path.join(ROOT, "data", "04_voces_extraidas")
DIR_SALIDA = os.path.join(ROOT, "spikes", "karaoke", "salida")
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
    audio, _ = librosa.load(ruta, sr=SR, mono=True)
    pico = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if pico > 0:
        audio = audio * (0.95 / pico)
    return audio


def _ruta_voz(num):
    for f in os.listdir(DIR_VOCES):
        if f.lower().endswith(".wav") and f.startswith(f"{num:02d}-") and "_voz_ia" in f:
            return os.path.join(DIR_VOCES, f), f
    return None, None


def main():
    args = [a for a in sys.argv[1:]]
    if "--modelos" in args:
        from google import genai
        client = genai.Client(api_key=_api_key())
        print("[i] Modelos disponibles que soportan generateContent:")
        for m in client.models.list():
            acc = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None)
            if not acc or "generateContent" in acc:
                print("   -", m.name)
        return

    num = next((int(a) for a in args if a.isdigit()), 2)
    temas = versos_originales_por_tema()
    if num not in temas:
        print(f"[X] No hay letra original para el tema {num}")
        return
    ruta, nombre = _ruta_voz(num)
    if not ruta:
        print(f"[X] No encontré la voz IA del tema {num} en {DIR_VOCES}")
        return

    print(f"[i] Tema {num}: {nombre}")
    print(f"[i] Modelo: {MODELO}")
    audio = _clip_16k_norm(ruta)
    dur = len(audio) / SR
    print(f"[i] Duración: {dur:.0f}s — preparando clip 16k mono normalizado...")

    tmp = os.path.join(tempfile.gettempdir(), f"spike_gemini_{num:02d}.wav")
    sf.write(tmp, audio, SR)

    from google import genai
    client = genai.Client(api_key=_api_key())

    print("[i] Subiendo audio (Files API)...")
    archivo = client.files.upload(file=tmp)

    original = "\n".join(temas[num])
    prompt = PROMPT.format(original=original)

    print("[i] Pidiendo la transcripción a Gemini (esto puede tardar)...\n")
    resp = client.models.generate_content(model=MODELO, contents=[prompt, archivo])
    texto = (resp.text or "").strip()

    print("=" * 60)
    print(texto)
    print("=" * 60)

    os.makedirs(DIR_SALIDA, exist_ok=True)
    salida = os.path.join(DIR_SALIDA, f"{num:02d}_gemini.txt")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(f"\n[i] Guardado en {salida}")
    print(f"[i] Líneas: {len(texto.splitlines())} | versos del original: {len(temas[num])}")


def _api_key():
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        print("[X] Falta la API key. Conseguila en https://aistudio.google.com/apikey "
              "y exportala:\n    set GEMINI_API_KEY=tu_key_aca")
        sys.exit(1)
    return k


if __name__ == "__main__":
    main()
