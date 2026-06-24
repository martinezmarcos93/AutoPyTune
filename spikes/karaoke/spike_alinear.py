"""
SPIKE (desechable) — Alineación letra↔audio para la sección Karaoke.

Valida el punto más riesgoso del diseño `docs/designs/DESIGN_20260623_karaoke.md`:
¿se puede sacar, de la VOZ IA extraída, una transcripción con timestamps por palabra
que (a) capture las REPETICIONES reales del canto y (b) tenga tiempos usables?

Enfoque: ASR con `faster-whisper` (CTranslate2, CPU). Se transcribe lo que realmente
se canta → las repeticiones aparecen solas. La letra limpia del PDF se pasa como
`initial_prompt` para sesgar la ortografía, NO como objetivo de alineación.

Salidas (en spikes/karaoke/salida/, gitignored):
  - abzu_cantado.txt   → la letra REAL cantada, con repeticiones (pedido de Marcos)
  - abzu_palabras.json → lista de {palabra, inicio, fin} en segundos

Ejecutar:
    .venv\\Scripts\\python spikes\\karaoke\\spike_alinear.py
"""

import os
import sys
import json
import time

import numpy as np
import soundfile as sf
import librosa

# --- truststore: usar el almacén de certificados de Windows (que confía en el CA
#     del antivirus que hace MITM) para que la descarga del modelo desde HuggingFace
#     no falle por SSL. Si no está instalado, se sigue igual (puede fallar la descarga).
try:
    import truststore
    truststore.inject_into_ssl()
    print("[i] truststore activo (usa el trust store de Windows)")
except Exception:
    print("[i] sin truststore; si la descarga del modelo falla por SSL, instalarlo")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOZ = os.path.join(ROOT, "data", "04_voces_extraidas", "01-Abzu_voz_ia.wav")
LETRAS = os.path.join(ROOT, "data", "ZZZ-Letras", "letras.txt")
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida")

SR = 16000          # faster-whisper trabaja a 16 kHz
SEGUNDOS_CLIP = 999  # tema completo (Abzu dura ~210s) para ver las repeticiones
MODELO = "small"     # arrancar chico; si la calidad no alcanza, probar "medium"


def letra_limpia_abzu():
    """Extrae la sección '1-Abzu' del .txt de letras (referencia/initial_prompt)."""
    if not os.path.exists(LETRAS):
        return ""
    with open(LETRAS, encoding="utf-8") as f:
        txt = f.read()
    ini = txt.find("1-Abzu")
    fin = txt.find("2-El descenso")
    bloque = txt[ini:fin] if ini >= 0 else ""
    # Limpiar líneas vacías / numeración suelta de página.
    lineas = [l.strip() for l in bloque.splitlines() if l.strip()]
    lineas = [l for l in lineas if not l.isdigit() and l != "1-Abzu"]
    return " ".join(lineas)


def preparar_clip():
    audio, _ = librosa.load(VOZ, sr=SR, mono=True)
    clip = audio[: SEGUNDOS_CLIP * SR]
    # La voz extraída viene bajita (pico ~0.21): normalizar ayuda al ASR.
    pico = np.max(np.abs(clip))
    if pico > 0:
        clip = clip * (0.95 / pico)
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, "abzu_clip.wav")
    sf.write(ruta, clip, SR)
    return ruta, len(clip) / SR


def main():
    if not os.path.exists(VOZ):
        print("[X] No encuentro la voz extraída:", VOZ)
        sys.exit(1)

    prompt = letra_limpia_abzu()
    print(f"[i] Letra limpia (prompt) {len(prompt)} chars:",
          (prompt[:90] + "...") if prompt else "(vacía)")

    ruta_clip, dur = preparar_clip()
    print(f"[i] Clip: {dur:.1f}s a {SR} Hz  ({ruta_clip})")

    from faster_whisper import WhisperModel
    print(f"[i] Cargando modelo '{MODELO}' (descarga la 1ª vez)...")
    t0 = time.time()
    modelo = WhisperModel(MODELO, device="cpu", compute_type="int8")
    print(f"[i] Modelo listo en {time.time()-t0:.1f}s")

    print("[i] Transcribiendo con timestamps por palabra...")
    t0 = time.time()
    segmentos, info = modelo.transcribe(
        ruta_clip, language="es", word_timestamps=True,
        initial_prompt=prompt or None,
        vad_filter=False,   # el VAD descartaba la voz cantada (la oye como no-habla)
    )

    palabras = []
    lineas_cantadas = []
    for seg in segmentos:            # generador: se consume al iterar
        lineas_cantadas.append(seg.text.strip())
        for w in (seg.words or []):
            palabras.append({"palabra": w.word.strip(),
                             "inicio": round(w.start, 3),
                             "fin": round(w.end, 3)})
    dt = time.time() - t0
    print(f"[i] Transcripción en {dt:.1f}s "
          f"({dur/dt:.1f}x tiempo real)\n")

    # --- Guardar entregables ---
    txt_cantado = "\n".join(lineas_cantadas)
    with open(os.path.join(SALIDA, "abzu_cantado.txt"), "w", encoding="utf-8") as f:
        f.write(txt_cantado + "\n")
    with open(os.path.join(SALIDA, "abzu_palabras.json"), "w", encoding="utf-8") as f:
        json.dump(palabras, f, ensure_ascii=False, indent=2)

    # --- Reporte para juzgar viabilidad ---
    print("=" * 60)
    print(f"LETRA REAL CANTADA ({dur:.0f}s, con repeticiones tal como suenan):")
    print("=" * 60)
    print(txt_cantado)
    print("\n" + "=" * 60)
    print("MUESTRA DE TIMESTAMPS POR PALABRA (primeras 25):")
    print("=" * 60)
    for p in palabras[:25]:
        print(f"  {p['inicio']:6.2f}–{p['fin']:6.2f}s   {p['palabra']}")

    n_lineas_limpia = len([l for l in prompt.split(".") if l.strip()])
    print("\n" + "=" * 60)
    print("MÉTRICAS DE VIABILIDAD")
    print("=" * 60)
    print(f"  palabras con timestamp: {len(palabras)}")
    print(f"  'segmentos' cantados   : {len(lineas_cantadas)}")
    print(f"  idioma detectado       : {info.language} (prob {info.language_probability:.2f})")
    print(f"  velocidad              : {dur/dt:.1f}x tiempo real en CPU")
    print(f"\n  Salidas en: {SALIDA}")


if __name__ == "__main__":
    main()
