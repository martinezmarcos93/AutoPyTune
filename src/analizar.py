"""
Análisis técnico de canciones — "ficha" de cada tema para decidir
tónica/escala (autotune) y efectos.

Uso:
    python src/analizar.py "data/00_originales/Abzu.wav"
    python src/analizar.py "data/00_originales"          # toda la carpeta

Da, por cada archivo:
  - duración
  - BPM (tempo)
  - tonalidad estimada (tónica + mayor/menor)  -> úsalo en la GUI
  - rango de notas de la voz (nota grave / aguda) tras aislar la voz
  - loudness (RMS en dB) y rango dinámico
  - brillo espectral (centroide) -> cuánta presencia de agudos

NOTA: la tonalidad y el tempo son estimaciones automáticas; sirven como
punto de partida, conviene confirmarlas de oído.
"""

import os
import sys
import numpy as np
import librosa

NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Perfiles de Krumhansl-Schmuckler para detección de tonalidad.
_MAYOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimar_tonalidad(y, sr):
    """Devuelve (nota, modo, confianza) usando correlación de chroma."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    perfil = chroma.mean(axis=1)
    perfil = perfil / (perfil.sum() + 1e-9)

    mejor = (-1, None, None)
    for desp in range(12):
        may = np.corrcoef(np.roll(_MAYOR, desp), perfil)[0, 1]
        men = np.corrcoef(np.roll(_MENOR, desp), perfil)[0, 1]
        if may > mejor[0]:
            mejor = (may, NOTAS[desp], "Mayor")
        if men > mejor[0]:
            mejor = (men, NOTAS[desp], "Menor")
    conf, nota, modo = mejor
    return nota, modo, conf


def rango_vocal(y, sr):
    """Estima la nota más grave y más aguda cantadas (en la pista dada)."""
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    f0 = f0[voiced & ~np.isnan(f0)]
    if len(f0) == 0:
        return None, None
    return librosa.hz_to_note(f0.min()), librosa.hz_to_note(f0.max())


def ficha(ruta):
    """Analiza un archivo y devuelve el reporte como texto (str)."""
    y, sr = librosa.load(ruta, sr=44100, mono=True)
    dur = len(y) / sr

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    nota, modo, conf = estimar_tonalidad(y, sr)

    rms = librosa.feature.rms(y=y)[0]
    rms_db = 20 * np.log10(np.mean(rms) + 1e-9)
    rango_din = 20 * np.log10((np.percentile(rms, 95) + 1e-9) /
                              (np.percentile(rms, 5) + 1e-9))

    centroide = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    grave, agudo = rango_vocal(y, sr)

    lineas = [
        f"🎵 {os.path.basename(ruta)}",
        f"   Duración          : {int(dur//60)}:{int(dur % 60):02d}",
        f"   Tempo             : {tempo:.0f} BPM",
        f"   Tonalidad (est.)  : {nota} {modo}   (confianza {conf:.2f})",
        f"   → en la GUI       : Tónica={nota}  Escala={modo}",
    ]
    if grave:
        lineas.append(f"   Rango vocal       : {grave}  a  {agudo}")
    lineas += [
        f"   Loudness          : {rms_db:.1f} dB RMS",
        f"   Rango dinámico    : {rango_din:.1f} dB "
        f"({'comprimido' if rango_din < 10 else 'dinámico'})",
        f"   Brillo (centroide): {centroide:.0f} Hz "
        f"({'oscuro' if centroide < 2000 else 'brillante'})",
    ]
    return "\n".join(lineas)


def analizar(ruta):
    print("\n" + ficha(ruta))


def main(args):
    if not args:
        print(__doc__)
        return
    objetivo = args[0]
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    if os.path.isdir(objetivo):
        rutas = [os.path.join(objetivo, f) for f in sorted(os.listdir(objetivo))
                 if f.lower().endswith(exts)]
        if not rutas:
            print(f"No hay audio en {objetivo}")
        for r in rutas:
            try:
                analizar(r)
            except Exception as e:
                print(f"   ⚠ Error con {r}: {e}")
    else:
        analizar(objetivo)


if __name__ == "__main__":
    # La consola de Windows usa cp1252 por defecto y revienta con los emojis
    # del reporte. Forzamos UTF-8 en la salida si el stream lo permite.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    main(sys.argv[1:])
