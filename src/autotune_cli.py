"""
Autotune sencillo: graba tu voz, la afina a la escala más cercana y
le añade compresión + reverb para que suene más pulida.

Uso:
    python autotune.py

Requisitos (ver requirements.txt):
    pip install sounddevice soundfile librosa numpy psola pedalboard scipy
"""

import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import psola
from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter

SR = 44100          # frecuencia de muestreo
FMIN = librosa.note_to_hz("C2")   # ~65 Hz, grave
FMAX = librosa.note_to_hz("C6")   # ~1046 Hz, agudo

# Notas (semitonos) que componen cada escala, relativas a la tónica.
ESCALAS = {
    "mayor":   [0, 2, 4, 5, 7, 9, 11],
    "menor":   [0, 2, 3, 5, 7, 8, 10],
    "cromatica": list(range(12)),   # afina a cualquier semitono
}

NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- #
# Grabación
# --------------------------------------------------------------------------- #
def grabar(segundos):
    print(f"\n>>> Grabando {segundos} s... ¡canta ahora!")
    audio = sd.rec(int(segundos * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print(">>> Grabación terminada.")
    return audio.flatten()


def cargar(ruta):
    audio, sr = librosa.load(ruta, sr=SR, mono=True)
    return audio.astype("float32")


# --------------------------------------------------------------------------- #
# Autotune
# --------------------------------------------------------------------------- #
def _grados_permitidos(tonica, escala):
    """Devuelve los semitonos MIDI (0-11) que pertenecen a la escala."""
    base = NOTAS.index(tonica)
    return [(base + g) % 12 for g in ESCALAS[escala]]


def _ajustar_a_escala(f0_midi, grados):
    """Redondea cada nota a la nota más cercana dentro de la escala."""
    salida = np.copy(f0_midi)
    for i, nota in enumerate(f0_midi):
        if np.isnan(nota):
            continue
        # candidatos: misma nota en todas las octavas cercanas
        candidatos = []
        base = int(np.round(nota))
        for octava in range(base - 12, base + 13):
            if octava % 12 in grados:
                candidatos.append(octava)
        if candidatos:
            salida[i] = min(candidatos, key=lambda c: abs(c - nota))
    return salida


def afinar(audio, tonica="C", escala="mayor", fuerza=1.0):
    """
    Detecta el tono nota por nota y lo corrige hacia la escala.

    fuerza: 0.0 = sin cambios, 1.0 = afinación total (clavada).
    """
    f0, voiced, _ = librosa.pyin(
        audio, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=2048
    )

    midi = librosa.hz_to_midi(f0)
    grados = _grados_permitidos(tonica, escala)
    objetivo_midi = _ajustar_a_escala(midi, grados)

    # mezcla entre tono original y tono corregido según 'fuerza'
    corregido = midi + (objetivo_midi - midi) * fuerza
    corregido[~voiced] = np.nan

    objetivo_hz = librosa.midi_to_hz(corregido)
    return psola.vocode(
        audio, sample_rate=SR, target_pitch=objetivo_hz, fmin=FMIN, fmax=FMAX
    )


# --------------------------------------------------------------------------- #
# Pulido final
# --------------------------------------------------------------------------- #
def pulir(audio, reverb=0.25):
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),      # quita retumbe grave
        Compressor(threshold_db=-20, ratio=4, attack_ms=5, release_ms=120),
        Reverb(room_size=0.4, wet_level=reverb, dry_level=1 - reverb * 0.5),
    ])
    procesado = board(audio.reshape(1, -1), SR)
    return procesado.flatten()


def normalizar(audio, pico=0.95):
    m = np.max(np.abs(audio))
    return audio if m == 0 else audio * (pico / m)


# --------------------------------------------------------------------------- #
# Menú interactivo
# --------------------------------------------------------------------------- #
def pedir(texto, defecto):
    r = input(f"{texto} [{defecto}]: ").strip()
    return r if r else defecto


def main():
    print("=" * 50)
    print("  AUTOTUNE en Python  —  grabar y afinar")
    print("=" * 50)

    modo = pedir("¿Grabar (g) o procesar archivo (a)?", "g")

    if modo.lower().startswith("a"):
        ruta = pedir("Ruta del archivo de audio", "voz.wav")
        if not os.path.exists(ruta):
            print(f"No encuentro el archivo: {ruta}")
            sys.exit(1)
        audio = cargar(ruta)
    else:
        seg = int(pedir("¿Cuántos segundos grabar?", "10"))
        audio = grabar(seg)
        sf.write("voz_original.wav", audio, SR)
        print(">>> Guardado: voz_original.wav (tu voz sin procesar)")

    tonica = pedir("Tónica de la canción (C, D, E, F, G, A, B, con # si aplica)", "C")
    escala = pedir("Escala (mayor / menor / cromatica)", "mayor").lower()
    if escala not in ESCALAS:
        escala = "mayor"
    fuerza = float(pedir("Fuerza de afinación (0.0 a 1.0)", "1.0"))
    reverb = float(pedir("Cantidad de reverb (0.0 a 0.6)", "0.25"))

    print("\n>>> Afinando...")
    afinado = afinar(audio, tonica=tonica.upper(), escala=escala, fuerza=fuerza)
    afinado = np.nan_to_num(afinado)

    print(">>> Aplicando compresión + reverb...")
    final = normalizar(pulir(afinado, reverb=reverb))

    salida = "resultado.wav"
    sf.write(salida, final, SR)
    print(f"\n✓ Listo. Resultado guardado en: {salida}")

    if pedir("¿Reproducir el resultado? (s/n)", "s").lower().startswith("s"):
        sd.play(final, SR)
        sd.wait()


if __name__ == "__main__":
    main()
