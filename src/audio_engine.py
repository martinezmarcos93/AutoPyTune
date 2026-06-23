"""
Motor de audio: grabación, afinación (autotune) y pulido.
Sin dependencias de GUI — la interfaz lo usa desde gui.py.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import psola
from pedalboard import (
    Pedalboard, Compressor, Reverb, HighpassFilter,
    PeakFilter, HighShelfFilter,
)

SR = 44100
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C6")

ESCALAS = {
    "Mayor":     [0, 2, 4, 5, 7, 9, 11],
    "Menor":     [0, 2, 3, 5, 7, 8, 10],
    "Cromática": list(range(12)),
}

NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- #
# Grabación / carga
# --------------------------------------------------------------------------- #
def grabar(segundos, sr=SR):
    audio = sd.rec(int(segundos * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def cargar(ruta, sr=SR):
    audio, _ = librosa.load(ruta, sr=sr, mono=True)
    return audio.astype("float32")


def guardar(ruta, audio, sr=SR):
    sf.write(ruta, audio, sr)


def reproducir(audio, sr=SR):
    sd.play(audio, sr)


def detener_reproduccion():
    sd.stop()


# --------------------------------------------------------------------------- #
# Autotune
# --------------------------------------------------------------------------- #
def _grados_permitidos(tonica, escala):
    base = NOTAS.index(tonica)
    return [(base + g) % 12 for g in ESCALAS[escala]]


def _ajustar_a_escala(f0_midi, grados):
    salida = np.copy(f0_midi)
    for i, nota in enumerate(f0_midi):
        if np.isnan(nota):
            continue
        base = int(np.round(nota))
        candidatos = [o for o in range(base - 12, base + 13) if o % 12 in grados]
        if candidatos:
            salida[i] = min(candidatos, key=lambda c: abs(c - nota))
    return salida


def _suavizar_transiciones(corregido, suavizado):
    """
    Suaviza la curva de tono corregida para que las transiciones entre
    notas sean graduales (canto natural) en vez de saltos bruscos (robot).

    suavizado: 0.0 = sin suavizar (efecto robótico tipo T-Pain),
               1.0 = transiciones muy graduales (natural).
    Se aplica un filtro de un polo (EMA) que se reinicia en los silencios.
    """
    if suavizado <= 0:
        return corregido
    # alpha cercano a 1 => glissando lento; mapeo perceptual suave.
    alpha = 0.45 + 0.5 * float(np.clip(suavizado, 0.0, 1.0))
    salida = np.copy(corregido)
    ema = np.nan
    for i, v in enumerate(corregido):
        if np.isnan(v):
            ema = np.nan          # reinicia tras un silencio
            continue
        ema = v if np.isnan(ema) else alpha * ema + (1 - alpha) * v
        salida[i] = ema
    return salida


def afinar(audio, tonica="C", escala="Mayor", fuerza=1.0, suavizado=0.0,
           sr=SR, progreso=None):
    if progreso:
        progreso("Detectando tono...", 20)
    f0, voiced, _ = librosa.pyin(audio, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=2048)

    if progreso:
        progreso("Calculando notas...", 45)
    midi = librosa.hz_to_midi(f0)
    grados = _grados_permitidos(tonica, escala)
    objetivo_midi = _ajustar_a_escala(midi, grados)
    corregido = midi + (objetivo_midi - midi) * fuerza
    corregido[~voiced] = np.nan
    corregido = _suavizar_transiciones(corregido, suavizado)

    if progreso:
        progreso("Aplicando afinación...", 70)
    objetivo_hz = librosa.midi_to_hz(corregido)
    return psola.vocode(audio, sample_rate=sr, target_pitch=objetivo_hz,
                        fmin=FMIN, fmax=FMAX)


# --------------------------------------------------------------------------- #
# Pulido
# --------------------------------------------------------------------------- #
def pulir(audio, reverb=0.25, brillo=3.0, sr=SR,
          highpass_hz=80.0, peak_gain_db=-2.0,
          comp_umbral_db=-20.0, comp_ratio=4.0):
    """
    Cadena de mezcla de voz (rack de efectos, Incremento C):
      - paso alto: quita retumbe grave            (highpass_hz)
      - peak @ 300 Hz: saca el sonido "encajonado" de cuartos  (peak_gain_db)
      - compresor: nivela el volumen              (comp_umbral_db, comp_ratio)
      - high shelf @ 10 kHz: 'aire'/brillo de estudio          (brillo)
      - reverb: espacio                           (reverb)

    Los defaults son los valores que estaban hardcodeados antes de C: con los
    valores por omisión la salida es idéntica a la anterior (retrocompatible).
    Quedan fijos (alcance curado): peak freq/Q, comp attack/release, shelf freq,
    reverb room_size.
    """
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_hz),
        PeakFilter(cutoff_frequency_hz=300, gain_db=peak_gain_db, q=1.0),
        Compressor(threshold_db=comp_umbral_db, ratio=comp_ratio,
                   attack_ms=5, release_ms=120),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=brillo),
        Reverb(room_size=0.4, wet_level=reverb, dry_level=1 - reverb * 0.5),
    ])
    return board(audio.reshape(1, -1), sr).flatten()


def normalizar(audio, pico=0.95):
    m = np.max(np.abs(audio))
    return audio if m == 0 else audio * (pico / m)


def procesar(audio, tonica, escala, fuerza, reverb, suavizado=0.0, brillo=3.0,
             sr=SR, progreso=None,
             highpass_hz=80.0, peak_gain_db=-2.0,
             comp_umbral_db=-20.0, comp_ratio=4.0):
    """Pipeline completo: afinar + pulir + normalizar."""
    afinado = np.nan_to_num(
        afinar(audio, tonica, escala, fuerza, suavizado, sr, progreso))
    if progreso:
        progreso("Puliendo (EQ + compresión + reverb)...", 90)
    final = normalizar(pulir(
        afinado, reverb, brillo, sr,
        highpass_hz=highpass_hz, peak_gain_db=peak_gain_db,
        comp_umbral_db=comp_umbral_db, comp_ratio=comp_ratio))
    if progreso:
        progreso("Listo.", 100)
    return final


# --------------------------------------------------------------------------- #
# Separación de fuentes (voz / instrumental) con Demucs
# --------------------------------------------------------------------------- #
def separar_pistas(ruta_audio, sr=SR, modelo="htdemucs", device=None, progreso=None):
    """
    Separa un archivo en (voz, instrumental), ambos arrays mono float32.

    Demucs trabaja en estéreo y necesita normalización mean/std; aquí se
    prepara la señal correctamente y luego se devuelve la voz por separado
    y el resto (bajo + batería + otros) sumado como instrumental.
    """
    # Imports perezosos: demucs/torch son pesados, solo se cargan al usarlos.
    import torch
    from demucs import pretrained
    from demucs.apply import apply_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if progreso:
        progreso(f"Cargando modelo {modelo} ({device})...", 8)
    model = pretrained.get_model(modelo)
    model.to(device).eval()
    model_sr = model.samplerate          # htdemucs -> 44100
    canales = model.audio_channels       # htdemucs -> 2 (estéreo)

    # Cargar audio en estéreo, a la frecuencia del modelo.
    audio, _ = librosa.load(ruta_audio, sr=model_sr, mono=False)
    if audio.ndim == 1:                  # mono -> duplicar a estéreo
        audio = np.stack([audio, audio])
    if audio.shape[0] == 1:
        audio = np.repeat(audio, canales, axis=0)
    audio = audio[:canales]              # recortar si trae más canales

    wav = torch.from_numpy(audio).float()
    # Normalización que espera Demucs.
    ref = wav.mean(0)
    desv = ref.std() + 1e-8
    wav = (wav - ref.mean()) / desv

    if progreso:
        progreso("Separando pistas (esto tarda)...", 15)
    with torch.no_grad():
        sources = apply_model(
            model, wav[None], device=device, shifts=1, split=True, overlap=0.25
        )[0]
    sources = sources * desv + ref.mean()   # deshacer normalización

    nombres = list(model.sources)            # ['drums','bass','other','vocals']
    idx_voz = nombres.index("vocals")
    voz = sources[idx_voz].cpu().numpy()
    instrumental = sum(
        sources[i].cpu().numpy() for i in range(len(nombres)) if i != idx_voz
    )

    # A mono para el resto del pipeline.
    if voz.ndim > 1:
        voz = voz.mean(axis=0)
    if instrumental.ndim > 1:
        instrumental = instrumental.mean(axis=0)

    # Remuestrear a nuestra SR si el modelo usa otra.
    if model_sr != sr:
        voz = librosa.resample(voz, orig_sr=model_sr, target_sr=sr)
        instrumental = librosa.resample(instrumental, orig_sr=model_sr, target_sr=sr)

    return voz.astype(np.float32), instrumental.astype(np.float32)


# --------------------------------------------------------------------------- #
# Alineación temporal (estirar la voz para que dure lo mismo)
# --------------------------------------------------------------------------- #
def ajustar_duracion(audio, duracion_objetivo, sr=SR):
    """
    Estira/comprime el audio para que dure 'duracion_objetivo' segundos,
    preservando el tono. Usa Rubber Band si está disponible (mejor calidad),
    si no, usa el phase-vocoder de librosa (solo pip, sin binarios extra).

    Convención de 'rate': rate > 1 acorta. Para llegar al objetivo,
    rate = duracion_actual / duracion_objetivo.
    """
    duracion_actual = len(audio) / sr
    if duracion_actual == 0 or duracion_objetivo <= 0:
        return audio
    rate = duracion_actual / duracion_objetivo
    if abs(rate - 1.0) < 1e-3:
        return audio
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(audio, sr, rate).astype(np.float32)
    except Exception:
        # Fallback sin dependencias externas.
        return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


# --------------------------------------------------------------------------- #
# Mezcla final (voz + instrumental)
# --------------------------------------------------------------------------- #
def mezclar(voz, instrumental, ganancia_voz=1.0, ganancia_inst=0.9):
    """Suma voz e instrumental igualando longitudes y evitando clipping."""
    n = max(len(voz), len(instrumental))
    voz = np.pad(voz, (0, n - len(voz)))
    instrumental = np.pad(instrumental, (0, n - len(instrumental)))
    mezcla = voz * ganancia_voz + instrumental * ganancia_inst
    pico = np.max(np.abs(mezcla))
    if pico > 0.95:
        mezcla = mezcla * (0.95 / pico)
    return mezcla.astype(np.float32)


# --------------------------------------------------------------------------- #
# Pipeline completo: reemplazar la voz IA de Suno por la tuya
# --------------------------------------------------------------------------- #
def reemplazar_voz(ruta_suno, ruta_mi_voz, tonica, escala, fuerza, reverb,
                   suavizado=0.0, brillo=3.0,
                   ganancia_voz=1.1, ganancia_inst=0.9, alinear=True,
                   sr=SR, progreso=None,
                   highpass_hz=80.0, peak_gain_db=-2.0,
                   comp_umbral_db=-20.0, comp_ratio=4.0):
    """
    1. Separa voz IA + instrumental del archivo de Suno.
    2. Carga tu voz y (opcional) la alinea a la duración de la voz original.
    3. Afina y pule tu voz.
    4. La mezcla con el instrumental.
    Devuelve el array de la mezcla final.
    """
    voz_original, instrumental = separar_pistas(ruta_suno, sr=sr, progreso=progreso)

    if progreso:
        progreso("Cargando tu voz...", 55)
    mi_voz = cargar(ruta_mi_voz, sr=sr)

    if alinear:
        objetivo = len(voz_original) / sr
        if progreso:
            progreso(f"Alineando a {objetivo:.1f}s...", 62)
        mi_voz = ajustar_duracion(mi_voz, objetivo, sr=sr)

    if progreso:
        progreso("Afinando tu voz...", 72)
    voz_afinada = np.nan_to_num(
        afinar(mi_voz, tonica, escala, fuerza, suavizado, sr=sr))

    if progreso:
        progreso("Puliendo voz...", 86)
    voz_pulida = normalizar(pulir(
        voz_afinada, reverb, brillo, sr=sr,
        highpass_hz=highpass_hz, peak_gain_db=peak_gain_db,
        comp_umbral_db=comp_umbral_db, comp_ratio=comp_ratio))

    if progreso:
        progreso("Mezclando con el instrumental...", 95)
    resultado = mezclar(voz_pulida, instrumental, ganancia_voz, ganancia_inst)

    if progreso:
        progreso("¡Mezcla lista!", 100)
    return resultado
