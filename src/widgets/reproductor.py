"""
Reproductor de audio con posición en tiempo real.

`sounddevice.play()` no expone la posición de reproducción. Acá se usa un
`OutputStream` con un callback que cuenta los frames entregados; la GUI lee
`segundos()` desde un QTimer para mover el playhead. Validado en el spike
`spikes/playhead/spike_playhead.py`.

Soporta play / pausa (reanuda desde la posición) / stop / seek / volumen.
Sin dependencias de Qt: solo numpy + sounddevice.
"""

import numpy as np
import sounddevice as sd

SR = 44100


class Reproductor:
    def __init__(self, sr=SR):
        self.sr = sr
        self.audio = np.zeros(0, dtype="float32")
        self.pos = 0                  # frames entregados (leído por la GUI)
        self.ganancia = 1.0
        self.stream = None

    # ------------------------------------------------------------------ #
    @property
    def activo(self):
        """True si está sonando (no en pausa ni detenido)."""
        return self.stream is not None

    @property
    def duracion(self):
        return len(self.audio) / self.sr if len(self.audio) else 0.0

    def segundos(self):
        return self.pos / self.sr

    # ------------------------------------------------------------------ #
    def cargar(self, audio):
        """Carga un array mono y reinicia la posición."""
        self.detener()
        self.audio = np.ascontiguousarray(
            np.zeros(0, dtype="float32") if audio is None else audio,
            dtype="float32",
        )
        self.pos = 0

    def _callback(self, outdata, frames, time_info, status):
        fin = self.pos + frames
        bloque = self.audio[self.pos:fin]
        n = len(bloque)
        outdata[:n, 0] = bloque * self.ganancia
        if n < frames:                       # último bloque: rellenar y parar
            outdata[n:, 0] = 0.0
            self.pos = len(self.audio)
            raise sd.CallbackStop
        self.pos = fin

    def reproducir(self):
        """Arranca (o reanuda desde la posición actual)."""
        if self.stream is not None or len(self.audio) == 0:
            return
        if self.pos >= len(self.audio):      # si estaba al final, vuelve a empezar
            self.pos = 0
        self.stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            callback=self._callback, finished_callback=self._al_terminar)
        self.stream.start()

    def pausa(self):
        """Detiene el stream conservando la posición (para reanudar).

        `stream.stop()` puede disparar el finished_callback (`_al_terminar`) en
        otro hilo, que anula `self.stream`. Por eso se toma una referencia local
        y se anula el atributo ANTES de cerrar: quien llegue segundo ve None y no
        hace nada (evita el AttributeError por doble cierre / carrera).
        """
        stream = self.stream
        self.stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _al_terminar(self):
        # corre en el hilo de audio al agotarse el audio (CallbackStop)
        stream = self.stream
        self.stream = None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def detener(self):
        """Para y vuelve la posición al inicio."""
        self.pausa()
        self.pos = 0

    def seek(self, segundos):
        """Mueve la posición de reproducción (mantiene el estado play/pausa)."""
        frame = int(max(0.0, segundos) * self.sr)
        self.pos = min(frame, len(self.audio))
