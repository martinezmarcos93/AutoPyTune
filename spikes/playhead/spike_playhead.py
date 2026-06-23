"""
SPIKE (desechable) — Playhead sincronizado con el audio real.

Objetivo: validar el punto más riesgoso del Incremento A del rediseño DAW
(`docs/designs/DESIGN_20260619_rediseno-daw-gui.md`): saber la posición de
reproducción en tiempo real para mover un playhead que NO se atrase ni adelante
respecto a lo que se escucha.

`sounddevice.play()` NO expone la posición. Acá se usa `sd.OutputStream` con un
callback que cuenta los frames entregados; un `QTimer` en el hilo de la GUI lee
ese contador y redibuja el playhead. Cero dependencias nuevas (numpy, sounddevice
y PyQt6 ya están en el proyecto).

Cómo juzgar el resultado:
    Se genera un click-track (un clic cada 0.5 s). Las marcas verticales grises
    son los clics. La línea roja es el playhead. VEREDICTO:
      - ✅ viable si el playhead cruza cada marca JUSTO cuando suena el clic.
      - ❌ si hay desfase perceptible (se ve adelantado/atrasado) → replantear A.

Ejecutar (con el venv del proyecto):
    .venv\\Scripts\\python spikes\\playhead\\spike_playhead.py
"""

import sys
import numpy as np
import sounddevice as sd

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
)

SR = 44100
DURACION = 12.0           # segundos del click-track
INTERVALO_CLICK = 0.5     # un clic cada medio segundo


def generar_click_track(duracion=DURACION, intervalo=INTERVALO_CLICK, sr=SR):
    """Audio de prueba: silencio con un clic corto cada `intervalo` segundos.

    Devuelve (audio float32 mono, lista de tiempos de clic en segundos).
    """
    n = int(duracion * sr)
    audio = np.zeros(n, dtype="float32")
    tiempos = []
    t = 0.0
    while t < duracion:
        i = int(t * sr)
        # Clic = ráfaga corta (5 ms) de una senoide a 1 kHz con envolvente.
        dur_clic = int(0.005 * sr)
        env = np.linspace(1.0, 0.0, dur_clic, dtype="float32")
        tono = np.sin(2 * np.pi * 1000 * np.arange(dur_clic) / sr).astype("float32")
        audio[i:i + dur_clic] = 0.6 * tono * env
        tiempos.append(t)
        t += intervalo
    return audio, tiempos


class Reproductor:
    """Reproduce un array con OutputStream y expone la posición en frames."""

    def __init__(self, audio, sr=SR):
        self.audio = audio
        self.sr = sr
        self.pos = 0            # frames entregados (leído por la GUI)
        self.stream = None

    @property
    def activo(self):
        return self.stream is not None

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print("status:", status, file=sys.stderr)
        fin = self.pos + frames
        bloque = self.audio[self.pos:fin]
        if len(bloque) < frames:                 # último bloque: rellenar y parar
            outdata[:len(bloque), 0] = bloque
            outdata[len(bloque):, 0] = 0.0
            self.pos = len(self.audio)
            raise sd.CallbackStop
        outdata[:, 0] = bloque
        self.pos = fin

    def reproducir(self):
        if self.stream is not None:
            return
        self.pos = 0
        self.stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            callback=self._callback, finished_callback=self._al_terminar)
        self.stream.start()

    def _al_terminar(self):
        # corre en el hilo de audio cuando se acaba el stream
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def detener(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def segundos(self):
        return self.pos / self.sr


class VistaPlayhead(QWidget):
    """Dibuja las marcas de clic y el playhead. Sin librerías de ploteo."""

    def __init__(self, tiempos, duracion):
        super().__init__()
        self.tiempos = tiempos
        self.duracion = duracion
        self.pos_seg = 0.0
        self.setMinimumHeight(160)

    def set_pos(self, segundos):
        self.pos_seg = segundos
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(25, 25, 30))

        # Marcas de clic (gris).
        p.setPen(QPen(QColor(110, 110, 120), 1))
        for t in self.tiempos:
            x = int(t / self.duracion * w)
            p.drawLine(x, 0, x, h)

        # Playhead (rojo).
        x = int(self.pos_seg / self.duracion * w)
        p.setPen(QPen(QColor(230, 60, 60), 2))
        p.drawLine(x, 0, x, h)


class Ventana(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPIKE — Playhead sincronizado")
        self.resize(900, 260)

        self.audio, tiempos = generar_click_track()
        self.repro = Reproductor(self.audio)
        self.vista = VistaPlayhead(tiempos, DURACION)

        self.btn = QPushButton("▶ Play")
        self.btn.clicked.connect(self._toggle)
        self.lbl = QLabel("0.00 s")
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        fila = QHBoxLayout()
        fila.addWidget(self.btn)
        fila.addWidget(self.lbl)

        layout = QVBoxLayout(self)
        layout.addWidget(self.vista)
        layout.addLayout(fila)

        # ~60 FPS: lee la posición del reproductor y mueve el playhead.
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _toggle(self):
        if self.repro.activo:
            self.repro.detener()
            self.btn.setText("▶ Play")
        else:
            self.repro.reproducir()
            self.btn.setText("⏹ Stop")

    def _tick(self):
        self.vista.set_pos(self.repro.segundos())
        self.lbl.setText(f"{self.repro.segundos():.2f} s")
        if not self.repro.activo and self.btn.text() != "▶ Play":
            self.btn.setText("▶ Play")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = Ventana()
    v.show()
    sys.exit(app.exec())
