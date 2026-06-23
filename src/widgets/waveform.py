"""
VistaOnda — forma de onda dibujada con QPainter (sin dependencias de ploteo).

Dibuja la envolvente (min/max por columna de píxel) del audio y un playhead que
avanza durante la reproducción. Un clic sobre la onda emite `seek_solicitado`
con el tiempo en segundos, para reposicionar la reproducción.

La envolvente se cachea por ancho: solo se recalcula si cambia el audio o el
ancho del widget, no en cada repintado del playhead.
"""

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget

SR = 44100

_FONDO = QColor(21, 23, 28)
_ONDA = QColor(108, 92, 231)        # violeta, igual que la paleta del proyecto
_EJE = QColor(52, 57, 69)
_PLAYHEAD = QColor(230, 60, 60)


class VistaOnda(QWidget):
    seek_solicitado = pyqtSignal(float)   # segundos

    def __init__(self, sr=SR):
        super().__init__()
        self.sr = sr
        self.audio = np.zeros(0, dtype="float32")
        self.pos_seg = 0.0
        self._picos = None            # (mins, maxs) cacheados
        self._cache_w = -1
        self.setMinimumHeight(120)

    # ------------------------------------------------------------------ #
    def set_audio(self, audio):
        self.audio = (np.zeros(0, dtype="float32") if audio is None
                      else np.asarray(audio, dtype="float32"))
        self.pos_seg = 0.0
        self._invalidar_cache()
        self.update()

    def set_pos(self, segundos):
        self.pos_seg = segundos
        self.update()

    @property
    def duracion(self):
        return len(self.audio) / self.sr if len(self.audio) else 0.0

    # ------------------------------------------------------------------ #
    def _invalidar_cache(self):
        self._picos = None
        self._cache_w = -1

    def _calcular_picos(self, w):
        """Envolvente min/max en `w` columnas, vectorizada con reduceat."""
        n = len(self.audio)
        if n == 0 or w <= 0:
            return None
        cols = min(w, n)
        bordes = np.linspace(0, n, cols + 1).astype(np.int64)
        starts = bordes[:-1]
        # reduceat reduce cada segmento audio[starts[i]:starts[i+1]]
        maxs = np.maximum.reduceat(self.audio, starts)
        mins = np.minimum.reduceat(self.audio, starts)
        return mins, maxs

    # ------------------------------------------------------------------ #
    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, _FONDO)
        medio = h / 2

        # Eje central.
        p.setPen(QPen(_EJE, 1))
        p.drawLine(0, int(medio), w, int(medio))

        if len(self.audio) == 0:
            p.setPen(QPen(QColor(120, 126, 140), 1))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "sin audio")
            return

        # Recalcular envolvente solo si cambió el ancho.
        if self._picos is None or self._cache_w != w:
            self._picos = self._calcular_picos(w)
            self._cache_w = w
        if self._picos is None:
            return
        mins, maxs = self._picos
        cols = len(maxs)

        p.setPen(QPen(_ONDA, 1))
        amp = (h / 2) * 0.92
        for x in range(cols):
            y1 = int(medio - maxs[x] * amp)
            y2 = int(medio - mins[x] * amp)
            px = int(x / cols * w)
            p.drawLine(px, y1, px, y2)

        # Playhead.
        dur = self.duracion
        if dur > 0:
            xph = int(min(self.pos_seg / dur, 1.0) * w)
            p.setPen(QPen(_PLAYHEAD, 2))
            p.drawLine(xph, 0, xph, h)

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev):
        dur = self.duracion
        if dur <= 0 or self.width() <= 0:
            return
        frac = min(max(ev.position().x() / self.width(), 0.0), 1.0)
        self.seek_solicitado.emit(frac * dur)
