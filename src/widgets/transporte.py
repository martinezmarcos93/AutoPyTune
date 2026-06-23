"""
BarraTransporte — controles de reproducción estilo DAW.

Play/Pausa · Stop · tiempo actual/total · selector de fuente (Original /
Resultado) · volumen. No reproduce nada por sí misma: emite señales que la
ventana conecta al Reproductor.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QSlider, QComboBox, QHBoxLayout,
)


def _mmss(segundos):
    segundos = int(max(0, segundos))
    return f"{segundos // 60}:{segundos % 60:02d}"


class BarraTransporte(QWidget):
    play_pausa = pyqtSignal()
    stop = pyqtSignal()
    fuente_cambiada = pyqtSignal(int)     # índice del combo
    volumen_cambiado = pyqtSignal(float)  # 0.0 .. 1.5

    def __init__(self):
        super().__init__()
        fila = QHBoxLayout(self)
        fila.setContentsMargins(0, 0, 0, 0)
        fila.setSpacing(10)

        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("transporte")
        self.btn_play.setFixedWidth(54)
        self.btn_play.clicked.connect(self.play_pausa.emit)

        self.btn_stop = QPushButton("■")
        self.btn_stop.setObjectName("transporte")
        self.btn_stop.setFixedWidth(54)
        self.btn_stop.clicked.connect(self.stop.emit)

        self.lbl_tiempo = QLabel("0:00 / 0:00")
        self.lbl_tiempo.setObjectName("tiempo")
        self.lbl_tiempo.setMinimumWidth(110)

        self.combo_fuente = QComboBox()
        self.combo_fuente.addItems(["Original", "Resultado"])
        self.combo_fuente.currentIndexChanged.connect(self.fuente_cambiada.emit)

        lbl_vol = QLabel("🔊")
        self.sl_vol = QSlider(Qt.Orientation.Horizontal)
        self.sl_vol.setRange(0, 150)
        self.sl_vol.setValue(100)
        self.sl_vol.setFixedWidth(120)
        self.sl_vol.valueChanged.connect(
            lambda v: self.volumen_cambiado.emit(v / 100.0))

        fila.addWidget(self.btn_play)
        fila.addWidget(self.btn_stop)
        fila.addWidget(self.lbl_tiempo)
        fila.addStretch(1)
        fila.addWidget(self.combo_fuente)
        fila.addWidget(lbl_vol)
        fila.addWidget(self.sl_vol)

    # ------------------------------------------------------------------ #
    def set_tiempo(self, actual, total):
        self.lbl_tiempo.setText(f"{_mmss(actual)} / {_mmss(total)}")

    def set_reproduciendo(self, on):
        self.btn_play.setText("⏸" if on else "▶")

    def set_fuentes_disponibles(self, original, resultado):
        """Habilita/deshabilita opciones del selector según haya audio."""
        modelo = self.combo_fuente.model()
        modelo.item(0).setEnabled(original)
        modelo.item(1).setEnabled(resultado)
