"""
Autotune Studio — interfaz gráfica en PyQt6.

Ejecutar:
    python run.py

Requisitos:
    pip install -r requirements.txt
"""

import os
import sys
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QSlider,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QProgressBar, QMessageBox, QCheckBox, QScrollArea,
)

import audio_engine as eng
from widgets.reproductor import Reproductor
from widgets.waveform import VistaOnda
from widgets.transporte import BarraTransporte

# Rutas del proyecto (gui.py vive en src/, los datos en ../data/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_ORIGINALES = os.path.join(ROOT, "data", "00_originales")
DIR_INSTRUMENTALES = os.path.join(ROOT, "data", "01_instrumentales")


# --------------------------------------------------------------------------- #
# Grabadora continua (sin límite de tiempo: arranca y para con un botón)
# --------------------------------------------------------------------------- #
class Grabadora:
    """Captura del micrófono hasta que se la detiene, sin límite de duración."""

    def __init__(self):
        self.frames = []
        self.stream = None

    @property
    def activa(self):
        return self.stream is not None

    def iniciar(self):
        import sounddevice as sd
        self.frames = []

        def callback(indata, n, t, status):   # corre en el hilo de sounddevice
            self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=eng.SR, channels=1, dtype="float32", callback=callback)
        self.stream.start()

    def detener(self):
        if self.stream is None:
            return np.zeros(0, dtype="float32")
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.frames:
            return np.zeros(0, dtype="float32")
        return np.concatenate(self.frames, axis=0).flatten()


# --------------------------------------------------------------------------- #
# Hilos de trabajo (para no congelar la interfaz)
# --------------------------------------------------------------------------- #
class ProcesoThread(QThread):
    progreso = pyqtSignal(str, int)
    terminado = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, audio, tonica, escala, fuerza, reverb, suavizado, brillo):
        super().__init__()
        self.audio = audio
        self.tonica = tonica
        self.escala = escala
        self.fuerza = fuerza
        self.reverb = reverb
        self.suavizado = suavizado
        self.brillo = brillo

    def run(self):
        try:
            final = eng.procesar(
                self.audio, self.tonica, self.escala, self.fuerza, self.reverb,
                suavizado=self.suavizado, brillo=self.brillo,
                progreso=lambda msg, pct: self.progreso.emit(msg, pct),
            )
            self.terminado.emit(final)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class ProcesoSunoThread(QThread):
    """Separa la canción de Suno, afina tu voz y mezcla, en segundo plano."""
    progreso = pyqtSignal(str, int)
    terminado = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ruta_suno, ruta_mi_voz, tonica, escala, fuerza,
                 reverb, suavizado, brillo, ganancia_voz, ganancia_inst, alinear):
        super().__init__()
        self.ruta_suno = ruta_suno
        self.ruta_mi_voz = ruta_mi_voz
        self.tonica = tonica
        self.escala = escala
        self.fuerza = fuerza
        self.reverb = reverb
        self.suavizado = suavizado
        self.brillo = brillo
        self.ganancia_voz = ganancia_voz
        self.ganancia_inst = ganancia_inst
        self.alinear = alinear

    def run(self):
        try:
            final = eng.reemplazar_voz(
                self.ruta_suno, self.ruta_mi_voz, self.tonica, self.escala,
                self.fuerza, self.reverb,
                suavizado=self.suavizado, brillo=self.brillo,
                ganancia_voz=self.ganancia_voz, ganancia_inst=self.ganancia_inst,
                alinear=self.alinear,
                progreso=lambda msg, pct: self.progreso.emit(msg, pct),
            )
            self.terminado.emit(final)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class AnalisisThread(QThread):
    """Calcula la ficha técnica (tempo, tonalidad, rango...) de un archivo."""
    terminado = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, ruta):
        super().__init__()
        self.ruta = ruta

    def run(self):
        try:
            import analizar
            self.terminado.emit(analizar.ficha(self.ruta))
        except Exception as e:
            self.error.emit(str(e))


class SepararThread(QThread):
    """Separa un archivo en voz e instrumental y los guarda como WAV."""
    progreso = pyqtSignal(str, int)
    terminado = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, ruta, carpeta_destino):
        super().__init__()
        self.ruta = ruta
        self.carpeta_destino = carpeta_destino

    def run(self):
        try:
            voz, inst = eng.separar_pistas(
                self.ruta, progreso=lambda m, p: self.progreso.emit(m, p))
            base = os.path.splitext(os.path.basename(self.ruta))[0]
            os.makedirs(self.carpeta_destino, exist_ok=True)
            ruta_inst = os.path.join(self.carpeta_destino, f"{base}_instrumental.wav")
            ruta_voz = os.path.join(self.carpeta_destino, f"{base}_voz_original.wav")
            eng.guardar(ruta_inst, inst)
            eng.guardar(ruta_voz, voz)
            self.terminado.emit(ruta_inst)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class SepararLoteThread(QThread):
    """Separa TODAS las canciones de data/00_originales/ (lo que antes se hacía
    por terminal). Instrumentales -> 01_instrumentales, voces -> 04_voces_extraidas."""
    progreso = pyqtSignal(str, int)
    terminado = pyqtSignal(str)
    error = pyqtSignal(str)

    def run(self):
        try:
            import separar_lote
            self.progreso.emit("Separando todos los temas (esto tarda)...", 5)
            separar_lote.separar_lote(separar_lote.DIR_ORIGINALES)
            self.terminado.emit(str(separar_lote.DIR_INSTRUMENTALES))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------- #
# Ventana principal
# --------------------------------------------------------------------------- #
class AutotuneStudio(QWidget):
    def __init__(self):
        super().__init__()
        self.audio_original = None
        self.audio_procesado = None
        self.grabadora = Grabadora()
        self.segundos_grab = 0
        self.proceso = None
        self.proceso_suno = None
        self.ruta_suno = None
        self.ruta_mi_voz = None

        # Reproductor con playhead (Incremento A del rediseño DAW).
        self.reproductor = Reproductor(eng.SR)
        self.fuente_idx = 0          # 0 = original, 1 = resultado

        # Timer para mostrar el tiempo de grabación en vivo.
        self.timer_grab = QTimer(self)
        self.timer_grab.timeout.connect(self._tic_grabacion)

        # Timer del playhead (~60 FPS) mientras se reproduce.
        self.timer_play = QTimer(self)
        self.timer_play.setInterval(16)
        self.timer_play.timeout.connect(self._tick_play)

        self.setWindowTitle("Autotune Studio")
        self.setMinimumSize(720, 600)
        self._construir_ui()
        self._aplicar_estilo()

    # --------------------------------------------------------------------- #
    # Construcción de la interfaz
    # --------------------------------------------------------------------- #
    def _construir_ui(self):
        # La ventana solo contiene un área con scroll; así, aunque esté chica o
        # haya muchas tarjetas, el contenido nunca se aplasta: se desplaza.
        externo = QVBoxLayout(self)
        externo.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        externo.addWidget(scroll)

        # El contenido se centra en una columna de ancho acotado para que se vea
        # bien también a pantalla completa.
        envoltura = QWidget()
        env_layout = QHBoxLayout(envoltura)
        env_layout.setContentsMargins(0, 0, 0, 0)
        env_layout.addStretch(1)

        columna = QWidget()
        columna.setMaximumWidth(960)
        root = QVBoxLayout(columna)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        env_layout.addWidget(columna, 6)
        env_layout.addStretch(1)
        scroll.setWidget(envoltura)

        # ---- Cabecera ----
        titulo = QLabel("🎙  AUTOTUNE STUDIO")
        titulo.setObjectName("titulo")
        subtitulo = QLabel("Grabá tu voz · afinala · pulila · reemplazá la voz de Suno")
        subtitulo.setObjectName("subtitulo")
        root.addWidget(titulo)
        root.addWidget(subtitulo)

        # ---- Tarjeta: Fuente de audio ----
        fuente = self._tarjeta("1 · Fuente de audio")
        fl = QHBoxLayout()
        self.btn_grabar = QPushButton("🎙  Grabar audio")
        self.btn_grabar.setObjectName("rec")
        self.btn_grabar.setMinimumHeight(46)
        self.btn_grabar.clicked.connect(self.al_grabar)

        self.btn_cargar = QPushButton("📂  Cargar audio")
        self.btn_cargar.setMinimumHeight(46)
        self.btn_cargar.clicked.connect(self.al_cargar)

        fl.addWidget(self.btn_grabar, 1)
        fl.addWidget(self.btn_cargar, 1)
        fuente.layout().addLayout(fl)

        self.lbl_fuente = QLabel("Sin audio cargado. La grabación no tiene límite "
                                 "de tiempo: apretá «Grabar audio» y apretá otra "
                                 "vez para detener.")
        self.lbl_fuente.setObjectName("estado")
        self.lbl_fuente.setWordWrap(True)
        fuente.layout().addWidget(self.lbl_fuente)
        root.addWidget(fuente)

        # ---- Tarjeta: Afinación ----
        ajustes = self._tarjeta("2 · Afinación")
        grid = QGridLayout()
        grid.setVerticalSpacing(14)
        grid.setHorizontalSpacing(12)

        self.combo_tonica = QComboBox()
        self.combo_tonica.addItems(eng.NOTAS)
        self.combo_escala = QComboBox()
        self.combo_escala.addItems(eng.ESCALAS.keys())

        grid.addWidget(QLabel("Tónica"), 0, 0)
        grid.addWidget(self.combo_tonica, 0, 1)
        grid.addWidget(QLabel("Escala"), 0, 2)
        grid.addWidget(self.combo_escala, 0, 3)

        self.sl_fuerza = self._slider(0, 100, 100)
        self.lbl_fuerza = QLabel("100%")
        self.sl_fuerza.valueChanged.connect(
            lambda v: self.lbl_fuerza.setText(f"{v}%"))
        grid.addWidget(QLabel("Fuerza"), 1, 0)
        grid.addWidget(self.sl_fuerza, 1, 1, 1, 2)
        grid.addWidget(self.lbl_fuerza, 1, 3)

        self.sl_suave = self._slider(0, 100, 35)
        self.lbl_suave = QLabel("Natural 35%")
        self.sl_suave.valueChanged.connect(
            lambda v: self.lbl_suave.setText(
                "Robot" if v == 0 else f"Natural {v}%"))
        grid.addWidget(QLabel("Estilo"), 2, 0)
        grid.addWidget(self.sl_suave, 2, 1, 1, 2)
        grid.addWidget(self.lbl_suave, 2, 3)

        self.sl_reverb = self._slider(0, 60, 25)
        self.lbl_reverb = QLabel("25%")
        self.sl_reverb.valueChanged.connect(
            lambda v: self.lbl_reverb.setText(f"{v}%"))
        grid.addWidget(QLabel("Reverb"), 3, 0)
        grid.addWidget(self.sl_reverb, 3, 1, 1, 2)
        grid.addWidget(self.lbl_reverb, 3, 3)

        self.sl_brillo = self._slider(0, 8, 3)   # dB en high shelf 10 kHz
        self.lbl_brillo = QLabel("+3 dB")
        self.sl_brillo.valueChanged.connect(
            lambda v: self.lbl_brillo.setText(f"+{v} dB"))
        grid.addWidget(QLabel("Brillo"), 4, 0)
        grid.addWidget(self.sl_brillo, 4, 1, 1, 2)
        grid.addWidget(self.lbl_brillo, 4, 3)

        # Que la columna de los sliders sea la que se estire.
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        ajustes.layout().addLayout(grid)

        self.btn_procesar = QPushButton("✨  AFINAR (solo voz)")
        self.btn_procesar.setObjectName("procesar")
        self.btn_procesar.setEnabled(False)
        self.btn_procesar.clicked.connect(self.al_procesar)
        ajustes.layout().addWidget(self.btn_procesar)
        root.addWidget(ajustes)

        # ---- Tarjeta: Reemplazar voz de Suno ----
        suno = self._tarjeta("3 · Reemplazar voz de Suno")
        sl = QVBoxLayout()
        sl.setSpacing(10)

        h1 = QHBoxLayout()
        self.btn_cargar_suno = QPushButton("📀  Cargar canción Suno")
        self.btn_cargar_suno.setMinimumHeight(42)
        self.btn_cargar_suno.clicked.connect(self.al_cargar_suno)
        self.btn_cargar_mi_voz = QPushButton("🎤  Cargar mi voz")
        self.btn_cargar_mi_voz.setMinimumHeight(42)
        self.btn_cargar_mi_voz.clicked.connect(self.al_cargar_mi_voz)
        h1.addWidget(self.btn_cargar_suno)
        h1.addWidget(self.btn_cargar_mi_voz)
        sl.addLayout(h1)

        self.lbl_suno = QLabel("Suno: (ninguno)")
        self.lbl_suno.setObjectName("estado")
        self.lbl_mi_voz = QLabel("Mi voz: (ninguno)")
        self.lbl_mi_voz.setObjectName("estado")
        sl.addWidget(self.lbl_suno)
        sl.addWidget(self.lbl_mi_voz)

        h3 = QHBoxLayout()
        lbl_v = QLabel("Voz")
        lbl_v.setMinimumWidth(90)
        h3.addWidget(lbl_v)
        self.sl_gan_voz = self._slider(50, 150, 110)   # /100 -> 0.50 a 1.50
        self.lbl_gan_voz = QLabel("1.10")
        self.sl_gan_voz.valueChanged.connect(
            lambda v: self.lbl_gan_voz.setText(f"{v/100:.2f}"))
        h3.addWidget(self.sl_gan_voz)
        h3.addWidget(self.lbl_gan_voz)
        sl.addLayout(h3)

        h3b = QHBoxLayout()
        lbl_i = QLabel("Instrumental")
        lbl_i.setMinimumWidth(90)
        h3b.addWidget(lbl_i)
        self.sl_gan_inst = self._slider(30, 130, 90)   # /100 -> 0.30 a 1.30
        self.lbl_gan_inst = QLabel("0.90")
        self.sl_gan_inst.valueChanged.connect(
            lambda v: self.lbl_gan_inst.setText(f"{v/100:.2f}"))
        h3b.addWidget(self.sl_gan_inst)
        h3b.addWidget(self.lbl_gan_inst)
        sl.addLayout(h3b)

        self.chk_alinear = QCheckBox("Alinear mi voz a la duración original (time-stretch)")
        self.chk_alinear.setChecked(True)
        sl.addWidget(self.chk_alinear)

        self.btn_reemplazar = QPushButton("🎛️  REEMPLAZAR VOZ Y MEZCLAR")
        self.btn_reemplazar.setObjectName("procesar")
        self.btn_reemplazar.setEnabled(False)
        self.btn_reemplazar.clicked.connect(self.al_reemplazar)
        sl.addWidget(self.btn_reemplazar)

        suno.layout().addLayout(sl)
        root.addWidget(suno)

        # ---- Tarjeta: Herramientas ----
        herr = self._tarjeta("4 · Herramientas")
        hg = QGridLayout()
        hg.setHorizontalSpacing(12)
        hg.setVerticalSpacing(10)

        self.btn_separar = QPushButton("✂️  Separar 1 canción (voz / instrumental)")
        self.btn_separar.setMinimumHeight(42)
        self.btn_separar.clicked.connect(self.al_separar)

        self.btn_separar_lote = QPushButton("📚  Separar TODOS los originales (lote)")
        self.btn_separar_lote.setMinimumHeight(42)
        self.btn_separar_lote.clicked.connect(self.al_separar_lote)

        self.btn_analizar = QPushButton("🔬  Analizar canción (tempo · tonalidad · rango)")
        self.btn_analizar.setMinimumHeight(42)
        self.btn_analizar.clicked.connect(self.al_analizar)

        hg.addWidget(self.btn_separar, 0, 0)
        hg.addWidget(self.btn_separar_lote, 0, 1)
        hg.addWidget(self.btn_analizar, 1, 0, 1, 2)
        herr.layout().addLayout(hg)
        root.addWidget(herr)

        # ---- Barra de progreso ----
        self.barra = QProgressBar()
        self.barra.setValue(0)
        self.barra.setTextVisible(True)
        root.addWidget(self.barra)

        # ---- Reproductor (waveform + transporte) ----
        reproductor = self._tarjeta("5 · Reproductor")
        self.vista_onda = VistaOnda(eng.SR)
        self.vista_onda.seek_solicitado.connect(self._on_seek)
        reproductor.layout().addWidget(self.vista_onda)

        self.transporte = BarraTransporte()
        self.transporte.play_pausa.connect(self._on_play_pausa)
        self.transporte.stop.connect(self._on_stop)
        self.transporte.fuente_cambiada.connect(self._on_fuente)
        self.transporte.volumen_cambiado.connect(self._on_volumen)
        reproductor.layout().addWidget(self.transporte)

        self.btn_guardar = QPushButton("💾  Guardar resultado")
        self.btn_guardar.setMinimumHeight(42)
        self.btn_guardar.clicked.connect(self.al_guardar)
        reproductor.layout().addWidget(self.btn_guardar)
        root.addWidget(reproductor)

        self._habilitar_resultado(False)
        root.addStretch()

    def _tarjeta(self, titulo):
        marco = QFrame()
        marco.setObjectName("tarjeta")
        v = QVBoxLayout(marco)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(10)
        lbl = QLabel(titulo)
        lbl.setObjectName("seccion")
        v.addWidget(lbl)
        return marco

    def _slider(self, mn, mx, val):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(mn, mx)
        s.setValue(val)
        s.setMinimumHeight(24)
        return s

    # --------------------------------------------------------------------- #
    # Acciones — grabación con arranque/parada (sin límite)
    # --------------------------------------------------------------------- #
    def al_grabar(self):
        if self.grabadora.activa:
            self._detener_grabacion()
        else:
            self._iniciar_grabacion()

    def _iniciar_grabacion(self):
        try:
            self.grabadora.iniciar()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No pude abrir el micrófono:\n{e}")
            return
        self.segundos_grab = 0
        self.timer_grab.start(1000)
        self._bloquear(True, excepto_grabar=True)
        self.btn_grabar.setText("⏹  Detener grabación")
        self.lbl_fuente.setText("● Grabando… 0:00  (apretá de nuevo para detener)")

    def _detener_grabacion(self):
        self.timer_grab.stop()
        audio = self.grabadora.detener()
        self.btn_grabar.setText("🎙  Grabar audio")
        self._bloquear(False)
        if len(audio) == 0:
            self.lbl_fuente.setText("No se capturó audio. Probá de nuevo.")
            return
        self.audio_original = audio
        self.audio_procesado = None
        self._habilitar_resultado(False)
        self.btn_procesar.setEnabled(True)
        dur = len(audio) / eng.SR
        self.lbl_fuente.setText(f"✓ Grabado: {self._mmss(dur)} en memoria.")

    def _tic_grabacion(self):
        self.segundos_grab += 1
        self.lbl_fuente.setText(
            f"● Grabando… {self._mmss(self.segundos_grab)}  "
            f"(apretá de nuevo para detener)")

    @staticmethod
    def _mmss(segundos):
        segundos = int(segundos)
        return f"{segundos // 60}:{segundos % 60:02d}"

    def al_cargar(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Cargar audio", "", "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if not ruta:
            return
        try:
            self.audio_original = eng.cargar(ruta)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No pude leer el archivo:\n{e}")
            return
        self.audio_procesado = None
        self._habilitar_resultado(False)
        self.btn_procesar.setEnabled(True)
        dur = len(self.audio_original) / eng.SR
        self.lbl_fuente.setText(f"✓ {os.path.basename(ruta)}  ({self._mmss(dur)})")

    # ---------------------------------------------------------------- #
    # Acciones para el flujo de Suno
    # ---------------------------------------------------------------- #
    def al_cargar_suno(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Abrir canción de Suno", DIR_ORIGINALES,
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if not ruta:
            return
        self.ruta_suno = ruta
        self.lbl_suno.setText(f"Suno: {os.path.basename(ruta)}")
        self._verificar_reemplazo()

    def al_cargar_mi_voz(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Abrir tu grabación", "", "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if not ruta:
            return
        self.ruta_mi_voz = ruta
        self.lbl_mi_voz.setText(f"Mi voz: {os.path.basename(ruta)}")
        self._verificar_reemplazo()

    def _verificar_reemplazo(self):
        self.btn_reemplazar.setEnabled(
            self.ruta_suno is not None and self.ruta_mi_voz is not None)

    def al_reemplazar(self):
        if not (self.ruta_suno and self.ruta_mi_voz):
            return
        self._bloquear(True)
        self.barra.setValue(0)
        self.proceso_suno = ProcesoSunoThread(
            self.ruta_suno,
            self.ruta_mi_voz,
            self.combo_tonica.currentText(),
            self.combo_escala.currentText(),
            self.sl_fuerza.value() / 100.0,
            self.sl_reverb.value() / 100.0,
            self.sl_suave.value() / 100.0,
            float(self.sl_brillo.value()),
            self.sl_gan_voz.value() / 100.0,
            self.sl_gan_inst.value() / 100.0,
            self.chk_alinear.isChecked(),
        )
        self.proceso_suno.progreso.connect(self.actualizar_progreso)
        self.proceso_suno.terminado.connect(self.reemplazo_listo)
        self.proceso_suno.error.connect(self.proceso_error)
        self.proceso_suno.start()

    def reemplazo_listo(self, audio):
        self.audio_procesado = audio   # reutiliza el reproductor "Resultado"
        self._bloquear(False)
        self._habilitar_resultado(True)
        self.transporte.combo_fuente.setCurrentIndex(1)   # mostrar la mezcla
        self.barra.setValue(100)
        self.barra.setFormat("✓ Mezcla completa  (100%)")

    # ---------------------------------------------------------------- #
    # Herramientas
    # ---------------------------------------------------------------- #
    def al_separar(self):
        ruta = self.ruta_suno
        if not ruta:
            ruta, _ = QFileDialog.getOpenFileName(
                self, "Canción a separar", DIR_ORIGINALES,
                "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if not ruta:
            return
        self._bloquear(True)
        self.barra.setValue(0)
        self.sep = SepararThread(ruta, DIR_INSTRUMENTALES)
        self.sep.progreso.connect(self.actualizar_progreso)
        self.sep.terminado.connect(self.separacion_lista)
        self.sep.error.connect(self.proceso_error)
        self.sep.start()

    def separacion_lista(self, ruta_inst):
        self._bloquear(False)
        self.barra.setValue(100)
        self.barra.setFormat("✓ Pistas guardadas en data/01_instrumentales")
        QMessageBox.information(
            self, "Separación lista",
            f"Guardado en:\n{os.path.dirname(ruta_inst)}\n\n"
            f"• {os.path.basename(ruta_inst)}\n"
            f"• {os.path.basename(ruta_inst).replace('_instrumental', '_voz_original')}")

    def al_separar_lote(self):
        n = len([f for f in os.listdir(DIR_ORIGINALES)
                 if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a"))]) \
            if os.path.isdir(DIR_ORIGINALES) else 0
        if n == 0:
            QMessageBox.warning(self, "Sin originales",
                                "No hay canciones en data/00_originales/.")
            return
        resp = QMessageBox.question(
            self, "Separar todo",
            f"Se van a separar {n} canciones de data/00_originales/.\n"
            "En CPU puede tardar bastante (varios minutos por tema).\n"
            "Las que ya estén separadas se saltean.\n\n¿Continuar?")
        if resp != QMessageBox.StandardButton.Yes:
            return
        self._bloquear(True)
        self.barra.setValue(0)
        self.barra.setFormat("Separando todos los temas (esto tarda)…")
        self.sep_lote = SepararLoteThread()
        self.sep_lote.progreso.connect(self.actualizar_progreso)
        self.sep_lote.terminado.connect(self.lote_listo)
        self.sep_lote.error.connect(self.proceso_error)
        self.sep_lote.start()

    def lote_listo(self, carpeta):
        self._bloquear(False)
        self.barra.setValue(100)
        self.barra.setFormat("✓ Separación por lote completa")
        QMessageBox.information(
            self, "Lote listo",
            "Instrumentales en data/01_instrumentales/\n"
            "Voces en data/04_voces_extraidas/")

    def al_analizar(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Canción a analizar", DIR_ORIGINALES,
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if not ruta:
            return
        self._bloquear(True)
        self.barra.setFormat("🔬 Analizando…")
        self.analisis = AnalisisThread(ruta)
        self.analisis.terminado.connect(self.analisis_listo)
        self.analisis.error.connect(self.proceso_error)
        self.analisis.start()

    def analisis_listo(self, texto):
        self._bloquear(False)
        self.barra.setFormat("✓ Análisis listo")
        QMessageBox.information(self, "Ficha técnica", texto)

    def al_procesar(self):
        if self.audio_original is None:
            return
        self._bloquear(True)
        self.barra.setValue(0)
        self.proceso = ProcesoThread(
            self.audio_original,
            self.combo_tonica.currentText(),
            self.combo_escala.currentText(),
            self.sl_fuerza.value() / 100.0,
            self.sl_reverb.value() / 100.0,
            self.sl_suave.value() / 100.0,
            float(self.sl_brillo.value()),
        )
        self.proceso.progreso.connect(self.actualizar_progreso)
        self.proceso.terminado.connect(self.proceso_listo)
        self.proceso.error.connect(self.proceso_error)
        self.proceso.start()

    def actualizar_progreso(self, msg, pct):
        self.barra.setValue(pct)
        self.barra.setFormat(f"{msg}  ({pct}%)")

    def proceso_listo(self, audio):
        self.audio_procesado = audio
        self._bloquear(False)
        self._habilitar_resultado(True)
        self.transporte.combo_fuente.setCurrentIndex(1)   # mostrar el resultado
        self.barra.setFormat("✓ Listo  (100%)")

    def proceso_error(self, msg):
        self._bloquear(False)
        self.barra.setValue(0)
        self.barra.setFormat("")
        QMessageBox.critical(self, "Error al procesar", msg)

    # --------------------------------------------------------------------- #
    # Reproductor con playhead (waveform + transporte)
    # --------------------------------------------------------------------- #
    def _audio_actual(self):
        return self.audio_procesado if self.fuente_idx == 1 else self.audio_original

    def _refrescar_fuentes(self):
        """Actualiza el selector y carga la onda de la fuente activa."""
        tiene_orig = self.audio_original is not None
        tiene_proc = self.audio_procesado is not None
        self.transporte.set_fuentes_disponibles(tiene_orig, tiene_proc)
        if self.fuente_idx == 1 and not tiene_proc:
            self.fuente_idx = 0
        self._cargar_fuente_en_reproductor()

    def _cargar_fuente_en_reproductor(self):
        audio = self._audio_actual()
        self.reproductor.cargar(audio)
        self.vista_onda.set_audio(audio)
        self.timer_play.stop()
        self.transporte.set_reproduciendo(False)
        self.transporte.set_tiempo(0, self.reproductor.duracion)

    def _on_play_pausa(self):
        if self.reproductor.activo:
            self.reproductor.pausa()
            self.timer_play.stop()
            self.transporte.set_reproduciendo(False)
        else:
            if self.reproductor.duracion <= 0:
                return
            self.reproductor.reproducir()
            self.timer_play.start()
            self.transporte.set_reproduciendo(True)

    def _on_stop(self):
        self.reproductor.detener()
        self.timer_play.stop()
        self.transporte.set_reproduciendo(False)
        self.vista_onda.set_pos(0.0)
        self.transporte.set_tiempo(0, self.reproductor.duracion)

    def _on_fuente(self, idx):
        self.fuente_idx = idx
        self.reproductor.detener()
        self._cargar_fuente_en_reproductor()
        self.vista_onda.set_pos(0.0)

    def _on_volumen(self, v):
        self.reproductor.ganancia = v

    def _on_seek(self, segundos):
        self.reproductor.seek(segundos)
        self.vista_onda.set_pos(segundos)
        self.transporte.set_tiempo(segundos, self.reproductor.duracion)

    def _tick_play(self):
        seg = self.reproductor.segundos()
        self.vista_onda.set_pos(seg)
        self.transporte.set_tiempo(seg, self.reproductor.duracion)
        if not self.reproductor.activo:        # terminó solo al llegar al final
            self.timer_play.stop()
            self.transporte.set_reproduciendo(False)

    def al_guardar(self):
        if self.audio_procesado is None:
            return
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar resultado", "resultado.wav", "WAV (*.wav)")
        if not ruta:
            return
        eng.guardar(ruta, self.audio_procesado)
        self.barra.setFormat(f"💾 Guardado: {os.path.basename(ruta)}")

    # --------------------------------------------------------------------- #
    # Estado de la interfaz
    # --------------------------------------------------------------------- #
    def _bloquear(self, ocupado, excepto_grabar=False):
        controles = (self.btn_cargar, self.btn_procesar,
                     self.combo_tonica, self.combo_escala,
                     self.btn_cargar_suno, self.btn_cargar_mi_voz,
                     self.btn_reemplazar, self.btn_separar,
                     self.btn_separar_lote, self.btn_analizar)
        for w in controles:
            w.setEnabled(not ocupado)
        if not excepto_grabar:
            self.btn_grabar.setEnabled(not ocupado)
        if not ocupado:
            if self.audio_original is None:
                self.btn_procesar.setEnabled(False)
            self._verificar_reemplazo()

    def _habilitar_resultado(self, on):
        self.btn_guardar.setEnabled(on)
        self._refrescar_fuentes()

    # --------------------------------------------------------------------- #
    # Estilo (QSS)
    # --------------------------------------------------------------------- #
    def _aplicar_estilo(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #15171c;
                color: #e6e8ec;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QScrollArea, #scroll { border: none; background-color: #15171c; }
            #titulo {
                font-size: 26px;
                font-weight: 800;
                color: #ffffff;
                letter-spacing: 1px;
            }
            #subtitulo { color: #8a90a0; font-size: 13px; }
            #seccion {
                color: #7c83ff;
                font-weight: 700;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            #estado { color: #9aa0b0; font-size: 13px; }
            #tarjeta {
                background-color: #1d2027;
                border: 1px solid #2a2e38;
                border-radius: 14px;
            }
            QPushButton {
                background-color: #262a33;
                border: 1px solid #343945;
                border-radius: 10px;
                padding: 11px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #2f3540; }
            QPushButton:pressed { background-color: #21252d; }
            QPushButton:disabled { color: #565d6b; background-color: #1b1e24; }
            #rec { background-color: #b8323c; border: none; color: white; font-weight: 700; }
            #rec:hover { background-color: #d23a45; }
            #procesar {
                background-color: #6c5ce7; border: none; color: white;
                font-size: 16px; font-weight: 800; padding: 14px;
            }
            #procesar:hover { background-color: #7d6ef0; }
            #procesar:disabled { background-color: #353147; color: #888; }
            #play { background-color: #1e8e5a; border: none; color: white; }
            #play:hover { background-color: #25a868; }
            #transporte {
                background-color: #6c5ce7; border: none; color: white;
                font-size: 18px; font-weight: 800; padding: 8px;
            }
            #transporte:hover { background-color: #7d6ef0; }
            #transporte:disabled { background-color: #353147; color: #888; }
            #tiempo {
                color: #cfd3dc; font-size: 13px;
                font-family: 'Consolas', monospace;
            }
            QComboBox {
                background-color: #262a33;
                border: 1px solid #343945;
                border-radius: 8px;
                padding: 7px 10px;
            }
            QComboBox::drop-down { border: none; width: 22px; }
            QComboBox QAbstractItemView {
                background-color: #262a33;
                selection-background-color: #6c5ce7;
                border: 1px solid #343945;
            }
            QCheckBox { color: #cfd3dc; }
            QSlider::groove:horizontal {
                height: 6px; background: #343945; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #6c5ce7; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff; width: 16px; height: 16px;
                margin: -6px 0; border-radius: 8px;
            }
            QProgressBar {
                background-color: #1d2027; border: 1px solid #2a2e38;
                border-radius: 9px; height: 22px; text-align: center;
                color: #cfd3dc; font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #6c5ce7; border-radius: 8px;
            }
            QScrollBar:vertical {
                background: #15171c; width: 12px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #343945; border-radius: 6px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #444a58; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)


def main():
    app = QApplication(sys.argv)
    ventana = AutotuneStudio()
    ventana.showMaximized()        # arranca a pantalla completa (ventana maximizada)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
