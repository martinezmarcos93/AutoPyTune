"""
VistaKaraoke — letra sincronizada que se resalta palabra por palabra.

Lee la lista de palabras cronometradas (`data/07_karaoke/NN-Tema.json`,
{palabra, inicio, fin}) y, a medida que avanza el tiempo de reproducción del
instrumental, resalta la palabra que toca cantar. Los saltos de línea se infieren
de los silencios entre palabras (fin de frase), así que no depende del .txt.

El resaltado usa `QTextEdit.setExtraSelections` (no reconstruye el documento en
cada tick) y solo se actualiza cuando cambia la palabra activa. La fuente del
timing es la voz IA extraída, alineada en el tiempo con el instrumental (misma
separación Demucs), así que los tiempos valen para reproducir sobre el instrumental.
"""

from bisect import bisect_right

from PyQt6.QtGui import QTextCharFormat, QColor, QTextCursor, QFont
from PyQt6.QtWidgets import QTextEdit

# Silencio (en s) entre el fin de una palabra y el inicio de la siguiente a
# partir del cual se mete un salto de línea (corte de frase/verso).
GAP_LINEA = 0.7
# Gracia (en s) tras el fin de una palabra antes de apagar su resaltado: en los
# interludios instrumentales largos (sin canto) no queda ninguna palabra "pegada".
GRACIA_FIN = 1.5
# Largo máximo aproximado de línea antes de cortar igual (legibilidad).
MAX_CHARS_LINEA = 52

_BG_ACTIVA = QColor(108, 92, 231)      # violeta de la paleta del proyecto
_FG_ACTIVA = QColor(255, 255, 255)


class VistaKaraoke(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setObjectName("karaoke")
        f = QFont()
        f.setPointSize(15)
        self.setFont(f)

        self._palabras = []        # [{palabra, inicio, fin}, ...]
        self._inicios = []         # inicios (para bisect)
        self._spans = []           # (char_ini, char_fin) por palabra
        self._activa = -1

        self._fmt = QTextCharFormat()
        self._fmt.setBackground(_BG_ACTIVA)
        self._fmt.setForeground(_FG_ACTIVA)

    # ------------------------------------------------------------------ #
    def cargar(self, palabras):
        """Carga la lista de palabras cronometradas y arma el texto."""
        self._palabras = palabras or []
        self._inicios = [p["inicio"] for p in self._palabras
                         if p.get("inicio") is not None]
        self._activa = -1
        self.setExtraSelections([])

        partes = []
        self._spans = []
        cursor_char = 0
        largo_linea = 0
        prev_fin = None
        for i, p in enumerate(self._palabras):
            palabra = p["palabra"]
            ini = p.get("inicio")
            salto = False
            if i > 0:
                hueco = (ini - prev_fin) if (ini is not None and prev_fin is not None) else 0
                if hueco is not None and hueco > GAP_LINEA:
                    salto = True
                elif largo_linea + len(palabra) > MAX_CHARS_LINEA:
                    salto = True

            if salto:
                partes.append("\n")
                cursor_char += 1
                largo_linea = 0
            elif i > 0:
                partes.append(" ")
                cursor_char += 1
                largo_linea += 1

            partes.append(palabra)
            self._spans.append((cursor_char, cursor_char + len(palabra)))
            cursor_char += len(palabra)
            largo_linea += len(palabra)
            if p.get("fin") is not None:
                prev_fin = p["fin"]

        self.setPlainText("".join(partes))

    # ------------------------------------------------------------------ #
    def set_tiempo(self, segundos):
        """Resalta la palabra que corresponde al instante `segundos`."""
        if not self._inicios:
            return
        # Última palabra cuyo inicio <= segundos.
        idx = bisect_right(self._inicios, segundos) - 1
        # Si ya pasó el fin de esa palabra + gracia, estamos en un hueco
        # (instrumental/silencio): no resaltar nada hasta la próxima palabra.
        if 0 <= idx < len(self._palabras):
            fin = self._palabras[idx].get("fin")
            if fin is not None and segundos > fin + GRACIA_FIN:
                idx = -1
        if idx == self._activa:
            return
        self._activa = idx
        if idx < 0 or idx >= len(self._spans):
            self.setExtraSelections([])
            return

        ini, fin = self._spans[idx]
        cursor = self.textCursor()
        cursor.setPosition(ini)
        cursor.setPosition(fin, QTextCursor.MoveMode.KeepAnchor)

        sel = QTextEdit.ExtraSelection()
        sel.cursor = cursor
        sel.format = self._fmt
        self.setExtraSelections([sel])

        # Autoscroll: llevar la palabra activa a la vista sin mostrar el caret.
        visible = self.textCursor()
        visible.setPosition(ini)
        self.setTextCursor(visible)
        self.ensureCursorVisible()

    def reiniciar(self):
        self._activa = -1
        self.setExtraSelections([])
        if self._spans:
            c = self.textCursor()
            c.setPosition(0)
            self.setTextCursor(c)
