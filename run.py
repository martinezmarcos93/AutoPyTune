"""
Punto de entrada de Autotune Studio.

Ejecuta la interfaz gráfica:
    python run.py
"""

import os
import sys

# Permite importar los módulos que viven en src/ sin instalar nada.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from gui import main   # noqa: E402

if __name__ == "__main__":
    main()
