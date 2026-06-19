# 🎙️ Autotune Studio

Estudio de voz casero en **Python + PyQt6** para grabar tu voz, **afinarla** (autotune),
pulirla con efectos de estudio (EQ, compresión, reverb) y —el objetivo final—
**reemplazar la voz IA de canciones generadas por IA (p. ej. Suno) por tu propia
voz** manteniendo el instrumental.

---

## ✨ Qué hace

- **Grabar / cargar** voz desde micrófono o archivo.
- **Afinar (autotune)** a la escala que elijas, con control de fuerza y un
  ajuste **Natural ↔ Robótico** (estilo T‑Pain).
- **Pulir** la voz: paso alto, recorte de medios "encajonados", compresión,
  brillo de estudio y reverb.
- **Separar voz / instrumental** de cualquier canción usando **Demucs** (y
  guardarlas por separado), de a una o en lote.
- **Alinear** tu voz a la duración original (time‑stretch) y **mezclar** con el
  instrumental para armar la canción final.
- **Analizar** una canción: tempo, tonalidad, rango vocal, loudness y brillo.

---

## 📂 Estructura del proyecto

```
AutoPyTune/
├── README.md            Este archivo
├── requirements.txt     Dependencias
├── .gitignore
├── run.py               ▶ Punto de entrada (lanza la GUI)
├── src/                 Código fuente
│   ├── audio_engine.py  Motor de audio (sin GUI): grabar, afinar, separar, mezclar
│   ├── gui.py           Interfaz PyQt6
│   ├── analizar.py      Ficha técnica de una canción (tempo, tonalidad, rango...)
│   ├── separar_lote.py  Separación por lotes (estéreo) de una carpeta
│   └── autotune_cli.py  Versión de consola (sin interfaz)
├── docs/
│   └── FAQ.md           Lógica del proyecto en preguntas
└── data/                Material de trabajo (el audio NO se versiona)
    ├── 00_originales/       Canciones fuente
    ├── 01_instrumentales/   Instrumentales separados
    ├── 02_mi_voz/           Grabaciones de voz
    └── 03_finales/          Mezclas terminadas
```

> **Por qué así:** el código vive en `src/`, los datos en `data/` y la
> documentación en `docs/`. Así el repo se mantiene limpio, el audio pesado no
> ensucia el control de versiones (ver `.gitignore`) y cualquiera entiende el
> flujo con solo mirar las carpetas numeradas. Las carpetas de `data/` traen un
> `.gitkeep` para existir vacías; vos ponés tu propio audio adentro.

---

## 🚀 Instalación

Requiere **Python 3.10–3.12** (recomendado **3.11**: `demucs`/`torch` todavía no
publican ruedas estables para 3.14). Conviene un entorno virtual propio:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # En Windows
pip install -r requirements.txt
```

Para el flujo de reemplazo de voz hacen falta además `demucs` y `torch` (van en
`requirements.txt`). Con **GPU NVIDIA** instalá la versión CUDA de torch para
acelerar la separación:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`pyrubberband` es **opcional** (mejora el time‑stretch). Si no está instalado,
el programa usa `librosa` automáticamente.

---

## ▶️ Uso

```bash
python run.py
```

### Flujo A — Afinar una voz suelta
1. **Grabar audio** (sin límite de tiempo: arranca y para con el mismo botón) o
   **Cargar audio**.
2. Elegí **tónica**, **escala**, **fuerza**, **estilo**, **reverb** y **brillo**.
3. **✨ AFINAR (solo voz)** → escuchá el **Resultado** y **Guardá**.

### Flujo B — Reemplazar la voz de una canción
1. **📀 Cargar canción** (la pista completa generada por IA).
2. **🎤 Cargar mi voz** (tu grabación cantando esa canción).
3. Ajustá tónica/escala/efectos y las **ganancias** de voz e instrumental.
4. **🎛️ REEMPLAZAR VOZ Y MEZCLAR** → la app separa el instrumental, afina tu
   voz, la alinea y mezcla.

> 💡 **Truco:** grabá escuchando el instrumental con auriculares para entrar a
> tiempo; así la alineación queda casi perfecta.

### Herramientas extra
- **✂️ Separar 1 canción** → corre Demucs y guarda instrumental + voz por separado.
- **📚 Separar TODOS los originales (lote)** → procesa toda la carpeta
  `data/00_originales/`. También por consola:
  ```bash
  python src/separar_lote.py "data/00_originales"
  ```
- **🔬 Analizar canción** → muestra tempo, tonalidad estimada, rango vocal,
  loudness y brillo. También por consola:
  ```bash
  python src/analizar.py "data/00_originales"
  ```

---

## 🧠 Cómo funciona (resumen técnico)

| Etapa | Herramienta | Función en `audio_engine.py` |
|-------|-------------|------------------------------|
| Detección de tono | `librosa.pyin` | `afinar` |
| Corrección a escala | redondeo a la nota más cercana | `_ajustar_a_escala` |
| Natural vs robótico | suavizado EMA del tono | `_suavizar_transiciones` |
| Desplazar tono | `psola.vocode` | `afinar` |
| EQ + compresión + reverb | `pedalboard` | `pulir` |
| Separar voz/instrumental | `demucs` (htdemucs) | `separar_pistas` |
| Alinear duración | Rubber Band / `librosa` | `ajustar_duracion` |
| Mezcla final | suma + normalización | `mezclar` |
| Análisis de canción | `librosa` (tempo, chroma, RMS) | `src/analizar.py` |

Más detalle y decisiones de diseño en **[docs/FAQ.md](docs/FAQ.md)**.

---

## ⚠️ Limitaciones honestas

- El autotune funciona mucho mejor con **grabaciones limpias** y la **escala
  correcta**.
- La alineación ajusta la **duración total**, no sincroniza frase por frase.
- Esto **no clona tu timbre** ni te convierte en otro cantante: para eso se
  necesita *singing voice conversion* (RVC), un paso futuro posible.
