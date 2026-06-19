# ❓ FAQ — Lógica y decisiones de Autotune Studio

Documento vivo. Cada pregunta guarda **por qué** está hecho así, no solo el qué.

---

### 1. ¿Cuál es el objetivo final del proyecto?

Partir de una canción con voz generada por IA (p. ej. hecha con **Suno**) y
**reemplazar esa voz IA por una voz real** cantando, afinada y pulida, sobre el
mismo instrumental. El proyecto cubre toda la cadena: grabar → separar → afinar
→ alinear → mezclar.

---

### 2. ¿Por qué el código está partido en `audio_engine.py` y `gui.py`?

Para **separar la lógica de audio de la interfaz**. El motor (`audio_engine.py`)
no sabe nada de PyQt6: son funciones puras que reciben audio y devuelven audio.
Eso permite reusarlo desde la GUI, desde la consola (`autotune_cli.py`) o desde
un futuro script por lotes, sin duplicar nada.

---

### 3. ¿Cómo se afina la voz exactamente?

Tres pasos en `afinar()`:
1. **`librosa.pyin`** detecta la frecuencia (nota) cuadro por cuadro.
2. **`_ajustar_a_escala`** redondea cada nota a la más cercana **dentro de la
   escala** elegida (no a cualquier semitono, salvo en modo Cromática).
3. **`psola.vocode`** desplaza el tono real hacia el objetivo preservando el
   timbre (por eso no suena a "ardilla").

El parámetro **fuerza** mezcla entre tono original (0.0) y tono clavado (1.0).

---

### 4. ¿Qué diferencia hay entre el modo "Natural" y el "Robótico"?

Lo controla el slider **Estilo** → parámetro `suavizado` en
`_suavizar_transiciones()`. Con suavizado 0 la nota salta de golpe al valor
corregido (efecto robótico tipo T‑Pain). Con suavizado alto se aplica un filtro
**EMA** que hace que el tono *glise* gradualmente entre notas, imitando el
portamento natural de un cantante. Se reinicia en los silencios para no arrastrar
tono entre frases.

---

### 5. ¿Qué hace la etapa de "pulido" y por qué esa cadena de efectos?

En `pulir()`, con **pedalboard**, en este orden:
1. **Paso alto 80 Hz** — quita retumbe y ruido grave.
2. **Peak −2 dB @ 300 Hz** — saca el sonido "encajonado" de grabar en cuartos sin acústica.
3. **Compresor** — nivela el volumen (los amateurs cantan partes fuertes y otras flojas).
4. **High shelf @ 10 kHz** (slider **Brillo**) — el "aire" de estudio.
5. **Reverb** — da sensación de espacio y disimula imperfecciones de la afinación.

El orden importa: primero limpiar/ecualizar, después comprimir, y el espacio
(reverb) al final.

---

### 6. ¿Por qué Demucs para separar la voz y no otra cosa?

**Demucs (htdemucs, de Meta)** da hoy la mejor calidad de separación
voz/instrumental con una API de Python sencilla. Lo usamos en `separar_pistas()`.
Detalles que el código maneja y son fáciles de olvidar:
- htdemucs espera **estéreo (2 canales)** → si entra mono, se duplica.
- Necesita **normalización mean/std** antes y deshacerla después.
- Devuelve 4 pistas (drums, bass, other, vocals); el **instrumental** es la suma
  de todas menos `vocals`.

---

### 7. ¿Por qué mi voz grabada no encaja en duración con la canción?

Porque la cantaste a un tempo ligeramente distinto. `ajustar_duracion()` aplica
**time‑stretch** (estirar/comprimir el tiempo **sin cambiar el tono**) para que
dure exactamente lo mismo que la voz original separada. Convención clave:
`rate = duración_actual / duración_objetivo` (rate > 1 acorta).

---

### 8. ¿Por qué `pyrubberband` es opcional?

Da mejor calidad de time‑stretch, pero **necesita un binario externo** (Rubber
Band CLI) que `pip` no instala — un dolor en Windows. Por eso `ajustar_duracion()`
lo intenta y, si falla, cae automáticamente al phase‑vocoder de **librosa**, que
es solo‑pip. El programa funciona igual sin instalarlo.

---

### 9. ¿Por qué la interfaz usa hilos (`QThread`)?

Separar con Demucs o procesar audio largo tarda **segundos o minutos**. Si se
hiciera en el hilo principal, la ventana se **congelaría**. Por eso
`ProcesoThread` y `ProcesoSunoThread` corren en segundo plano y emiten señales
de **progreso** para mover la barra sin bloquear la UI.

---

### 10. ¿Qué NO hace este proyecto y cuál sería el siguiente paso?

No **clona tu timbre** ni te convierte en otro cantante: solo corrige tu
afinación y pule tu voz real. Tampoco sincroniza palabra por palabra (solo
duración total). El siguiente salto sería **singing voice conversion** con
**RVC** para cambiar el timbre manteniendo tu melodía — es otro stack (modelos
IA pesados) y quedaría como evolución futura sobre este mismo motor.

---

### 11. ¿Necesito Audacity u otro programa para separar la voz del instrumental?

**No.** La separación la hace el propio proyecto con **Demucs** dentro de
`separar_pistas()` — no hace falta Audacity, ni plugins, ni subir nada a una web.
Audacity por sí solo **no separa bien** voz de instrumental (sus efectos
"vocal removal" cancelan el centro del estéreo y arruinan el audio); Demucs usa
un modelo de IA y da un resultado muy superior.

Lo único que se necesita es tener las dependencias pesadas instaladas
(`torch` + `demucs`) y los archivos de audio en `data/00_originales/`. Una vez
instalado el entorno con `requirements.txt`, ya se puede separar sin
herramientas extra.

> Audacity puede seguir siendo útil **después**, como editor manual para recortar
> silencios o retocar a mano, pero no es parte obligatoria del flujo.

---

## 📝 Cómo usar este FAQ

Cada vez que tomemos una decisión de diseño o resolvamos un problema no obvio,
agregá una pregunta aquí con su **por qué**. Así el FAQ se vuelve la memoria del
proyecto.
