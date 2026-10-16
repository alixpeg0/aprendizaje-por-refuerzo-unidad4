"""Guía explicativa, diapositiva por diapositiva, del PDF 16:9 (ejecutar desde la raíz del repo)."""
import re, sys, tempfile
from pathlib import Path
raiz = Path(__file__).resolve().parent.parent

import fitz
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
                                PageBreak)

ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for n, f in [("DV", "DejaVuSans"), ("DVB", "DejaVuSans-Bold")]:
    pdfmetrics.registerFont(TTFont(n, str(ttf / f"{f}.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

AZ, GU, DO = colors.HexColor("#1B3A6B"), colors.HexColor("#8B1A3A"), colors.HexColor("#B8923A")
NOMBRE = "Nombre Apellido"
GH = "https://github.com/alixpeg0/aprendizaje-por-refuerzo-unidad4"
COLAB = GH.replace("github.com/", "colab.research.google.com/github/") + "/blob/main/Unidad4_Aprendizaje_por_Refuerzo.ipynb"
FUENTE = raiz / "pdfs" / "Unidad4_Aprendizaje_por_Refuerzo_TecNM_con_Introduccion.pdf"
SALIDA = raiz / "pdfs" / "Guia_Explicativa_de_Diapositivas.pdf"

sT = ParagraphStyle("t", fontName="DVB", fontSize=22, textColor=AZ, leading=27, spaceAfter=6)
sS = ParagraphStyle("s", fontName="DV", fontSize=10, leading=14, textColor=colors.HexColor("#444"), spaceAfter=4)
sB = ParagraphStyle("b", fontName="DV", fontSize=8.6, leading=12, alignment=0, spaceAfter=3)
sL = ParagraphStyle("l", parent=sB, leftIndent=10, firstLineIndent=-10, spaceAfter=2)
sH = ParagraphStyle("h", fontName="DVB", fontSize=8.8, textColor=GU, spaceBefore=4, spaceAfter=1.5)
sN = ParagraphStyle("n", fontName="DVB", fontSize=10.5, textColor=colors.white, leading=13)
sSec = ParagraphStyle("sec", fontName="DVB", fontSize=14, textColor=AZ, spaceBefore=6, spaceAfter=6)


def md(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t


# (titulo, que_muestra, [explicacion...], idea_clave)
D = []
def S(titulo, muestra, expl, clave, seccion=None): D.append((titulo, muestra, expl, clave, seccion))

S("Portada",
  "El título del trabajo, el índice de secciones, el nombre del alumno y los enlaces a Colab y GitHub.",
  ["Presenta el documento: **Aprendizaje por Refuerzo, Unidad 4**, resuelto con teoría, pasos y código.",
   "El índice lista la introducción y las 7 secciones (teoría general y Actividades 25 a 30).",
   "Los enlaces azules son clicables: **Colab** abre el cuaderno para ejecutarlo; **GitHub** abre el repositorio con todo el código y los PDFs."],
  "Es el punto de entrada: indica qué contiene el documento y dónde está el código ejecutable. Recuerda sustituir «Nombre Apellido» por tu nombre.",
  "Introducción")
S("Introducción · Mapa de toda la unidad",
  "Un diagrama con los tres bloques de la unidad (4.1, 4.2, 4.3), sus temas y las actividades que corresponden a cada uno.",
  ["Arriba, la definición en una frase: un agente aprende a decidir interactuando con un entorno para **maximizar la recompensa acumulada**.",
   "**4.1 Definición y conceptos básicos** (Act. 25 y 26): medio ambiente, estado, acción, recompensa, política y valor. Es el **vocabulario**.",
   "**4.2 Algoritmos** (Act. 27 y 28): valores, políticas y modelos; la ecuación de Bellman y cómo elegir algoritmo. Es el **cálculo**.",
   "**4.3 Tipos y modelos** (Act. 29 y 30): refuerzo positivo y negativo, procesos de Markov, Q-learning y análisis de la recompensa. Es el **análisis y diseño**.",
   "Las flechas indican el orden recomendado: primero se aprende el lenguaje, después a calcular y por último a analizar y diseñar."],
  "Toda la unidad se resume en un ciclo: estado → acción → recompensa → nuevo estado → aprender.")
S("Introducción · Cómo aprende un agente: el algoritmo Q-learning",
  "Un diagrama de flujo con los 7 pasos de Q-learning y la flecha que repite el ciclo.",
  ["**1. Inicializar:** la tabla Q(s,a) empieza en 0 porque el agente no sabe nada.",
   "**2. Observar el estado** s en el que está.",
   "**3. Elegir la acción** con ε-greedy: al azar con probabilidad ε (explorar) o la de mayor Q con 1−ε (explotar).",
   "**4. Ejecutar:** el entorno responde con la recompensa r y el nuevo estado s'.",
   "**5. Calcular el objetivo** y = r + γ·max Q(s',·): lo que ahora se cree que vale esa acción.",
   "**6. Actualizar Q** acercándola al objetivo: Q ← Q + α·(y − Q).",
   "**7. Avanzar** (s ← s'). Si el episodio no terminó, se vuelve a decidir; si terminó, empieza otro."],
  "Repetir este ciclo miles de veces hace que Q se acerque a Q* y que la política greedy sea la óptima.")
S("Introducción · Fórmulas esenciales (1/2)",
  "Cuatro tarjetas: retorno descontado, valor de un estado, valor de una acción y ecuación de Bellman. Cada una trae fórmula, significado y ejemplo numérico.",
  ["**Retorno G:** suma de recompensas futuras, cada una multiplicada por γᵏ. Con r = (1, 1, 10) y γ = 0.9: 1 + 0.9·1 + 0.81·10 = **10.0**.",
   "**V(s):** retorno esperado empezando en s y siguiendo la política. Con «siempre Quedarse» y r = 3: V = 3/(1−0.9) = **30**.",
   "**Q(s,a):** igual que V pero fijando primero la acción; sirve para comparar acciones. Q*(S3,Izq) = 31.05 supera a Q*(S3,Quedarse) = 30.95.",
   "**Bellman óptima:** valor de (s,a) = recompensa inmediata + valor descontado de lo mejor que sigue. Q*(S3,Izq) = 5 + 0.9·28.95 = **31.05**."],
  "Distingue siempre la recompensa (lo inmediato) del valor (lo esperado a futuro): Bellman los conecta.")
S("Introducción · Fórmulas esenciales (2/2)",
  "Cuatro tarjetas más: actualización de Q-learning, error TD, política codiciosa con ε-greedy y recompensa constante con MDP.",
  ["**Actualización de Q-learning:** Q ← Q + α·[r + γ·max Q(s',·) − Q]. Es equivalente a un promedio ponderado (1−α)·Q + α·objetivo. Ejemplo: 0 + 0.5·(5 + 0.9·0.5 − 0) = **2.725**.",
   "**Error TD δ:** diferencia entre el objetivo y la estimación actual (5.45 en el ejemplo). Si δ = 0 la tabla ya cumple Bellman.",
   "**Política:** π*(s) = argmax Q*(s,a) elige la mejor acción; con ε-greedy la mejor acción tiene probabilidad 1−ε+ε/|A| y cada otra ε/|A| (con ε = 0.2 y 3 acciones: 86.7 % y 6.7 %).",
   "**Serie geométrica y MDP:** con recompensa constante V = r/(1−γ) (3 → 30, 6 → 60). El MDP es la tupla (S, A, P, R, γ) con propiedad de Markov."],
  "La actualización de Q es una corrección parcial: acerca el valor al objetivo, no lo reemplaza.")
S("Introducción · Los tres parámetros: γ, α y ε",
  "Tres gráficas, una por parámetro, que muestran cómo cambian el aprendizaje.",
  ["**γ (descuento):** curvas del peso γᵏ. Con γ = 0.5 el futuro se desvanece enseguida; con 0.99 casi no pierde peso. Con 0.9, una recompensa a 10 pasos vale 0.35.",
   "**α (tasa de aprendizaje):** cómo Q se acerca al objetivo 5.45 tras repetir la actualización. α = 1 llega de inmediato (pero es inestable), α = 0.5 converge rápido y α = 0.1 es lento y suave.",
   "**ε (exploración):** barras con la probabilidad de elegir la mejor acción (33 %, 67 %, 87 %, 97 % para ε = 1, 0.5, 0.2, 0.05). Se empieza explorando mucho y se reduce."],
  "Los tres parámetros controlan el equilibrio entre rapidez, estabilidad, visión de futuro y exploración.")
S("Introducción · Glosario de símbolos",
  "Una tabla con cada símbolo, cómo se lee, su significado y valores típicos.",
  ["Sirve como **referencia rápida** mientras se leen las fórmulas: s, a, r, π, G, γ, α, ε, V, Q, Q*, δ y P(s'|s,a).",
   "La columna «Valores típicos» da ejemplos concretos de este trabajo (por ejemplo V*(S2) = 28.95 o δ = 5.45) para que cada símbolo tenga un número asociado."],
  "Si aparece un símbolo que no se recuerda, esta diapositiva lo resuelve.")
S("Introducción · Gráfico explicativo: anatomía de una actualización de Q-learning",
  "Un gráfico en tres partes que desglosa el paso 4 de la Actividad 27 con números reales.",
  ["**A) La experiencia:** el agente está en S3, hace Izquierda, recibe r = +5 y llega a S2. La tabla dice que Q(S3,Izq) = 0 y que max Q(S2,·) = 0.5.",
   "**B) La fórmula pieza por pieza:** cada símbolo aparece con su color y su valor (α = 0.5, r = 5, γ = 0.9, max = 0.5). El **objetivo** vale 5 + 0.9·0.5 = 5.45 y el **error TD** 5.45 − 0 = 5.45.",
   "**C) El resultado:** en una recta numérica, Q pasa de 0 a **2.725**: avanza α·δ = 0.5·5.45, la mitad del camino hacia el objetivo. El resto se corrige en visitas futuras."],
  "Es la diapositiva clave de la introducción: muestra, con un caso real, qué hace cada símbolo de la fórmula.")
S("1. Teoría general paso a paso",
  "Diez pasos numerados con las ideas centrales del aprendizaje por refuerzo.",
  ["Pasos 1–3: qué es el RL, el ciclo agente–entorno y los seis elementos (medio ambiente, estado, acción, recompensa, política y valor).",
   "Pasos 4–6: retorno con descuento γ, modelo de Markov (MDP) y ecuación de Bellman.",
   "Pasos 7–8: la regla de Q-learning con su error TD y la estrategia ε-greedy.",
   "Pasos 9–10: familias de algoritmos (valores, políticas, actor-crítico, modelos) y tipos de refuerzo (positivo, negativo y castigo)."],
  "Es el resumen teórico en el que se apoyan las seis actividades.", "Teoría general")
S("Esquema de la teoría",
  "Un esquema visual con el ciclo agente–entorno, los elementos, las fórmulas, las familias de algoritmos y los tipos de refuerzo.",
  ["A la izquierda, el **ciclo**: el agente envía una acción; el entorno devuelve estado y recompensa.",
   "En el centro, los **elementos** y las fórmulas de retorno, Bellman y Q-learning.",
   "A la derecha, las **familias de algoritmos** y los tipos de refuerzo positivo y negativo."],
  "Es una vista de conjunto para repasar la teoría de un vistazo.")
S("2. Actividad 25 · Teoría: cómo reconocer cada elemento",
  "El objetivo de la actividad, una tabla para identificar los elementos del RL y varias ideas de apoyo.",
  ["La tabla da, para cada elemento (agente, entorno, estado, acción, recompensa, política y valor), una **pregunta guía**, una definición y el **error más frecuente**.",
   "**Recompensa vs. valor:** la recompensa es lo inmediato; el valor, lo que se espera acumular. En ajedrez, capturar una pieza da recompensa pero puede dejar un estado de valor bajo.",
   "**Tipos de recompensa:** dispersa (solo al final) y densa (guía cada paso, con el riesgo de *reward hacking*).",
   "Después aparece la receta «Cómo se resuelve» en cinco pasos."],
  "Con la pregunta guía correcta, identificar los elementos en cualquier ejemplo es mecánico.", "Actividad 25")
S("Actividad 25 · Resultado: tabla de los tres ejemplos",
  "Los pasos finales de la receta y la tabla con los seis elementos para el coche autónomo, el videojuego y el recomendador.",
  ["**Coche autónomo:** el agente es el sistema de conducción; el entorno, el estacionamiento; las recompensas, +100 al estacionar, −50 por chocar y −1 por segundo.",
   "**Videojuego:** el agente es el personaje; las recompensas son +1 por moneda, +50 por la meta y −10 por tocar un enemigo.",
   "**Recomendador:** el agente es el algoritmo; el entorno, usuarios y catálogo; la recompensa, +1 si ve la película y −1 si la ignora.",
   "Cada ejemplo distingue política (cómo decide) y valor (qué tan buena es una situación a largo plazo)."],
  "Es el entregable principal de la actividad: la tabla completa con los seis elementos en los tres casos.")
S("Actividad 25 · Ciclo agente–entorno y papel de la recompensa",
  "Un gráfico del ciclo de cada ejemplo y los párrafos que explican el papel de la recompensa.",
  ["El gráfico dibuja, por ejemplo, el agente arriba y el entorno abajo, con la **acción** bajando y el **estado y la recompensa** subiendo.",
   "Cada párrafo explica la función de la recompensa: en el coche premia el éxito y castiga choques y tardanza; en el videojuego es la única guía del personaje; en el recomendador mide si la recomendación fue buena y puede incluir el largo plazo."],
  "La recompensa es la señal que define qué significa «hacerlo bien» en cada problema.")
S("Actividad 25 · Código: la recompensa como función",
  "Una función en Python que calcula la recompensa del coche y su salida.",
  ["La función parte de −1 por cada segundo, resta 50 si choca y suma 100 si estaciona.",
   "Salida: **maniobrando = −1**, **chocó = −51** y **estacionó bien = +99**.",
   "Demuestra que la recompensa es un **número** que se puede programar."],
  "Traducir el objetivo a una función numérica es el primer paso para entrenar a un agente.")
S("3. Actividad 26 · Teoría: cómo se diseña un problema de RL",
  "Los pasos para diseñar un entorno de RL, cómo medir el aprendizaje y el inicio del diseño de AspiBot.",
  ["Diseñar un entorno es **formular un MDP**: objetivo, estado, acciones, recompensa, fin de episodio, política de exploración y evidencia de aprendizaje.",
   "Ejemplo de tamaño de estado: 25 celdas × 3 niveles de batería × 2 (sucio/limpio) = 150 estados, que todavía caben en una tabla.",
   "Las magnitudes importan: si chocar costara menos que dar un paso, el agente no aprendería a evitar choques.",
   "Para saber si aprendió: retorno que sube y se estabiliza, pasos que bajan, política greedy sin choques y comparación con el óptimo (+5 en 6 movimientos)."],
  "Antes de entrenar hay que decidir cómo se va a comprobar que el agente aprendió.", "Actividad 26")
S("Actividad 26 · Diseño de AspiBot y simulación",
  "El resto de la tabla de diseño y el código de la simulación con su salida.",
  ["La tabla incluye acciones (cuatro movimientos y aspirar), recompensas (+10 limpiar, −1 por paso, −3 por choque, −20 por quedarse sin batería), política ε-greedy y la evidencia de aprendizaje.",
   "El código entrena a AspiBot 400 episodios con α = 0.5 y γ = 0.95, e imprime el retorno medio al inicio y al final, los pasos y la ruta aprendida."],
  "La simulación convierte el diseño en una prueba: se ve si el agente realmente mejora.")
S("Actividad 26 · Gráficos de aprendizaje y conclusión",
  "Tres gráficos (retorno, pasos y ruta) y la conclusión de la actividad.",
  ["**Retorno por episodio:** pasa de ≈ −82 en los primeros 20 episodios a ≈ +4.5 en los últimos 20, cerca del máximo posible (+5).",
   "**Pasos hasta limpiar:** bajan hasta unos 6, la ruta más corta.",
   "**Ruta aprendida:** la política greedy rodea los obstáculos y llega a la suciedad en 6 movimientos.",
   "La conclusión resume los criterios para decir que el agente aprendió."],
  "Las curvas son la evidencia: sin ellas, «aprendió» sería solo una afirmación.")
S("4. Actividad 27 · Teoría: de Bellman a Q-learning",
  "Los siete pasos teóricos de Q-learning, un ejemplo anotado y la receta de resolución.",
  ["**Tabla Q:** una estimación por pareja estado–acción, iniciada en 0.",
   "**Bellman:** Q(s,a) = r + γ·max Q(s',a'). Mientras Q sea inexacta, la diferencia es el **error TD**.",
   "**Corrección parcial:** Q ← Q + α·δ, un promedio ponderado entre lo viejo y lo nuevo.",
   "**Propagación hacia atrás:** los estados cercanos a una recompensa ganan valor primero y luego lo heredan los anteriores.",
   "**Ejemplo anotado (paso 4):** objetivo 5 + 0.9·0.5 = 5.45; error 5.45; Q = 0 + 0.5·5.45 = **2.725**."],
  "Q-learning corrige poco a poco una tabla hasta que cumple la ecuación de Bellman.", "Actividad 27")
S("Actividad 27 · Entorno y código de la actualización",
  "La tabla de reglas del entorno (3 estados × 3 acciones) y el código que aplica Q-learning.",
  ["La tabla indica, para cada (estado, acción), el siguiente estado y la recompensa; por ejemplo (S3, Izquierda) → S2 con +5 y (S3, Quedarse) → S3 con +3.",
   "La función actualizar calcula el mejor valor siguiente, el objetivo, el error TD y aplica la corrección con α.",
   "La trayectoria tiene 6 experiencias que recorren los tres estados y repiten algunos pares."],
  "El código es una traducción literal de la receta de siete pasos.")
S("Actividad 27 · Salida y tabla de cálculos",
  "La salida del código y la tabla con los 6 pasos.",
  ["Los seis nuevos valores de Q son: 0, 0.5, 1.5, **2.725**, **1.9763** y **4.7518**.",
   "En el paso 5 Q(S2,Derecha) sube de 0.5 a 1.9763 porque el estado siguiente ya tiene valor alto: así se **propaga el aprendizaje hacia atrás**.",
   "La tabla Q final deja Q(S3,Izq) = 4.7518 como el valor más alto."],
  "Cada fila de la tabla es una aplicación completa de la fórmula con sus valores.")
S("Actividad 27 · Gráfico de evolución y conclusión",
  "Líneas con la evolución de cuatro valores Q, la tabla Q final como mapa de calor y la conclusión.",
  ["Las líneas muestran **cuándo** se actualiza cada Q y cuánto sube; Q(S3,Izq) llega a ≈ 4.75.",
   "El mapa de calor destaca los valores más altos en azul oscuro.",
   "La conclusión explica que el Q más alto resume la **recompensa inmediata más el valor futuro descontado**, y que con más episodios Q converge a Q*."],
  "Una iteración solo da el primer tramo del aprendizaje; los valores finales (≈ 31) llegan con muchos episodios.")
S("5. Actividad 28 · Teoría: mapa de algoritmos",
  "Una tabla con seis algoritmos, los criterios de decisión y el inicio de la receta de resolución.",
  ["La tabla compara **Q-learning tabular, SARSA, DQN, REINFORCE/PPO, actor-crítico y Dyna-Q**: cómo funcionan, su fortaleza y su limitación.",
   "Los criterios son tres preguntas: ¿cuántos estados?, ¿las acciones son discretas o continuas? y ¿se puede simular el entorno?",
   "DQN usa una red neuronal para estimar Q, con memoria de repetición y red objetivo."],
  "Elegir algoritmo es contestar preguntas sobre el problema, no escoger el más famoso.", "Actividad 28")
S("Actividad 28 · El agente de semáforos y la comparación",
  "El resto de la receta, la tabla del agente y la comparación de algoritmos aplicada.",
  ["El agente es un **controlador de semáforos**; sus acciones son mantener o cambiar la fase y su recompensa penaliza la espera total y los cambios excesivos.",
   "El algoritmo elegido es **DQN** porque el espacio de estados es enorme y las acciones son pocas y discretas.",
   "La comparación explica por qué se descartan Q-learning tabular, PPO y Dyna-Q para este caso."],
  "La justificación se apoya en las características del problema.")
S("Actividad 28 · Código: tamaño de la tabla y objetivo de DQN",
  "Código con tres cálculos y su salida.",
  ["**Tamaño:** 31⁴ · 10⁴ · 4 ≈ 36 940 840 000 estados × 2 acciones ≈ 73 881 680 000 celdas Q: inviable.",
   "**Recompensa:** con espera 12 y cambio de fase da r = −17.",
   "**Objetivo de DQN:** y = r + γ·max Q = −17 + 0.9·(−9.8) = **−25.82**, el valor al que se acerca la red."],
  "DQN conserva la lógica de Q-learning y cambia la tabla por una red.")
S("Actividad 28 · Gráficos: la tabla no escala y árbol de decisión",
  "Una curva del crecimiento de la tabla Q y un árbol de decisión para elegir algoritmo.",
  ["La curva (escala logarítmica) supera el límite práctico (≈ 10⁹ celdas) hacia los 10 autos por carril, mucho antes de los 30 de un cruce real.",
   "El árbol: pocos estados → Q-learning tabular; acciones continuas → PPO o SAC; se puede simular → Dyna-Q; en otro caso → **DQN**."],
  "El árbol resume el criterio de decisión de la actividad.")
S("6. Actividad 29 · Teoría: valores óptimos y sensibilidad",
  "Siete pasos teóricos sobre Q* y la receta de la actividad.",
  ["La política óptima es π*(s) = argmax Q*(s,a), y Q* se obtiene repitiendo Bellman hasta converger.",
   "**Quedarse siempre** vale r/(1−γ); el **ciclo S3→S2→S3** vale 5.9/(1−0.81) = 31.05.",
   "**Umbral:** quedarse gana cuando r ≥ 5.9/1.9 ≈ **3.105**. Con +3 gana el ciclo por poco; con +6 gana quedarse.",
   "Lección: cuando dos opciones están casi empatadas, un cambio pequeño de recompensa invierte la política."],
  "Una recompensa mal calibrada puede cambiar el comportamiento del agente.", "Actividad 29")
S("Actividad 29 · Código",
  "Código que compara el entorno original (+3) con el modificado (+6).",
  ["Tres bloques: **una iteración** con la misma trayectoria; los **valores óptimos Q\\*** de ambos entornos; y una comprobación con **Q-learning entrenado**."],
  "Se compara con el mismo procedimiento para que la única diferencia sea la recompensa.")
S("Actividad 29 · Salida: tablas Q y políticas",
  "Las tablas Q tras una iteración y las tablas Q* con las políticas.",
  ["Tras una iteración, Q(S3,Quedarse) pasa de 1.5 a 3.0 con la nueva recompensa.",
   "Q* original: S3 → Izquierda (31.05 vs 30.95). Q* nueva: S3 → **Quedarse** (60 vs 54.5).",
   "Q-learning entrenado llega a la misma política que la solución exacta."],
  "La política cambia solo en S3, justo donde se modificó la recompensa.")
S("Actividad 29 · Gráficos: políticas y umbral",
  "Dos mapas de calor de Q* (original y modificada) y una curva de Q*(S3,·) según la recompensa.",
  ["Los recuadros marcan la mejor acción por estado; en S3 el recuadro pasa de «Izquierda» a «Quedarse».",
   "La curva muestra el **cruce** en ≈ 3.105; las líneas punteadas marcan +3 (justo antes) y +6 (claramente después)."],
  "El gráfico explica visualmente por qué +3 y +6 llevan a políticas distintas.")
S("Actividad 29 · Análisis",
  "Las respuestas a las dos preguntas de la actividad.",
  ["**¿Qué cambió?** En S3 la acción óptima pasó de Izquierda a Quedarse y todos los Q suben.",
   "**¿Por qué?** Quedarse produce V(S3) = 6/(1−0.9) = 60, mientras que el ciclo rinde ≈ 54.5; al duplicar la recompensa, quedarse supera al ciclo."],
  "El valor de una acción es la recompensa inmediata más el valor futuro descontado.")
S("7. Actividad 30 · Teoría: tipos de refuerzo",
  "Una tabla de condicionamiento operante y ideas sobre cómo se adapta el agente.",
  ["La tabla 2×2: refuerzo **positivo** (añadir un premio), **negativo** (retirar algo molesto), castigo **positivo** (añadir algo desagradable) y castigo **negativo** (quitar algo bueno).",
   "«Negativo» no es castigo: el refuerzo negativo **aumenta** la conducta.",
   "En RL todo se codifica como un número r; r > 0 favorece y r < 0 desalienta.",
   "La adaptación ocurre porque ε baja, Q se actualiza y la política converge."],
  "Conviene distinguir conceptualmente el tipo de consecuencia aunque en el código sea solo un signo.", "Actividad 30")
S("Actividad 30 · Diseño de SkyBot",
  "La descripción de SkyBot, la tabla de diseño y el tipo de refuerzo.",
  ["SkyBot es un dron repartidor que aprende rutas cortas, evita obstáculos y recarga a tiempo.",
   "La tabla cubre agente, entorno, estados, acciones, recompensas, política y evidencia de aprendizaje.",
   "Usa **ambos refuerzos**: positivo (+20 al entregar), negativo (desaparece la batería baja al cargar) y penalizaciones por chocar o caer."],
  "Cada columna de la tabla corresponde a un elemento visto en la unidad.")
S("Actividad 30 · Código de recompensas y política ε-greedy",
  "Dos funciones (recompensa y elección de acción) y su salida.",
  ["Recompensas netas: **entrega = +19**, **carga a tiempo = +4**, **choque = −11**, **caída = −31** (incluyen el −1 por paso).",
   "La elección de acción con ε = 1, 0.5 y 0.05 muestra que la mejor acción se elige cada vez más (cerca de 33 %, 67 % y 97 %)."],
  "Al bajar ε el agente explota lo aprendido: así se adapta con el tiempo.")
S("Actividad 30 · Gráficos de SkyBot",
  "Una gráfica de barras con la recompensa de cada evento y una curva del efecto de ε.",
  ["Las barras verdes son premios; las guindas, castigos. La caída es el evento más costoso.",
   "La curva teórica (1−ε)+ε/3 coincide con los puntos de la simulación."],
  "Cierra el documento mostrando recompensas y exploración en un solo vistazo.")

assert len(D) == 34, len(D)

# ----------------------------------------------------------------- miniaturas
tmp = Path(tempfile.mkdtemp())
src = fitz.open(str(FUENTE))
assert len(src) == len(D), (len(src), len(D))
for i, pg in enumerate(src, 1):
    pg.get_pixmap(dpi=62).save(str(tmp / f"d{i}.png"))

doc = SimpleDocTemplate(str(SALIDA), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                        bottomMargin=1.5 * cm, title="Guía explicativa de las diapositivas", author=NOMBRE)
W = doc.width
E = [Spacer(1, 2 * cm), Paragraph("Guía explicativa de las diapositivas", sT),
     Paragraph("Aprendizaje por Refuerzo · Unidad 4", sS),
     Paragraph(f"Aprendizaje Automático · Maestría en Inteligencia Artificial · TecNM Virtual · Alumno(a): {NOMBRE}", sS),
     Spacer(1, 10),
     Paragraph("Este documento acompaña al PDF de 34 diapositivas. Para cada una indica <b>qué muestra</b>, "
               "<b>cómo se explica paso a paso</b> y cuál es su <b>idea clave</b>, de modo que sirva para estudiar o para exponer.", sB),
     Paragraph(f'<link href="{COLAB}" color="#1155CC"><u>▶ Abrir el cuaderno en Google Colab</u></link><br/>'
               f'<link href="{GH}" color="#1155CC"><u>Repositorio en GitHub</u></link>', sB), Spacer(1, 10)]

indice = [["Diap.", "Tema"]]
for i, (t, *_r) in enumerate(D, 1):
    indice.append([str(i), t])
ti = Table([[Paragraph(md(c), ParagraphStyle("i", fontName="DVB" if k == 0 else "DV", fontSize=7.8, leading=10,
                                             textColor=colors.white if r == 0 else colors.black))
             for k, c in enumerate(f)] for r, f in enumerate(indice)], colWidths=[1.3 * cm, W - 1.3 * cm], repeatRows=1)
ti.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), AZ), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#BBB")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F7")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
E += [ti, PageBreak()]

for i, (titulo, muestra, expl, clave, sec) in enumerate(D, 1):
    if sec:
        E.append(Paragraph(sec, sSec))
    cab = Table([[Paragraph(f"Diapositiva {i} · {md(titulo)}", sN)]], colWidths=[W])
    cab.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), AZ), ("TOPPADDING", (0, 0), (-1, -1), 4),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    th = Image(str(tmp / f"d{i}.png"), width=7.4 * cm, height=7.4 * cm * 540 / 960)
    th.hAlign = "LEFT"
    txt = [Paragraph("Qué muestra", sH), Paragraph(md(muestra), sB), Paragraph("Explicación paso a paso", sH)]
    txt += [Paragraph("•&nbsp;&nbsp;" + md(x), sL) for x in expl]
    txt += [Paragraph("Idea clave", sH), Paragraph(md(clave), sB)]
    cuerpo = Table([[th, txt]], colWidths=[7.8 * cm, W - 7.8 * cm])
    cuerpo.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#BBB")), ("LEFTPADDING", (1, 0), (1, 0), 6)]))
    E.append(KeepTogether([cab, cuerpo, Spacer(1, 9)]))

def pie(c, d):
    c.saveState(); c.setFont("DV", 7.5); c.setFillColor(colors.HexColor("#777"))
    c.drawString(1.6 * cm, 0.9 * cm, "Guía explicativa · Aprendizaje por Refuerzo · Unidad 4")
    c.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"Página {d.page}"); c.restoreState()

doc.build(E, onFirstPage=pie, onLaterPages=pie)
print("OK", SALIDA)
