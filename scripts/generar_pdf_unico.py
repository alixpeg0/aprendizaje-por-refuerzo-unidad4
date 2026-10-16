"""Genera un único PDF vertical con teoría, pasos y código de las actividades 25-30."""
import sys, io, re, contextlib
from pathlib import Path
raiz = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(raiz))

import matplotlib
matplotlib.use("Agg")
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, Preformatted, KeepTogether)
from PIL import Image as PI

import rl_core
from rl_core import *
from contenido import *
import contenido

ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for n, f in [("DV", "DejaVuSans"), ("DVB", "DejaVuSans-Bold"), ("DM", "DejaVuSansMono")]:
    pdfmetrics.registerFont(TTFont(n, str(ttf / f"{f}.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

NOMBRE = "Nombre Apellido"
AZ, GU = colors.HexColor(AZUL), colors.HexColor(GUINDA)
sT = ParagraphStyle("t", fontName="DVB", fontSize=22, textColor=AZ, leading=27, spaceAfter=6)
sA = ParagraphStyle("a", fontName="DVB", fontSize=15, textColor=AZ, leading=19, spaceBefore=4, spaceAfter=6)
sH = ParagraphStyle("h", fontName="DVB", fontSize=11, textColor=GU, spaceBefore=9, spaceAfter=3)
sB = ParagraphStyle("b", fontName="DV", fontSize=9, leading=13, alignment=4, spaceAfter=4)
sL = ParagraphStyle("l", parent=sB, leftIndent=14, firstLineIndent=-14, alignment=0, spaceAfter=3)
sC = ParagraphStyle("c", fontName="DV", fontSize=7.5, leading=10)
sCH = ParagraphStyle("ch", fontName="DVB", fontSize=7.5, leading=10, textColor=colors.white)
sM = ParagraphStyle("m", fontName="DV", fontSize=8, leading=11, textColor=colors.HexColor("#444"))
sCode = ParagraphStyle("code", fontName="DM", fontSize=6.9, leading=8.6, backColor=colors.HexColor("#F3F4F7"),
                       borderColor=colors.HexColor("#CCC"), borderWidth=0.5, borderPadding=4, leftIndent=4,
                       rightIndent=4, spaceBefore=4, spaceAfter=6)
sOut = ParagraphStyle("out", parent=sCode, backColor=colors.HexColor("#FFF8E7"), borderColor=colors.HexColor("#D9B86A"))

doc = SimpleDocTemplate(str(raiz / "pdfs" / "Unidad4_Aprendizaje_por_Refuerzo_Completo.pdf"), pagesize=A4,
                        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="Unidad 4 - Aprendizaje por Refuerzo (paso a paso con código)", author=NOMBRE)
W = doc.width
E = []


def md(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r'<font name="DM" size="8">\1</font>', t)


def P(t, s=sB): E.append(Paragraph(md(t), s))
def H(t): E.append(Paragraph(md(t), sH))
def pasos(lista):
    for i, t in enumerate(lista, 1):
        E.append(Paragraph(f"<b>Paso {i}.</b> {md(t)}", sL))


def pre(texto, estilo):
    E.append(Preformatted(texto.rstrip("\n"), estilo))


def codigo(src, mostrar_salida=True, ejecutar=True, ns=None, previo=0):
    """Muestra el código y su salida REAL (se ejecuta aquí)."""
    src = src.strip("\n")
    n0 = len(E) - previo
    H("Código")
    pre(src, sCode)
    if ejecutar and mostrar_salida:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(src, ns if ns is not None else NS)
        H("Salida al ejecutarlo")
        pre(buf.getvalue(), sOut)
        E[n0:] = [KeepTogether(E[n0:])]
    elif ejecutar:
        exec(src, ns if ns is not None else NS)


def tabla(cab, filas, anchos, negrita=True):
    d = [[Paragraph(md(str(c)), sCH) for c in cab]]
    for f in filas:
        d.append([Paragraph(("<b>%s</b>" if k == 0 and negrita else "%s") % md(str(c)), sC) for k, c in enumerate(f)])
    t = Table(d, colWidths=anchos, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), AZ), ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#999")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F7")])]))
    E.append(t)
    E.append(Spacer(1, 6))


def imagen(ruta, ancho):
    w, h = PI.open(ruta).size
    E.append(Image(str(ruta), width=ancho, height=ancho * h / w))


NS = {}
exec("import numpy as np, pandas as pd\nfrom rl_core import *\nfrom contenido import *", NS)

# =============================================================== PORTADA
E += [Spacer(1, 3 * cm), Paragraph("Aprendizaje por Refuerzo", sT),
      Paragraph("Unidad 4 · Cuaderno de trabajo resuelto, paso a paso y con código", sA),
      Paragraph(f"Aprendizaje Automático · Maestría en Inteligencia Artificial · TecNM Virtual<br/>Alumno(a): {NOMBRE}", sM),
      Spacer(1, 1 * cm)]
tabla(["Sección", "Contenido"], [
    ["1", "Teoría paso a paso y esquema"], ["2", "Actividad 25 · Elementos del RL en tres ejemplos"],
    ["3", "Actividad 26 · Entorno propio (AspiBot)"], ["4", "Actividad 27 · Una iteración de Q-learning"],
    ["5", "Actividad 28 · Agente y elección de algoritmo"], ["6", "Actividad 29 · Nueva tabla Q (S3: +3 → +6)"],
    ["7", "Actividad 30 · Agente propio (SkyBot)"]], [W * .12, W * .88], False)
P("El código completo (ejecutable en Google Colab) está en: https://github.com/alixpeg0/aprendizaje-por-refuerzo-unidad4. "
  "Todas las salidas de este documento se obtuvieron ejecutando el código mostrado.", sM)
E.append(PageBreak())

# =============================================================== 1 TEORIA
E.append(Paragraph("1. Teoría paso a paso", sA))
teo = [
    ("¿Qué es el aprendizaje por refuerzo?",
     "Un **agente** aprende a decidir **interactuando** con un entorno. No recibe ejemplos etiquetados: recibe "
     "**recompensas** y descubre por ensayo y error qué acciones convienen. Su meta es maximizar la recompensa "
     "**total acumulada**, no solo la inmediata."),
    ("Ciclo agente–entorno",
     "En cada instante: el agente observa el estado `s`, elige una acción `a`, el entorno responde con una "
     "recompensa `r` y un nuevo estado `s'`, y el agente aprende de la experiencia `(s, a, r, s')`."),
    ("Seis elementos básicos",
     "**Medio ambiente** (lo externo), **estado** (situación actual), **acción** (lo que puede hacer), "
     "**recompensa** (número que dice qué tan bueno fue lo ocurrido), **política** π(s) (regla que elige acción por "
     "estado) y **valor** V(s) o Q(s,a) (recompensa futura esperada). La recompensa es lo de *ahora*; el valor, "
     "lo esperado *de aquí en adelante*."),
    ("Retorno y descuento γ",
     "El agente maximiza `G = r₁ + γ·r₂ + γ²·r₃ + …` con 0 ≤ γ ≤ 1. Con γ cercano a 0 es miope; cercano a 1, "
     "previsor. Con γ = 0.9, una recompensa dentro de 2 pasos vale 0.81 de su valor."),
    ("Procesos de decisión de Markov (MDP)",
     "Es el modelo formal: (S, A, P, R, γ). Propiedad de Markov: el futuro depende **solo del estado actual**."),
    ("Ecuación de Bellman",
     "El valor es recursivo: *recompensa inmediata + valor descontado de lo que sigue*. "
     "`Q*(s,a) = r + γ · max Q*(s',a')`."),
    ("Q-learning",
     "Sin conocer el entorno, estima Q con cada experiencia: `Q(s,a) ← Q(s,a) + α·[ r + γ·max Q(s',a') − Q(s,a) ]`. "
     "α es la tasa de aprendizaje; el corchete es el **error TD** (objetivo − estimación actual). Es *off-policy*."),
    ("Exploración vs. explotación (ε-greedy)",
     "Con probabilidad ε se elige una acción al azar (explorar) y con 1−ε la de mayor Q (explotar). Se suele "
     "empezar con ε alto y reducirlo."),
    ("Familias de algoritmos",
     "**Basados en valores** (Q-learning, SARSA, DQN: acciones discretas), **en políticas** (REINFORCE, PPO: acciones "
     "continuas), **actor-crítico** (A2C, SAC) y **basados en modelos** (Dyna-Q, MCTS: cuando se puede simular)."),
    ("Tipos de refuerzo",
     "**Positivo:** se añade un premio para que la conducta aumente. **Negativo:** se retira algo desagradable para que "
     "aumente. **Castigo/penalización:** se añade algo desagradable para que disminuya (en RL se escribe como r < 0)."),
]
for i, (t, x) in enumerate(teo, 1):
    E.append(Paragraph(f"<b>Paso {i}. {md(t)}.</b> {md(x)}", sL))
H("Receta para aplicar Q-learning a mano")
pasos(["Identifica `s`, `a`, `r` y `s'` de la experiencia.", "Lee `Q(s,a)` actual en la tabla.",
       "Lee `max Q(s',·)`: el mejor valor del estado siguiente.", "Objetivo = `r + γ·max Q(s',·)`.",
       "Error TD = objetivo − `Q(s,a)`.", "Nuevo `Q(s,a)` = `Q(s,a) + α·error`.", "Repite con la siguiente experiencia."])
_n = len(E)
H("Esquema de la teoría")
imagen(raiz / "imagenes" / "esquema_teoria.png", W)
E[_n:] = [KeepTogether(E[_n:])]
E.append(PageBreak())

# =============================================================== ACT 25
E.append(Paragraph("2. Actividad 25 · Detecta los elementos del aprendizaje por refuerzo", sA))
P("**Objetivo:** identificar los conceptos básicos del RL en situaciones reales.")
H("Cómo se resuelve")
pasos(["Para cada ejemplo, pregunta **quién aprende** (agente) y **con qué interactúa** (medio ambiente).",
       "Describe **qué observa** el agente (estado) y **qué puede hacer** (acciones).",
       "Define **qué señal** le dice si lo hizo bien o mal (recompensa), con valores concretos.",
       "Redacta la política (cómo decide) y el valor (qué tan buena es una situación a futuro).",
       "Explica el papel de la recompensa en cada caso."])
H("Resultado")
tabla(["Elemento", "a) Coche", "b) Videojuego", "c) Recomendador"], [[f[0], f[1], f[2], f[3]] for f in A25_FILAS],
      [W * .15, W * .29, W * .28, W * .28])
H("Papel de la recompensa")
for t, x in A25_RECOMPENSA:
    P(f"**{t}.** {x}")
H("Código: la recompensa del coche como función")
P("Para comprobar que la recompensa es una **señal numérica** que premia o castiga, se programa la del ejemplo a):")
codigo('''
def recompensa_coche(estacionado, choco):
    r = -1                 # cada segundo cuesta un punto (hay que ser rápido)
    if choco:
        r -= 50            # castigo por chocar
    if estacionado:
        r += 100           # premio por terminar bien
    return r

for situacion, args in {"maniobrando": (False, False), "chocó": (False, True),
                        "estacionó bien": (True, False)}.items():
    print(f"{situacion:15s} -> r = {recompensa_coche(*args):4d}")
''', previo=2)
E.append(PageBreak())

# =============================================================== ACT 26
E.append(Paragraph("3. Actividad 26 · Diseña tu propio entorno de aprendizaje por refuerzo", sA))
P("**Objetivo:** aplicar los conceptos básicos de forma creativa.")
H("Cómo se resuelve")
pasos(["Elige un agente: aquí **AspiBot**, un robot aspiradora.", "Describe el **entorno**: rejilla 5×5 con muebles y una zona sucia.",
       "Enumera **estados** (celda, batería) y **acciones** (4 movimientos).",
       "Asigna **recompensas**: +10 limpiar, −1 por paso, −3 por choque.",
       "Define la **política**: ε-greedy sobre Q(s,a).",
       "Decide **cómo medir que aprendió**: retorno que sube, pasos que bajan y ruta corta sin choques."])
H("Descripción")
P(f"**{A26_TITULO}.** {A26_DESCRIPCION}")
tabla(["Elemento", "Diseño"], A26_FILAS, [W * .24, W * .76])
H("Código: simulación para comprobar el aprendizaje")
P("La rejilla se implementa en `rl_core.py` (clase `Rejilla`). El agente aplica la fórmula de Q-learning en cada paso.")
codigo('''
env = Rejilla()                                  # 5x5, base (0,0), suciedad (3,3)
Q, retornos, pasos = entrenar_rejilla(env, episodios=400, alpha=0.5, gamma=0.95)
print("Retorno medio, primeros 20 episodios:", round(retornos[:20].mean(), 1))
print("Retorno medio, últimos 20 episodios :", round(retornos[-20:].mean(), 1))
print("Pasos medios, últimos 20 episodios  :", round(pasos[-20:].mean(), 1))
print("Ruta aprendida (greedy):", ruta_greedy(env, Q))
''', previo=2)
imagen(raiz / "imagenes" / "act26_aprendizaje.png", W)
H("Conclusión")
P(A26_CONCLUSION)
E.append(PageBreak())

# =============================================================== ACT 27
E.append(Paragraph("4. Actividad 27 · Simula una iteración de Q-learning", sA))
P("**Objetivo:** aplicar la ecuación de Bellman para calcular Q(s,a). **Parámetros:** α = 0.5, γ = 0.9, Q = 0 al inicio.")
H("Cómo se resuelve")
pasos(["Define el entorno: 3 estados (S1, S2, S3) y 3 acciones (Izquierda, Derecha, Quedarse) con su recompensa y siguiente estado.",
       "Inicia la tabla Q en 0 (3×3).",
       "Elige una trayectoria de 6 experiencias que visite los tres estados y repita algún par (s,a).",
       "Para **cada** experiencia aplica la receta: objetivo = r + γ·max Q(s',·); error = objetivo − Q; Q nuevo = Q + α·error.",
       "Registra cada cálculo en una tabla y localiza el valor Q más alto.",
       "Interpreta el resultado."])
H("Entorno")
tabla(["Estado s", "Acción a", "Siguiente s'", "Recompensa r"], [list(map(str, f)) for f in tabla_entorno(construir_entorno(3)).values],
      [W / 4] * 4, False)
codigo('''
T = construir_entorno(r_s3_quedarse=3)       # T[(s, a)] = (s', r)
Q = q_vacia()                                # tabla 3x3 con ceros

def actualizar(Q, s, a, r, s2, alpha=0.5, gamma=0.9):
    mejor_siguiente = Q.loc[s2].max()                # max Q(s', a')
    objetivo = r + gamma * mejor_siguiente           # lo que ahora creemos que vale
    error_td = objetivo - Q.loc[s, a]                # cuánto nos equivocamos
    Q.loc[s, a] += alpha * error_td                  # corrección parcial (alpha)
    return mejor_siguiente, objetivo, error_td

trayectoria = [("S1","Derecha"), ("S2","Derecha"), ("S3","Quedarse"),
               ("S3","Izquierda"), ("S2","Derecha"), ("S3","Izquierda")]
for i, (s, a) in enumerate(trayectoria, 1):
    s2, r = T[(s, a)]
    q_ant = Q.loc[s, a]
    m, obj, err = actualizar(Q, s, a, r, s2)
    print(f"Paso {i}: Q({s},{a}) = {q_ant:.4f} + 0.5*[{r} + 0.9*{m:.4f} - {q_ant:.4f}] = {Q.loc[s, a]:.4f}")

print()
print(Q.round(4))
print("\\nQ más alto:", Q.stack().idxmax(), "=", round(Q.stack().max(), 4))
''')
H("Tabla de cálculos")
_, p27 = simular_trayectoria(construir_entorno(3))
f = lambda v: f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
tabla(["Paso", "s", "a", "r", "s'", "Q ant.", "max Q(s',·)", "Objetivo", "Error TD", "Q nuevo"],
      [[f(v) for v in fila] for fila in p27.drop(columns="Cálculo").itertuples(index=False)],
      [W * .07, W * .07, W * .15, W * .06, W * .07, W * .1, W * .14, W * .12, W * .11, W * .11], False)
H("Conclusión")
for x in A27_CONCLUSION:
    P(x)
E.append(PageBreak())

# =============================================================== ACT 28
E.append(Paragraph("5. Actividad 28 · Diseña un agente y elige su algoritmo", sA))
P("**Objetivo:** seleccionar un algoritmo adecuado al problema.")
H("Cómo se resuelve")
pasos(["Describe el agente y el entorno (aquí: un semáforo inteligente en un cruce).",
       "Lista las **acciones**: pocas y discretas (mantener / cambiar fase).",
       "Diseña la **recompensa**: penalizar la espera total de los autos y los cambios de fase excesivos.",
       "Estima el **tamaño del espacio de estados**: si es enorme, descarta Q-learning tabular.",
       "Compara las familias de algoritmos y elige la que encaja.",
       "Justifica la elección con las características del problema."])
tabla(["Apartado", "Descripción"], A28_FILAS, [W * .22, W * .78])
H("Comparación de algoritmos")
tabla(["Algoritmo", "Familia", "Ideal cuando…", "¿Para este agente?"], A28_COMPARATIVA, [W * .17, W * .2, W * .27, W * .36])
H("Código: por qué no basta una tabla Q, y cómo se calcula el objetivo de DQN")
P("Primero se muestra el tamaño de la tabla Q si se discretiza el cruce; después, el objetivo que DQN calcula "
  "(igual que Q-learning, pero con una red que estima Q).")
codigo('''
import numpy as np
# --- 1) ¿Cabe una tabla Q? 4 carriles con filas de 0..30 autos, 4 fases, espera en 10 niveles
n_estados = (31 ** 4) * 4 * 10
print(f"Estados posibles: {n_estados:,}  x 2 acciones = {n_estados*2:,} celdas Q")

# --- 2) Recompensa del controlador
def recompensa_semaforo(espera_total, cambio_de_fase, fila_vaciada):
    return -espera_total - (5 if cambio_de_fase else 0) + (10 if fila_vaciada else 0)
print("r =", recompensa_semaforo(espera_total=12, cambio_de_fase=True, fila_vaciada=False))

# --- 3) Objetivo de DQN: y = r + gamma * max_a' Q_objetivo(s', a')  (con una red, aquí simulada)
gamma = 0.9
q_red_objetivo_siguiente = np.array([-14.2, -9.8])   # salida de la red para [mantener, cambiar]
r = recompensa_semaforo(12, True, False)
y = r + gamma * q_red_objetivo_siguiente.max()
print("Objetivo DQN y =", round(y, 2), " (la red se entrena para acercar Q(s,a) a y)")
''', previo=2)
E.append(PageBreak())

# =============================================================== ACT 29
E.append(Paragraph("6. Actividad 29 · Calcula tu propia tabla Q", sA))
P("**Objetivo:** ver cómo cambiar una recompensa modifica los valores Q y la acción óptima.")
H("Cómo se resuelve")
pasos(["Parte del mismo entorno de la Actividad 27.", "Cambia la recompensa de (S3, Quedarse) de **+3 a +6**.",
       "Recalcula Q con la **misma trayectoria** (una iteración) y compara con la original.",
       "Calcula también los valores **óptimos Q\\*** repitiendo Bellman hasta converger, para ver la política final.",
       "Compara las políticas y explica qué cambió y por qué."])
codigo('''
T3, T6 = construir_entorno(3), construir_entorno(6)      # original y modificado

# (1) Una iteración con la misma trayectoria
Q_orig, _ = simular_trayectoria(T3)
Q_new,  _ = simular_trayectoria(T6)
print("Tras una iteración (original +3):"); print(Q_orig.round(4))
print("\\nTras una iteración (nueva +6):");   print(Q_new.round(4))

# (2) Valores óptimos Q*: repetir Q(s,a) = r + gamma * max Q(s',a') hasta converger
Qs3, Qs6 = q_optima(T3, gamma=0.9), q_optima(T6, gamma=0.9)
print("\\nQ* original (+3):"); print(Qs3.round(2))
print("\\nQ* nueva (+6):");   print(Qs6.round(2))
print("\\nPolítica original:", politica(Qs3).to_dict())
print("Política nueva   :", politica(Qs6).to_dict())

# (3) Comprobación con Q-learning entrenado por interacción
Qe = entrenar_q_learning(T6, episodios=500)
print("\\nPolítica aprendida por Q-learning:", politica(Qe).to_dict())
''')
imagen(raiz / "imagenes" / "act29_politicas.png", W * .95)
H("Análisis")
for x in A29_ANALISIS:
    P(x)
E.append(PageBreak())

# =============================================================== ACT 30
E.append(Paragraph("7. Actividad 30 · Diseña tu propio agente y aplica la lógica del aprendizaje por refuerzo", sA))
P("**Objetivo:** comprender estado, acción, recompensa, política y valor diseñando un entorno propio.")
H("Cómo se resuelve")
pasos(["Imagina un agente: un dron repartidor (**SkyBot**).", "Escribe su propósito en un párrafo.",
       "Completa la tabla: agente, entorno, estados, acciones, recompensas, política y evidencia de aprendizaje.",
       "Indica qué **tipo de refuerzo** usa (positivo, negativo o ambos) y por qué.",
       "Explica cómo **se adapta** con el tiempo (ε baja, Q se actualiza, la política converge)."])
H(A30_TITULO)
P("**Propósito.** " + A30_PROPOSITO)
tabla(["Elemento", "Diseño"], list(zip(A30_COLUMNAS, A30_FILA)), [W * .24, W * .76])
H("Tipo de refuerzo y adaptación")
P(A30_TIPO_Y_ADAPTACION)
H("Código: recompensas y política ε-greedy de SkyBot")
codigo('''
import random
def recompensa_dron(entrego, choco, cayo, cargo_con_bateria_baja):
    r = -1                                   # cada paso gasta batería
    r += 20 if entrego else 0                # refuerzo POSITIVO: premio añadido
    r += 5 if cargo_con_bateria_baja else 0  # refuerzo NEGATIVO: desaparece la batería baja
    r -= 10 if choco else 0                  # penalización (castigo)
    r -= 30 if cayo else 0                   # penalización fuerte
    return r

def elegir_accion(Q_estado, acciones, epsilon):
    if random.random() < epsilon:            # explorar
        return random.choice(acciones)
    return max(acciones, key=lambda a: Q_estado[a])   # explotar: mayor Q

print("Entrega ........:", recompensa_dron(True, False, False, False))
print("Choca ..........:", recompensa_dron(False, True, False, False))
print("Carga a tiempo .:", recompensa_dron(False, False, False, True))
print("Cae sin batería :", recompensa_dron(False, False, True, False))

random.seed(0)
Q_estado = {"subir": 1.0, "avanzar": 4.5, "volver_a_cargar": 2.0}
for eps in (1.0, 0.5, 0.05):
    elecciones = [elegir_accion(Q_estado, list(Q_estado), eps) for _ in range(1000)]
    print(f"epsilon={eps:4}: elige 'avanzar' (la mejor) el {elecciones.count('avanzar')/10:.1f}% de las veces")
''', previo=1)
P("Al bajar ε, SkyBot explota cada vez más lo aprendido: así se refleja que su comportamiento se adapta con el tiempo.")

# pie de página
def pie(c, d):
    c.saveState(); c.setFont("DV", 7.5); c.setFillColor(colors.HexColor("#777"))
    c.drawString(1.8 * cm, 1 * cm, "Aprendizaje Automático · Unidad 4 · Aprendizaje por Refuerzo")
    c.drawRightString(A4[0] - 1.8 * cm, 1 * cm, f"Página {d.page}"); c.restoreState()

doc.build(E, onFirstPage=pie, onLaterPages=pie)
print("OK", doc.filename)
