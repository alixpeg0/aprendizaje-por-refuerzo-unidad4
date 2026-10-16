"""PDF 16:9 con el fondo institucional TecNM: teoría detallada + pasos + código por actividad."""
import sys, io, re, contextlib
from pathlib import Path
raiz = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(raiz))

import matplotlib
matplotlib.use("Agg")
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, Preformatted)
from PIL import Image as PI

from rl_core import *
from contenido import *

ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for n, f in [("DV", "DejaVuSans"), ("DVB", "DejaVuSans-Bold"), ("DM", "DejaVuSansMono")]:
    pdfmetrics.registerFont(TTFont(n, str(ttf / f"{f}.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

NOMBRE = "Nombre Apellido"
AZ, GU = colors.HexColor(AZUL), colors.HexColor(GUINDA)
PW, PH = 960, 540                                   # 16:9 (13.33 x 7.5 pulgadas)
FX, FY, FW, FH = 45, 66, 675, 402                   # zona blanca libre de adornos, engrane y franjas
FONDO = str(raiz / "imagenes" / "fondo_tecnm.jpg")

sT = ParagraphStyle("t", fontName="DVB", fontSize=26, textColor=AZ, leading=31, spaceAfter=6)
sA = ParagraphStyle("a", fontName="DVB", fontSize=15, textColor=AZ, leading=19, spaceAfter=5)
sH = ParagraphStyle("h", fontName="DVB", fontSize=10.5, textColor=GU, spaceBefore=7, spaceAfter=3, keepWithNext=1)
sB = ParagraphStyle("b", fontName="DV", fontSize=8.4, leading=11.6, alignment=4, spaceAfter=3.5)
sL = ParagraphStyle("l", parent=sB, leftIndent=12, firstLineIndent=-12, alignment=0, spaceAfter=2.5)
sC = ParagraphStyle("c", fontName="DV", fontSize=7, leading=9.2)
sCH = ParagraphStyle("ch", fontName="DVB", fontSize=7, leading=9.2, textColor=colors.white)
sM = ParagraphStyle("m", fontName="DV", fontSize=8, leading=11, textColor=colors.HexColor("#444"))
sCode = ParagraphStyle("code", fontName="DM", fontSize=6.5, leading=8.1, backColor=colors.HexColor("#F3F4F7"),
                       borderColor=colors.HexColor("#CCC"), borderWidth=.5, borderPadding=3, leftIndent=3,
                       rightIndent=3, spaceBefore=3, spaceAfter=5)
sOut = ParagraphStyle("out", parent=sCode, backColor=colors.HexColor("#FFF8E7"), borderColor=colors.HexColor("#D9B86A"))


def fondo(c, d):
    c.drawImage(FONDO, 0, 0, width=PW, height=PH)


doc = BaseDocTemplate(str(raiz / "pdfs" / "Unidad4_Aprendizaje_por_Refuerzo_Formato_TecNM.pdf"),
                      pagesize=(PW, PH), title="Unidad 4 - Aprendizaje por Refuerzo", author=NOMBRE,
                      leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0)
doc.addPageTemplates([PageTemplate(id="tecnm", frames=[Frame(FX, FY, FW, FH, leftPadding=0, rightPadding=0,
                                                              topPadding=0, bottomPadding=0)], onPage=fondo)])
W = FW
E = []


def md(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<i>\1</i>", t)
    return re.sub(r"`(.+?)`", r'<font name="DM" size="7.4">\1</font>', t)


def P(t, s=sB): E.append(Paragraph(md(t), s))
def H(t): E.append(Paragraph(md(t), sH))
GH = "https://github.com/alixpeg0/aprendizaje-por-refuerzo-unidad4"
COLAB = "https://colab.research.google.com/github/alixpeg0/aprendizaje-por-refuerzo-unidad4/blob/main/Unidad4_Aprendizaje_por_Refuerzo.ipynb"
sLink = ParagraphStyle("lk", fontName="DV", fontSize=7.5, leading=10, textColor=colors.HexColor("#444"), spaceAfter=4)


def enlaces():
    return Paragraph(f'<link href="{COLAB}" color="#1155CC"><u>▶ Abrir en Google Colab</u></link>&nbsp;&nbsp;·&nbsp;&nbsp;'
                     f'<link href="{GH}" color="#1155CC"><u>GitHub: alixpeg0/aprendizaje-por-refuerzo-unidad4</u></link>', sLink)


def titulo(t):
    E.append(PageBreak()); E.append(Paragraph(md(t), sA)); E.append(enlaces())
def B(lista):
    for t in lista: E.append(Paragraph("•&nbsp;&nbsp;" + md(t), sL))
def pasos(lista):
    for i, t in enumerate(lista, 1): E.append(Paragraph(f"<b>Paso {i}.</b> {md(t)}", sL))


NS = {}
exec("import numpy as np, pandas as pd\nfrom rl_core import *\nfrom contenido import *", NS)


def codigo(src):
    src = src.strip("\n")
    H("Código")
    E.append(Preformatted(src, sCode))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, NS)
    H("Salida al ejecutarlo")
    E.append(Preformatted(buf.getvalue().rstrip("\n"), sOut))


def tabla(cab, filas, anchos, negrita=True):
    d = [[Paragraph(md(str(c)), sCH) for c in cab]]
    for f in filas:
        d.append([Paragraph(("<b>%s</b>" if k == 0 and negrita else "%s") % md(str(c)), sC) for k, c in enumerate(f)])
    t = Table(d, colWidths=anchos, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), AZ), ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#999")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F7")])]))
    E.append(t); E.append(Spacer(1, 5))


def figura(nombre, fig, maxw=W, maxh=210):
    ruta = raiz / "imagenes" / nombre
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    matplotlib.pyplot.close(fig)
    imagen(ruta, maxw, maxh)


def imagen(ruta, maxw=W, maxh=FH - 30):
    w, h = PI.open(ruta).size
    k = min(maxw / w, maxh / h)
    E.append(Image(str(ruta), width=w * k, height=h * k))


def fmt(v): return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


# ====================================================================== PORTADA
E += [Spacer(1, 70), Paragraph("Aprendizaje por Refuerzo", sT),
      Paragraph("Unidad 4 · Cuaderno de trabajo resuelto: teoría, pasos y código", sA),
      Paragraph(f"Aprendizaje Automático · Maestría en Inteligencia Artificial · TecNM Virtual<br/>Alumno(a): {NOMBRE}", sM),
      Spacer(1, 18)]
tabla(["Sección", "Contenido"], [
    ["1", "Teoría general paso a paso y esquema"], ["2", "Actividad 25 · Elementos del RL en tres ejemplos"],
    ["3", "Actividad 26 · Entorno propio (AspiBot)"], ["4", "Actividad 27 · Una iteración de Q-learning"],
    ["5", "Actividad 28 · Agente y elección de algoritmo"], ["6", "Actividad 29 · Nueva tabla Q (S3: +3 → +6)"],
    ["7", "Actividad 30 · Agente propio (SkyBot)"]], [W * .12, W * .5], False)
E.append(Paragraph(f'<b>Notebook en Google Colab:</b> <link href="{COLAB}" color="#1155CC"><u>abrir y ejecutar el cuaderno</u></link><br/>'
                   f'<b>Repositorio en GitHub:</b> <link href="{GH}" color="#1155CC"><u>{GH}</u></link>', sB))
P("Las salidas mostradas se obtuvieron ejecutando el código de este documento.", sM)

# ====================================================================== 1 TEORIA GENERAL
titulo("1. Teoría general paso a paso")
teo = [
    ("¿Qué es el aprendizaje por refuerzo?",
     "Un **agente** aprende a decidir **interactuando** con un entorno. No recibe ejemplos etiquetados: recibe "
     "**recompensas** y descubre por ensayo y error qué acciones convienen. Su meta es maximizar la recompensa **total acumulada**."),
    ("Ciclo agente–entorno",
     "El agente observa el estado `s`, elige una acción `a`; el entorno responde con recompensa `r` y nuevo estado `s'`; "
     "el agente aprende de `(s, a, r, s')` y repite."),
    ("Seis elementos",
     "**Medio ambiente**, **estado**, **acción**, **recompensa** (señal inmediata), **política** π(s) (regla de decisión) y "
     "**valor** V(s) / Q(s,a) (recompensa futura esperada)."),
    ("Retorno y descuento γ",
     "El agente maximiza `G = r₁ + γ·r₂ + γ²·r₃ + …` con 0 ≤ γ ≤ 1. Con γ = 0.9, una recompensa a 2 pasos vale 0.81."),
    ("MDP",
     "Modelo formal (S, A, P, R, γ). Propiedad de Markov: el futuro depende **solo del estado actual**."),
    ("Ecuación de Bellman", "Valor = recompensa inmediata + valor descontado de lo que sigue: `Q*(s,a) = r + γ·max Q*(s',a')`."),
    ("Q-learning", "`Q(s,a) ← Q(s,a) + α·[ r + γ·max Q(s',a') − Q(s,a) ]`. El corchete es el **error TD**. Es *off-policy*."),
    ("ε-greedy", "Con prob. ε explora (acción al azar); con 1−ε explota (mayor Q). Se suele reducir ε con el tiempo."),
    ("Familias de algoritmos", "Valores (Q-learning, DQN), políticas (REINFORCE, PPO), actor-crítico (A2C, SAC), modelos (Dyna-Q, MCTS)."),
    ("Tipos de refuerzo", "**Positivo:** se añade un premio. **Negativo:** se retira algo desagradable. **Castigo:** se añade algo desagradable."),
]
for i, (t, x) in enumerate(teo, 1):
    E.append(Paragraph(f"<b>Paso {i}. {md(t)}.</b> {md(x)}", sL))
E.append(PageBreak())
E.append(Paragraph("Esquema de la teoría", sA))
imagen(raiz / "imagenes" / "esquema_teoria.png", W, FH - 28)

# ====================================================================== ACT 25
titulo("2. Actividad 25 · Detecta los elementos del aprendizaje por refuerzo")
P("**Objetivo:** identificar y comprender los conceptos básicos del RL en situaciones reales.")
H("Teoría necesaria: cómo reconocer cada elemento")
tabla(["Elemento", "Pregunta guía", "Definición detallada", "Error frecuente"], [
    ["Agente", "¿Quién decide y aprende?", "Entidad que observa, elige acciones y mejora con la experiencia.", "Confundirlo con el usuario o con todo el sistema."],
    ["Medio ambiente", "¿Qué responde a sus acciones y no controla directamente?", "Todo lo externo al agente: reglas, física, otros actores. Produce el siguiente estado y la recompensa.", "Meter dentro del agente cosas que no puede controlar."],
    ["Estado s", "¿Qué información tiene para decidir?", "Resumen de la situación. Idealmente cumple la propiedad de Markov (contiene todo lo relevante).", "Incluir datos irrelevantes o dejar fuera datos clave."],
    ["Acción a", "¿Qué puede hacer directamente?", "Conjunto de decisiones disponibles (discretas o continuas).", "Listar resultados en lugar de decisiones."],
    ["Recompensa r", "¿Qué número le dice si le fue bien o mal?", "Escalar que entrega el entorno tras cada acción. Define el objetivo del problema.", "Confundirla con el valor o con el estado."],
    ["Política π(s)", "¿Cómo elige su acción?", "Función de estado a acción (determinista) o a probabilidades (estocástica). Es lo que se aprende.", "Describir una acción suelta en vez de una regla."],
    ["Valor V(s), Q(s,a)", "¿Qué tan bueno es estar aquí a largo plazo?", "Retorno futuro esperado: V(s) = E[G | s]. No es una señal del entorno: lo estima el agente.", "Tomar la recompensa inmediata como valor."]],
    [W * .12, W * .2, W * .4, W * .28])
H("Recompensa vs. valor (la diferencia más importante)")
P("La **recompensa** es lo que ocurre *ahora*; el **valor** es lo que se *espera acumular desde ahora*. En ajedrez, capturar una "
  "pieza da recompensa inmediata, pero si deja al rey expuesto, el estado resultante tiene **valor bajo**. Por eso el agente "
  "no es voraz: elige por valor, no por recompensa inmediata.")
H("Tipos de recompensa y diseño")
B(["**Dispersa:** solo aparece al final (ej.: +100 al estacionar). Es fiel al objetivo, pero el aprendizaje es lento.",
   "**Densa:** guía cada paso (ej.: +0.1 por acercarse al cajón). Acelera el aprendizaje, pero mal diseñada genera *reward hacking* "
   "(el agente explota el premio sin cumplir el objetivo real).",
   "**Hipótesis de la recompensa:** cualquier objetivo puede expresarse como maximizar la recompensa acumulada esperada."])
H("Cómo se resuelve")
pasos(["Para cada ejemplo, pregunta **quién aprende** (agente) y **con qué interactúa** (medio ambiente).",
       "Describe **qué observa** (estado) y **qué puede hacer** (acciones).",
       "Define la **señal numérica** que indica si lo hizo bien o mal (recompensa), con valores concretos.",
       "Redacta la política (cómo decide) y el valor (qué tan buena es una situación a futuro).",
       "Explica el papel de la recompensa en cada caso."])
H("Resultado")
tabla(["Elemento", "a) Coche autónomo", "b) Videojuego", "c) Recomendador"], A25_FILAS, [W * .15, W * .29, W * .28, W * .28])
H("Gráfico: ciclo agente–entorno de cada ejemplo")
figura("act25_ciclos.png", figura_ciclo_ejemplos(), maxh=170)
H("Papel de la recompensa en cada caso")
for t, x in A25_RECOMPENSA:
    P(f"**{t}.** {x}")
P("Para comprobar que la recompensa es una señal numérica, se programa la del ejemplo a):")
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
''')

# ====================================================================== ACT 26
titulo("3. Actividad 26 · Diseña tu propio entorno de aprendizaje por refuerzo")
P("**Objetivo:** aplicar los conceptos básicos del RL de forma creativa.")
H("Teoría necesaria: cómo se diseña un problema de RL")
P("Diseñar un entorno de RL es **formular un MDP**: decidir qué ve el agente, qué hace y qué se le premia. "
  "Un mal diseño produce un agente que aprende bien… algo equivocado.")
pasos(["**Define el objetivo** en lenguaje natural (llegar a la suciedad con poca batería).",
       "**Elige el estado:** solo lo necesario para decidir. Si hay 25 celdas × 3 niveles de batería × 2 (sucio/limpio) → 150 estados: aún cabe en una tabla.",
       "**Elige las acciones:** pocas, claras y bajo control directo del agente.",
       "**Diseña la recompensa:** premia el objetivo (+10), cobra los costos (−1 por paso) y penaliza lo que quieres evitar (−3 por choque). Cuida las magnitudes: si el choque costara menos que el paso, no aprendería a evitarlos.",
       "**Define el fin del episodio:** al limpiar o al superar un máximo de pasos.",
       "**Elige la política de exploración:** ε-greedy con ε decreciente: `ε = max(ε_min, ε₀·(1 − episodio / (0.7·N)))`.",
       "**Define la evidencia de aprendizaje** antes de entrenar (métricas)."])
H("Cómo saber si el agente aprendió")
B(["**Retorno por episodio** (suma de recompensas): debe subir y estabilizarse. Se suaviza con media móvil.",
   "**Pasos por episodio:** debe bajar hasta el mínimo posible.",
   "**Política greedy** (ε = 0): ejecutarla y comprobar que llega siempre por la ruta corta, sin choques.",
   "**Comparar con el óptimo teórico:** aquí la ruta mínima tiene 6 movimientos: 5 pasos (−1 cada uno) + el último que limpia (+10) = **+5**."])
H("Descripción y diseño")
P(f"**{A26_TITULO}.** {A26_DESCRIPCION}")
tabla(["Elemento", "Diseño"], A26_FILAS, [W * .24, W * .76])
H("Simulación para comprobar el aprendizaje")
P("La rejilla está en `rl_core.py` (clase `Rejilla`). En cada paso el agente aplica la fórmula de Q-learning.")
codigo('''
env = Rejilla()                                  # 5x5, base (0,0), suciedad (3,3)
Q, retornos, pasos = entrenar_rejilla(env, episodios=400, alpha=0.5, gamma=0.95)
print("Retorno medio, primeros 20 episodios:", round(retornos[:20].mean(), 1))
print("Retorno medio, últimos 20 episodios :", round(retornos[-20:].mean(), 1))
print("Pasos medios, últimos 20 episodios  :", round(pasos[-20:].mean(), 1))
print("Ruta aprendida (greedy):", ruta_greedy(env, Q))
''')
imagen(raiz / "imagenes" / "act26_aprendizaje.png", W, 190)
H("Conclusión")
P(A26_CONCLUSION)

# ====================================================================== ACT 27
titulo("4. Actividad 27 · Simula una iteración de Q-learning")
P("**Objetivo:** aplicar la ecuación de Bellman para calcular Q(s,a). **Parámetros:** α = 0.5, γ = 0.9, Q = 0 al inicio.")
H("Teoría necesaria: de Bellman a Q-learning")
pasos(["**La tabla Q.** Guarda una estimación Q(s,a) por cada pareja estado-acción: «cuánta recompensa total espero si en s hago a y luego actúo bien». Se inicia en 0 (no sabe nada).",
       "**Idea de Bellman.** El retorno se descompone: `G = r + γ·G'`. Por tanto, el valor de (s,a) es la recompensa inmediata más γ por el mejor valor posible en el estado siguiente: `Q(s,a) = r + γ·max Q(s',a')`.",
       "**Problema:** al inicio Q es inexacta, así que esa igualdad no se cumple. La diferencia se llama **error TD**: `δ = [r + γ·max Q(s',a')] − Q(s,a)`.",
       "**Corrección parcial.** Se mueve Q hacia el objetivo un paso α: `Q ← Q + α·δ`, equivalente a un promedio ponderado `Q ← (1−α)·Q + α·objetivo`. Con α = 0.5 se toma mitad de lo viejo y mitad de lo nuevo; α = 1 olvidaría todo lo anterior y α = 0 no aprendería.",
       "**Papel de γ.** Con γ = 0.9 el futuro cuenta casi igual que el presente; el `max` supone que a partir de s' el agente actuará lo mejor posible (por eso Q-learning es *off-policy*).",
       "**Propagación hacia atrás.** Un estado cercano a una recompensa gana valor primero; en las siguientes visitas, los estados que llevan a él heredan ese valor.",
       "**Convergencia.** Si todas las parejas (s,a) se visitan muchas veces y α es adecuada, Q converge a Q*. Una iteración es solo el primer tramo."])
H("Ejemplo anotado (paso 4 de la tabla)")
P("Estado S3, acción Izquierda, `r = 5`, siguiente estado S2. Antes del paso, `Q(S3,Izq) = 0` y `max Q(S2,·) = 0.5`. "
  "Objetivo = 5 + 0.9·0.5 = **5.45**; error TD = 5.45 − 0 = 5.45; `Q = 0 + 0.5·5.45 =` **2.725**.")
H("Cómo se resuelve")
pasos(["Define el entorno: 3 estados y 3 acciones con su recompensa y siguiente estado.", "Inicia la tabla Q en 0.",
       "Elige una trayectoria de 6 experiencias que visite los tres estados y repita algún par (s,a).",
       "Para **cada** experiencia: objetivo = r + γ·max Q(s',·); error = objetivo − Q; Q nuevo = Q + α·error.",
       "Registra cada cálculo en una tabla y localiza el valor Q más alto.", "Interpreta el resultado."])
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
tabla(["Paso", "s", "a", "r", "s'", "Q ant.", "max Q(s',·)", "Objetivo", "Error TD", "Q nuevo"],
      [[fmt(v) for v in f] for f in p27.drop(columns="Cálculo").itertuples(index=False)],
      [W * .07, W * .07, W * .15, W * .06, W * .07, W * .1, W * .14, W * .12, W * .11, W * .11], False)
H("Gráfico: cómo evoluciona Q")
figura("act27_q.png", figura_q_pasos(construir_entorno(3)), maxh=190)
H("Conclusión")
for x in A27_CONCLUSION:
    P(x)

# ====================================================================== ACT 28
titulo("5. Actividad 28 · Diseña un agente y elige su algoritmo")
P("**Objetivo:** aplicar los conceptos aprendidos para seleccionar un algoritmo adecuado.")
H("Teoría necesaria: mapa de algoritmos de RL")
P("Elegir algoritmo es contestar tres preguntas: **¿conozco o puedo simular el entorno?**, **¿cuántos estados tengo?** y "
  "**¿las acciones son discretas o continuas?**")
tabla(["Algoritmo", "Cómo funciona", "Fortaleza", "Limitación"], [
    ["Q-learning (tabular)", "Tabla Q(s,a) actualizada con la fórmula de Bellman. Off-policy.", "Simple y con garantías de convergencia.", "La tabla crece con el nº de estados: no escala."],
    ["SARSA", "Como Q-learning, pero usa la acción a' realmente elegida (on-policy).", "Más cauteloso con riesgos durante el aprendizaje.", "Aprende la política que ejecuta, con exploración incluida."],
    ["DQN", "Una red neuronal estima Q(s,a;θ). Pérdida: (y − Q)², con y = r + γ·max Q(s',a';θ⁻). Usa **memoria de repetición** y **red objetivo**.", "Maneja espacios de estados enormes o continuos.", "Acciones discretas; entrenamiento sensible."],
    ["REINFORCE / PPO", "Optimizan directamente la política: ∇J = E[∇log π(a|s)·G]. PPO limita cada actualización (clipping).", "Acciones continuas, políticas estocásticas, estable (PPO).", "Necesitan muchas muestras."],
    ["Actor-crítico (A2C, SAC)", "El **actor** propone acciones; el **crítico** estima el valor y reduce la varianza.", "Equilibrio entre estabilidad y eficiencia.", "Más componentes que ajustar."],
    ["Dyna-Q", "Tras cada paso real aprende un **modelo** del entorno y hace k pasos de planificación simulados.", "Muy eficiente en datos cuando interactuar es caro.", "Un modelo impreciso lleva a planes erróneos."]],
    [W * .15, W * .35, W * .25, W * .25])
H("Criterios de decisión")
B(["**Pocos estados y acciones discretas** → Q-learning tabular o SARSA.",
   "**Estados muy grandes o continuos, acciones discretas** → DQN.",
   "**Acciones continuas** (ángulos, fuerzas) → PPO, SAC.",
   "**Interactuar es caro pero se puede simular/modelar** → Dyna-Q u otros basados en modelos."])
H("Cómo se resuelve")
pasos(["Describe el agente y el entorno (aquí: un semáforo inteligente en un cruce).",
       "Lista las **acciones**: pocas y discretas (mantener / cambiar fase).",
       "Diseña la **recompensa**: penalizar la espera total y los cambios de fase excesivos.",
       "Estima el **tamaño del espacio de estados**: si es enorme, descarta Q-learning tabular.",
       "Compara las familias con los criterios anteriores y elige.", "Justifica la elección con las características del problema."])
tabla(["Apartado", "Descripción"], A28_FILAS, [W * .2, W * .8])
H("Comparación aplicada a este agente")
tabla(["Algoritmo", "Familia", "Ideal cuando…", "¿Para este agente?"], A28_COMPARATIVA, [W * .17, W * .2, W * .27, W * .36])
H("Por qué no basta una tabla Q y cómo se calcula el objetivo de DQN")
codigo('''
import numpy as np
# --- 1) ¿Cabe una tabla Q? 4 carriles: fila de 0..30 autos y espera en 10 niveles por carril, 4 fases
n_estados = (31 ** 4) * (10 ** 4) * 4
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
''')

H("Gráfico: crecimiento de la tabla Q y árbol de decisión")
figura("act28_algoritmo.png", figura_algoritmo(), maxh=200)

# ====================================================================== ACT 29
titulo("6. Actividad 29 · Calcula tu propia tabla Q")
P("**Objetivo:** ver cómo cambiar una recompensa modifica los valores Q y la acción óptima.")
H("Teoría necesaria: valores óptimos y sensibilidad a la recompensa")
pasos(["**Política óptima.** Dado Q*, la mejor política es codiciosa: `π*(s) = argmax_a Q*(s,a)`.",
       "**Cálculo de Q*.** Se repite `Q(s,a) ← r + γ·max Q(s',a')` para todas las parejas hasta que ya no cambie (*iteración de valor*). Converge porque cada barrido reduce el error por un factor γ.",
       "**Una iteración vs. Q*.** Con una sola trayectoria la tabla es parcial; Q* es el valor al que tiende el aprendizaje completo.",
       "**Valor de quedarse siempre.** Si se cobra `r` en cada paso: `V = r + γ·r + γ²·r + … = r/(1−γ)`. Con γ = 0.9: V = 10·r (+3 → 30; +6 → 60).",
       "**Valor del ciclo S3→S2→S3.** `V3 = 5 + 0.9·V2` y `V2 = 1 + 0.9·V3`, luego `V3 = 5.9/(1 − 0.81) = 31.05`.",
       "**Umbral de cambio.** Quedarse gana cuando `10·r ≥ 5.9 + 8.1·r`, es decir `r ≥ 5.9/1.9 ≈ 3.105`. Con +3 el ciclo gana por poco; con +6 gana quedarse.",
       "**Lección de diseño.** Cuando dos opciones están casi empatadas, un cambio pequeño en la recompensa invierte la política. Por eso la recompensa debe diseñarse con cuidado."])
H("Cómo se resuelve")
pasos(["Parte del mismo entorno de la Actividad 27.", "Cambia la recompensa de (S3, Quedarse) de **+3 a +6**.",
       "Recalcula Q con la **misma trayectoria** y compara con la original.",
       "Calcula los valores **óptimos Q\\*** para ver la política final.", "Compara políticas y explica qué cambió y por qué."])
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
H("Gráficos: políticas óptimas y umbral de la recompensa")
imagen(raiz / "imagenes" / "act29_politicas.png", W * .9, 170)
figura("act29_umbral.png", figura_umbral(), maxw=W * .75, maxh=210)
H("Análisis")
for x in A29_ANALISIS:
    P(x)

# ====================================================================== ACT 30
titulo("7. Actividad 30 · Diseña tu propio agente y aplica la lógica del aprendizaje por refuerzo")
P("**Objetivo:** comprender estado, acción, recompensa, política y valor mediante un entorno propio.")
H("Teoría necesaria: tipos de refuerzo")
P("El RL se inspira en el **condicionamiento operante**: una conducta cambia según sus consecuencias. Hay cuatro casos:")
tabla(["", "Conducta AUMENTA", "Conducta DISMINUYE"], [
    ["Se **añade** un estímulo", "**Refuerzo positivo**: añadir un premio (+20 por entregar el paquete).", "**Castigo positivo**: añadir algo desagradable (−10 por chocar)."],
    ["Se **retira** un estímulo", "**Refuerzo negativo**: quitar algo molesto (al llegar a la estación desaparece la alarma de batería baja).", "**Castigo negativo**: quitar algo bueno (perder puntos o tiempo de vuelo)."]],
    [W * .2, W * .4, W * .4])
B(["**Cuidado con el nombre:** «negativo» no significa castigo. El refuerzo negativo **aumenta** la conducta (la que elimina lo desagradable).",
   "**En RL** todo se codifica como un número `r`: r > 0 favorece la acción, r < 0 la desalienta. Conceptualmente se distingue qué tipo de consecuencia representa.",
   "**Adaptación con el tiempo:** (1) ε baja y el agente explota más; (2) cada experiencia actualiza Q; (3) las acciones con mayor Q se vuelven más frecuentes; (4) la política converge a rutas cortas y seguras.",
   "**Evidencia de aprendizaje:** retorno medio alto y estable, pocos pasos por entrega, cero choques/caídas y política greedy consistente."])
H("Cómo se resuelve")
pasos(["Imagina un agente: un dron repartidor (**SkyBot**).", "Escribe su propósito en un párrafo.",
       "Completa la tabla: agente, entorno, estados, acciones, recompensas, política y evidencia de aprendizaje.",
       "Indica qué **tipo de refuerzo** usa y por qué.", "Explica cómo **se adapta** con el tiempo."])
H(A30_TITULO)
P("**Propósito.** " + A30_PROPOSITO)
tabla(["Elemento", "Diseño"], list(zip(A30_COLUMNAS, A30_FILA)), [W * .24, W * .76])
H("Tipo de refuerzo y adaptación")
P(A30_TIPO_Y_ADAPTACION)
H("Recompensas y política ε-greedy de SkyBot")
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
''')
P("Al bajar ε, SkyBot explota cada vez más lo aprendido: así se refleja que su comportamiento se adapta con el tiempo.")
H("Gráficos: recompensas de SkyBot y efecto de ε")
figura("act30_skybot.png", figura_skybot(), maxh=200)

doc.build(E)
print("OK", doc.filename)
