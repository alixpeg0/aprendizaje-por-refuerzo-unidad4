"""Genera los PDFs entregables de las actividades 25-30 en /pdfs (ejecutar desde la raíz)."""
import sys, re
from pathlib import Path
raiz = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(raiz))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from rl_core import *
from contenido import *

# Fuente con soporte de π, γ, −, ≈, etc.
ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(ttf / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

NOMBRE = "Nombre Apellido"      # <- cambia por tu nombre antes de entregar
AZUL_C, GUINDA_C = colors.HexColor(AZUL), colors.HexColor(GUINDA)
sT = ParagraphStyle("t", fontName="DVB", fontSize=16, textColor=AZUL_C, spaceAfter=4, leading=20)
sH = ParagraphStyle("h", fontName="DVB", fontSize=12, textColor=GUINDA_C, spaceBefore=10, spaceAfter=4)
sB = ParagraphStyle("b", fontName="DV", fontSize=9.5, leading=14, alignment=4, spaceAfter=5)
sC = ParagraphStyle("c", fontName="DV", fontSize=8, leading=10.5)
sCH = ParagraphStyle("ch", fontName="DVB", fontSize=8, leading=10.5, textColor=colors.white)
sM = ParagraphStyle("m", fontName="DV", fontSize=7.5, leading=10, textColor=colors.HexColor("#444"))


def md_html(t):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t.replace("&", "&amp;").replace("<", "&lt;"))


def tabla_pdf(cabecera, filas, anchos, primera_col_negrita=True):
    data = [[Paragraph(md_html(str(c)), sCH) for c in cabecera]]
    for f in filas:
        data.append([Paragraph(("<b>%s</b>" if (k == 0 and primera_col_negrita) else "%s") % md_html(str(c)), sC)
                     for k, c in enumerate(f)])
    t = Table(data, colWidths=anchos, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL_C),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def df_a_filas(df, dec=4):
    return [[(f"{v:.{dec}f}".rstrip("0").rstrip(".") if isinstance(v, float) else v) for v in fila]
            for fila in df.itertuples(index=False)]


def construir(nombre, titulo, elementos, horizontal=False, actividad=""):
    pag = landscape(A4) if horizontal else A4
    doc = SimpleDocTemplate(str(raiz / "pdfs" / nombre), pagesize=pag, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title=titulo, author=NOMBRE)
    cab = [Paragraph(titulo, sT),
           Paragraph(f"Aprendizaje Automático · Unidad 4 · Maestría en Inteligencia Artificial · TecNM Virtual<br/>"
                     f"Alumno(a): {NOMBRE}", sM), Spacer(1, 8)]
    doc.build(cab + elementos)
    return doc.width


def img(fig, nombre, ancho):
    ruta = raiz / "imagenes" / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image as PI
    w, h = PI.open(ruta).size
    return Image(str(ruta), width=ancho, height=ancho * h / w)


(raiz / "pdfs").mkdir(exist_ok=True)

# --------------------------------------------------------------- 25
W = landscape(A4)[0] - 3.2 * cm
els = [Paragraph("Identifica el medio ambiente, estado, acción, recompensa, política y valor en cada ejemplo.", sB),
       tabla_pdf(A25_COLUMNAS, A25_FILAS, [W * .13, W * .29, W * .29, W * .29]),
       Paragraph("Papel de la recompensa en cada caso", sH)]
els += [Paragraph(f"<b>{t}.</b> {x}", sB) for t, x in A25_RECOMPENSA]
construir("Actividad25_Tema4.1_NombreApellido.pdf", "Actividad 25. Detecta los elementos del aprendizaje por refuerzo",
          els, horizontal=True)

# --------------------------------------------------------------- 26
W = A4[0] - 3.2 * cm
els = [Paragraph(A26_TITULO, sH), Paragraph(A26_DESCRIPCION, sB),
       tabla_pdf(["Elemento", "Diseño"], A26_FILAS, [W * .24, W * .76]),
       Paragraph("Evidencia de aprendizaje (simulación con Q-learning)", sH)]
env = Rejilla()
Qg, ret, pas = entrenar_rejilla(env)
els += [img(figura_rejilla(env, ret, pas, ruta_greedy(env, Qg)), "act26_aprendizaje.png", W),
        Paragraph("Conclusión", sH), Paragraph(A26_CONCLUSION, sB)]
construir("Actividad26_Tema4.1_NombreApellido.pdf", "Actividad 26. Diseña tu propio entorno de aprendizaje por refuerzo", els)

# --------------------------------------------------------------- 27
T, T6 = construir_entorno(3), construir_entorno(6)
Q27, p27 = simular_trayectoria(T)
els = [Paragraph("Entorno (3 estados × 3 acciones)", sH),
       tabla_pdf(["Estado s", "Acción a", "Siguiente s'", "Recompensa r"], df_a_filas(tabla_entorno(T)),
                 [W * .25] * 4, False),
       Paragraph("Parámetros: α = 0.5, γ = 0.9, Q(s,a) = 0 al inicio. Fórmula (ecuación de Bellman aplicada a Q-learning):<br/>"
                 "<b>Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s',a') − Q(s,a) ]</b>", sB),
       Paragraph("Cálculo paso a paso", sH),
       tabla_pdf(["Paso", "s", "a", "r", "s'", "Q ant.", "max Q(s',·)", "Objetivo", "Error TD", "Q nuevo"],
                 df_a_filas(p27.drop(columns="Cálculo")),
                 [W * .07, W * .07, W * .15, W * .06, W * .07, W * .1, W * .14, W * .12, W * .11, W * .11], False),
       Spacer(1, 6)]
els += [Paragraph(f"<b>Paso {r.Paso}:</b> {r.Cálculo}", sC) for r in p27.itertuples()]
els += [Paragraph("Tabla Q final", sH),
        tabla_pdf(["Estado"] + ACCIONES, [[s] + [f"{v:.4f}" for v in Q27.loc[s]] for s in ESTADOS],
                  [W * .25] * 4),
        Paragraph("Conclusión: ¿qué significa el valor Q más alto?", sH)]
els += [Paragraph(md_html(p), sB) for p in A27_CONCLUSION]
construir("Actividad27_Tema4.2_NombreApellido.pdf", "Actividad 27. Simula una iteración de Q-learning", els)

# --------------------------------------------------------------- 28
els = [Paragraph(A28_AGENTE, sH), tabla_pdf(["Apartado", "Descripción"], A28_FILAS, [W * .22, W * .78]),
       Paragraph("Comparación de algoritmos", sH),
       tabla_pdf(["Algoritmo", "Familia", "Ideal cuando…", "¿Para este agente?"], A28_COMPARATIVA,
                 [W * .17, W * .2, W * .28, W * .35])]
construir("Actividad28_Tema4.2_NombreApellido.pdf", "Actividad 28. Diseña un agente y elige su algoritmo", els)

# --------------------------------------------------------------- 29
Qn, p29 = simular_trayectoria(T6)
Qo_, Qn_ = q_optima(T), q_optima(T6)
fil = lambda Q: [[s] + [f"{v:.2f}" for v in Q.loc[s]] for s in ESTADOS]
els = [Paragraph("Cambio realizado: recompensa de (S3, Quedarse) de +3 a +6. Parámetros: α = 0.5, γ = 0.9.", sB),
       Paragraph("1) Una iteración (misma trayectoria de la Actividad 27, con +6)", sH),
       tabla_pdf(["Paso", "s", "a", "r", "s'", "Q ant.", "max Q(s',·)", "Objetivo", "Error TD", "Q nuevo"],
                 df_a_filas(p29.drop(columns="Cálculo")),
                 [W * .07, W * .07, W * .15, W * .06, W * .07, W * .1, W * .14, W * .12, W * .11, W * .11], False),
       Spacer(1, 4)]
els += [Paragraph(f"<b>Paso {r.Paso}:</b> {r.Cálculo}", sC) for r in p29.itertuples()]
els += [Paragraph("Tabla Q tras una iteración: original (+3) vs. nueva (+6)", sH),
        tabla_pdf(["Estado", "Orig. Izq", "Orig. Der", "Orig. Quedarse", "Nueva Izq", "Nueva Der", "Nueva Quedarse"],
                  [[s] + [f"{v:.4f}" for v in Q27.loc[s]] + [f"{v:.4f}" for v in Qn.loc[s]] for s in ESTADOS],
                  [W * .13] + [W * .145] * 6),
        Paragraph("2) Valores óptimos Q* (Bellman iterado hasta converger)", sH),
        Paragraph("<b>Original (+3)</b>", sB), tabla_pdf(["Estado"] + ACCIONES, fil(Qo_), [W * .25] * 4),
        Spacer(1, 4), Paragraph("<b>Nueva (+6)</b>", sB), tabla_pdf(["Estado"] + ACCIONES, fil(Qn_), [W * .25] * 4),
        Spacer(1, 6), img(figura_politicas(Qo_, Qn_), "act29_politicas.png", W * .95),
        Paragraph("Análisis", sH)]
els += [Paragraph(md_html(p), sB) for p in A29_ANALISIS]
construir("Actividad29_Tema4.3_NombreApellido.pdf", "Actividad 29. Calcula tu propia tabla Q", els)

# --------------------------------------------------------------- 30
W = landscape(A4)[0] - 3.2 * cm
els = [Paragraph(A30_TITULO, sH), Paragraph("<b>Propósito.</b> " + A30_PROPOSITO, sB),
       tabla_pdf(A30_COLUMNAS, [A30_FILA], [W / 7] * 7, False),
       Paragraph("Tipo de refuerzo y adaptación en el tiempo", sH), Paragraph(A30_TIPO_Y_ADAPTACION, sB)]
construir("Actividad30_Tema4.3_NombreApellido.pdf", "Actividad 30. Diseña tu propio agente y aplica la lógica del aprendizaje por refuerzo",
          els, horizontal=True)

# --------------------------------------------------------------- esquema
figura_esquema().savefig(raiz / "imagenes" / "esquema_teoria.png", dpi=130)
print("PDFs generados:", *sorted(p.name for p in (raiz / "pdfs").glob("*.pdf")), sep="\n  ")
