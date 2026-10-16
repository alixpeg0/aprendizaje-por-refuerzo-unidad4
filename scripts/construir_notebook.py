"""Genera Unidad4_Aprendizaje_por_Refuerzo.ipynb (ejecutar desde la raíz del repo)."""
import nbformat as nbf
from pathlib import Path

raiz = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
C = nb.cells
md = lambda t: C.append(nbf.v4.new_markdown_cell(t.strip("\n")))
code = lambda t: C.append(nbf.v4.new_code_cell(t.strip("\n")))

REPO = "alixpeg0/aprendizaje-por-refuerzo-unidad4"

md(f"""
# Aprendizaje Automático — Unidad 4: Aprendizaje por Refuerzo
**Cuaderno de trabajo resuelto (Actividades 25 a 30)** · Maestría en Inteligencia Artificial · TecNM Virtual

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/Unidad4_Aprendizaje_por_Refuerzo.ipynb)

**Contenido:** teoría paso a paso → esquema → Act. 25 → Act. 26 → Act. 27 → Act. 28 → Act. 29 → Act. 30.
Para ejecutar todo: *Entorno de ejecución → Ejecutar todo*. Solo usa numpy, pandas y matplotlib.
""")

md("## 0. Código base (entorno, Q-learning y figuras)\nEjecuta primero esta celda. Es el mismo archivo `rl_core.py` del repositorio.")
code((raiz / "rl_core.py").read_text(encoding="utf-8").split('"""', 2)[2])
code("""
# Respuestas de las actividades conceptuales (mismo archivo contenido.py del repositorio)
""" + (raiz / "contenido.py").read_text(encoding="utf-8").split('"""', 2)[2])
code("""
from IPython.display import display, Markdown
pd.set_option("display.max_colwidth", None, "display.width", 200)

def tabla(columnas, filas):
    return pd.DataFrame(filas, columns=columnas)
""")

# --------------------------------------------------------------- INTRODUCCION
md("# Introducción: la unidad en diagramas y fórmulas")
md("## Mapa de la unidad\nTres bloques que van del vocabulario (4.1), a los algoritmos (4.2) y al análisis de tipos y modelos (4.3); cada uno con dos actividades.")
code("figura_mapa_unidad(); plt.show()")
md("## Cómo aprende un agente: flujo de Q-learning\nEl agente repite los pasos 2-7 durante muchos episodios; Q se acerca a Q* y la política greedy se vuelve óptima.")
code("figura_flujo_qlearning(); plt.show()")
md("## Fórmulas esenciales\nCada tarjeta trae la fórmula, su significado y un ejemplo numérico de este cuaderno.")
code("figura_formulas(1); plt.show()")
code("figura_formulas(2); plt.show()")
md("## Los parámetros γ, α y ε\n- **γ**: cuánto pesa el futuro (horizonte efectivo ≈ 1/(1−γ)).\n- **α**: qué fracción del error se corrige en cada actualización.\n- **ε**: probabilidad de explorar con una acción al azar.")
code("figura_parametros(); plt.show()")
md("## Gráfico explicativo: anatomía de una actualización de Q-learning\nSe desglosa el paso 4 de la Actividad 27: experiencia (S3, Izquierda, +5, S2), cada pieza de la fórmula con su valor, el objetivo, el error TD y cómo Q se mueve una fracción α hacia el objetivo.")
code("figura_anatomia_update(); plt.show()")

# --------------------------------------------------------------- TEORIA
teoria = (raiz / "teoria.md").read_text(encoding="utf-8").replace("![Esquema de la teoría](imagenes/esquema_teoria.png)", "*(El esquema se dibuja en la siguiente sección.)*")
md("# 1. Teoría\n" + teoria.replace("## Teoría paso a paso", "", 1))

md("## Esquema de la teoría")
code("""
fig = figura_esquema()
plt.show()
""")

# --------------------------------------------------------------- ACT 25
md("""
# Actividad 25. Detecta los elementos del aprendizaje por refuerzo
**Objetivo:** identificar los conceptos básicos del RL en situaciones reales.
**Entregable:** tabla con los elementos de cada ejemplo + explicación (3 a 5 líneas) del papel de la recompensa.
""")
code("""
act25 = tabla(A25_COLUMNAS, A25_FILAS)
display(act25.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}).hide(axis="index"))
""")
md("### Gráfico: ciclo agente–entorno de cada ejemplo")
code("""
figura_ciclo_ejemplos()
plt.show()
""")
md("### Papel de la recompensa en cada caso")
code("""
for titulo, texto in A25_RECOMPENSA:
    display(Markdown(f"**{titulo}.** {texto}"))
""")

# --------------------------------------------------------------- ACT 26
md("""
# Actividad 26. Diseña tu propio entorno de aprendizaje por refuerzo
**Entregable:** descripción del agente (mínimo 5 renglones) y tabla con: agente, medio ambiente, estados, acciones, recompensas, política y cómo sabrás que aprendió.
""")
code("""
display(Markdown(f"## {A26_TITULO}\\n\\n{A26_DESCRIPCION}"))
display(tabla(["Elemento", "Diseño"], A26_FILAS).style.set_properties(**{"text-align": "left"}).hide(axis="index"))
""")
md("### Comprobación: ¿aprende AspiBot? (simulación con Q-learning)")
code("""
env = Rejilla()
Q_g, retornos, pasos = entrenar_rejilla(env)
ruta = ruta_greedy(env, Q_g)
print(f"Retorno medio, primeros 20 episodios: {retornos[:20].mean():.1f}")
print(f"Retorno medio, últimos 20 episodios : {retornos[-20:].mean():.1f}")
print(f"Pasos medios, últimos 20 episodios  : {pasos[-20:].mean():.1f}")
print("Ruta greedy:", ruta)
figura_rejilla(env, retornos, pasos, ruta)
plt.show()
""")
md("### Conclusión")
code("display(Markdown(A26_CONCLUSION))")

# --------------------------------------------------------------- ACT 27
md(r"""
# Actividad 27. Simula una iteración de Q-learning
**Objetivo:** aplicar la ecuación de Bellman para calcular Q(s,a).
**Parámetros:** $\alpha = 0.5$, $\gamma = 0.9$, $Q = 0$ al inicio.

Como el cuaderno no incluye el "ejemplo proporcionado", se crea un entorno similar con **3 estados** (S1, S2, S3) en línea y **3 acciones** (Izquierda, Derecha, Quedarse). Estas son sus reglas:
""")
code("""
T = construir_entorno(r_s3_quedarse=3)
tabla_entorno(T)
""")
md(r"""
**Fórmula:** $Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]$

Una *iteración* = un episodio corto de 6 experiencias elegidas para visitar los tres estados y repetir algunas parejas (s,a). Cada fila de la tabla es una actualización.
""")
code("""
Q27, pasos27 = simular_trayectoria(T)
display(pasos27.drop(columns="Cálculo").round(4).style.hide(axis="index"))
print("Cálculos con la fórmula sustituida:")
for f in pasos27["Cálculo"]:
    print(" ", f)
""")
md("### Tabla Q final (después de las 6 actualizaciones)")
code("""
display(Q27.round(4))
mejor = Q27.stack().idxmax()
print(f"Valor Q más alto: Q{mejor} = {Q27.stack().max():.4f}")
""")
md("### Gráfico: cómo evoluciona Q en las 6 actualizaciones")
code("""
figura_q_pasos(T)
plt.show()
""")
md("### Conclusión")
code("""
for p in A27_CONCLUSION:
    display(Markdown(p))
""")


# --------------------------------------------------------------- ACT 28
md("""
# Actividad 28. Diseña un agente y elige su algoritmo
**Entregable:** descripción estructurada con entorno, acciones, recompensas, tipo de algoritmo y justificación.
""")
code("""
display(Markdown(f"## Agente: {A28_AGENTE}"))
display(tabla(["Apartado", "Descripción"], A28_FILAS).style.set_properties(**{"text-align": "left"}).hide(axis="index"))
display(Markdown("### Comparación de familias de algoritmos para este problema"))
display(tabla(["Algoritmo", "Familia", "Ideal cuando…", "¿Para este agente?"], A28_COMPARATIVA).style.set_properties(**{"text-align": "left"}).hide(axis="index"))
""")

md("### Gráfico: la tabla Q no escala y qué algoritmo elegir")
code("""
figura_algoritmo()
plt.show()
""")
# --------------------------------------------------------------- ACT 29
md(r"""
# Actividad 29. Calcula tu propia tabla Q
Se usa el mismo entorno de la Actividad 27, pero se cambia la recompensa de **(S3, Quedarse) de +3 a +6**. Se recalculan los valores Q de dos formas:
1. **Una iteración** (la misma trayectoria de 6 pasos de la Act. 27).
2. **Valores óptimos Q\*** resolviendo la ecuación de Bellman hasta converger, para comparar la política final.
""")
code("""
T6 = construir_entorno(r_s3_quedarse=6)
Q27_nuevo, pasos29 = simular_trayectoria(T6)
display(Markdown("**Una iteración con +6 en (S3, Quedarse)**"))
display(pasos29.drop(columns="Cálculo").round(4).style.hide(axis="index"))
for f in pasos29["Cálculo"]:
    print(" ", f)
display(Markdown("**Comparación de la tabla tras una iteración**"))
comp = pd.concat({"Original (+3)": Q27, "Nueva (+6)": Q27_nuevo}, axis=1).round(4)
display(comp)
""")
code("""
Q_orig = q_optima(T)       # recompensa original +3
Q_new = q_optima(T6)       # recompensa nueva +6
display(Markdown("**Q\\* óptima — original (+3)**")); display(Q_orig.round(2))
display(Markdown("**Q\\* óptima — nueva (+6)**"));  display(Q_new.round(2))
display(Markdown("**Política (mejor acción por estado)**"))
display(pd.DataFrame({"Original (+3)": politica(Q_orig), "Nueva (+6)": politica(Q_new)}))
figura_politicas(Q_orig, Q_new); plt.show()
""")
md("### Gráfico: ¿a partir de qué recompensa conviene quedarse?")
code("""
figura_umbral()
plt.show()
""")
md("### Comprobación: Q-learning entrenado (no solo la solución exacta)")
code("""
Q_ent = entrenar_q_learning(T6, episodios=500)
display(Q_ent.round(2))
print("Política aprendida por Q-learning:", politica(Q_ent).to_dict())
print("Diferencia máxima con Q* =", round(float((Q_ent - Q_new).abs().max().max()), 4))
""")
md("### Análisis")
code("""
for p in A29_ANALISIS:
    display(Markdown("- " + p))
""")


# --------------------------------------------------------------- ACT 30
md("""
# Actividad 30. Diseña tu propio agente y aplica la lógica del aprendizaje por refuerzo
**Archivo a entregar:** `Actividad2_Tema4.2_NombreApellido.pdf` (según el cuaderno).
""")
code("""
display(Markdown(f"## {A30_TITULO}\\n\\n**Propósito.** {A30_PROPOSITO}"))
display(pd.DataFrame([A30_FILA], columns=A30_COLUMNAS).style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}).hide(axis="index"))
display(Markdown("### Tipo de refuerzo y adaptación en el tiempo\\n\\n" + A30_TIPO_Y_ADAPTACION))
""")
md("### Gráfico: recompensas de SkyBot y efecto de ε")
code("""
figura_skybot()
plt.show()
""")
md("""
---
**Fin del cuaderno.** Repositorio: https://github.com/""" + REPO)

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["colab"] = {"provenance": []}
out = raiz / "Unidad4_Aprendizaje_por_Refuerzo.ipynb"
nbf.write(nb, out)
print("escrito", out)
