"""Lógica compartida de la Unidad 4 (Aprendizaje por Refuerzo).

Este archivo es la ÚNICA fuente del código: el notebook de Colab lo incluye
tal cual en su primera celda y los PDFs se generan importándolo.
Solo requiere numpy, pandas y matplotlib.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------
# 1. Entorno de ejemplo: 3 estados x 3 acciones (determinista)
# ----------------------------------------------------------------------------
ESTADOS = ["S1", "S2", "S3"]
ACCIONES = ["Izquierda", "Derecha", "Quedarse"]


def construir_entorno(r_s3_quedarse=3):
    """Devuelve T[(estado, accion)] = (siguiente_estado, recompensa).

    S1 -- S2 -- S3   (los extremos chocan contra la pared)
    En S3: 'Quedarse' recoge una recompensa r_s3_quedarse (+3 en el original);
    'Izquierda' regresa a S2 con +5 (hay un atajo que rinde bien, pero obliga
    a ir y venir entre S2 y S3).
    """
    return {
        ("S1", "Izquierda"): ("S1", -1),
        ("S1", "Derecha"):   ("S2", 0),
        ("S1", "Quedarse"):  ("S1", 0),
        ("S2", "Izquierda"): ("S1", 0),
        ("S2", "Derecha"):   ("S3", 1),
        ("S2", "Quedarse"):  ("S2", 0),
        ("S3", "Izquierda"): ("S2", 5),
        ("S3", "Derecha"):   ("S3", -1),
        ("S3", "Quedarse"):  ("S3", r_s3_quedarse),
    }


def tabla_entorno(T):
    filas = [(s, a, s2, r) for (s, a), (s2, r) in T.items()]
    return pd.DataFrame(filas, columns=["Estado s", "Acción a", "Siguiente s'", "Recompensa r"])


# ----------------------------------------------------------------------------
# 2. Q-learning: una actualización y una trayectoria
# ----------------------------------------------------------------------------
def q_vacia():
    return pd.DataFrame(0.0, index=ESTADOS, columns=ACCIONES)


def actualizar_q(Q, s, a, r, s2, alpha=0.5, gamma=0.9):
    """Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]"""
    q_old = Q.loc[s, a]
    max_next = Q.loc[s2].max()
    objetivo = r + gamma * max_next
    error_td = objetivo - q_old
    q_new = q_old + alpha * error_td
    Q.loc[s, a] = q_new
    return dict(q_old=q_old, max_next=max_next, objetivo=objetivo, error_td=error_td, q_new=q_new)


TRAYECTORIA = [("S1", "Derecha"), ("S2", "Derecha"), ("S3", "Quedarse"),
               ("S3", "Izquierda"), ("S2", "Derecha"), ("S3", "Izquierda")]


def simular_trayectoria(T, trayectoria=TRAYECTORIA, alpha=0.5, gamma=0.9):
    """Aplica Q-learning paso a paso. Devuelve (Q final, tabla de pasos)."""
    Q = q_vacia()
    filas = []
    for i, (s, a) in enumerate(trayectoria, 1):
        s2, r = T[(s, a)]
        d = actualizar_q(Q, s, a, r, s2, alpha, gamma)
        formula = (f"Q({s},{a}) = {d['q_old']:.4f} + {alpha}·[{r} + {gamma}·{d['max_next']:.4f} "
                   f"− {d['q_old']:.4f}] = {d['q_new']:.4f}")
        filas.append(dict(Paso=i, s=s, a=a, r=r, **{"s'": s2},
                          **{"Q anterior": d["q_old"], "max Q(s',·)": d["max_next"],
                             "Objetivo r+γ·max": d["objetivo"], "Error TD": d["error_td"],
                             "Q nuevo": d["q_new"]}, Cálculo=formula))
    return Q, pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# 3. Solución "exacta" (ecuación de Bellman óptima) y Q-learning entrenado
# ----------------------------------------------------------------------------
def q_optima(T, gamma=0.9, iteraciones=500):
    """Iteración de valor sobre Q: Q(s,a) <- r + gamma * max_a' Q(s',a')."""
    Q = q_vacia()
    for _ in range(iteraciones):
        nueva = Q.copy()
        for (s, a), (s2, r) in T.items():
            nueva.loc[s, a] = r + gamma * Q.loc[s2].max()
        Q = nueva
    return Q


def entrenar_q_learning(T, episodios=500, pasos=20, alpha=0.5, gamma=0.9, eps=0.2, seed=0):
    rng = np.random.default_rng(seed)
    Q = q_vacia()
    for _ in range(episodios):
        s = ESTADOS[rng.integers(3)]
        for _ in range(pasos):
            a = ACCIONES[rng.integers(3)] if rng.random() < eps else Q.loc[s].idxmax()
            s2, r = T[(s, a)]
            actualizar_q(Q, s, a, r, s2, alpha, gamma)
            s = s2
    return Q


def politica(Q):
    return Q.idxmax(axis=1)


# ----------------------------------------------------------------------------
# 4. Mini-entorno de rejilla para ver "aprender" con una curva (AspiBot)
# ----------------------------------------------------------------------------
class Rejilla:
    """Rejilla N x N. El robot sale de (0,0) y debe llegar a la suciedad."""
    MOVS = {"Arriba": (-1, 0), "Abajo": (1, 0), "Izquierda": (0, -1), "Derecha": (0, 1)}
    NOMBRES = list(MOVS)

    def __init__(self, n=5, meta=(3, 3), obstaculos=((1, 1), (2, 3), (3, 1))):
        self.n, self.meta, self.obst = n, meta, set(obstaculos)

    def paso(self, s, accion):
        dr, dc = self.MOVS[accion]
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < self.n and 0 <= c < self.n) or (r, c) in self.obst:
            return s, -3, False          # choque: se queda y se penaliza
        if (r, c) == self.meta:
            return (r, c), 10, True      # limpia la suciedad
        return (r, c), -1, False         # cada paso cuesta batería


def entrenar_rejilla(env, episodios=400, alpha=0.5, gamma=0.95, eps_ini=1.0, eps_fin=0.05,
                     max_pasos=100, seed=1):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n, env.n, 4))
    retornos, longitudes = [], []
    for ep in range(episodios):
        eps = max(eps_fin, eps_ini * (1 - ep / (0.7 * episodios)))
        s, G, t = (0, 0), 0, 0
        for t in range(1, max_pasos + 1):
            i = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, fin = env.paso(s, env.NOMBRES[i])
            Q[s][i] += alpha * (r + gamma * (0 if fin else Q[s2].max()) - Q[s][i])
            s, G = s2, G + r
            if fin:
                break
        retornos.append(G)
        longitudes.append(t)
    return Q, np.array(retornos), np.array(longitudes)


def ruta_greedy(env, Q, max_pasos=30):
    s, ruta = (0, 0), [(0, 0)]
    for _ in range(max_pasos):
        s, _, fin = env.paso(s, env.NOMBRES[int(np.argmax(Q[s]))])
        ruta.append(s)
        if fin:
            break
    return ruta


def media_movil(x, k=20):
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ----------------------------------------------------------------------------
# 5. Figuras
# ----------------------------------------------------------------------------
AZUL, GUINDA, DORADO, GRIS = "#1B3A6B", "#8B1A3A", "#B8923A", "#F3F4F7"


def _caja(ax, xy, w, h, texto, color, tam=11, txt="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, texto, ha="center", va="center",
            color=txt, fontsize=tam, fontweight="bold", linespacing=1.3)


def _flecha(ax, p, q, texto="", rad=0.0, color="#333"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=2,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if texto:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, texto, ha="center", va="center",
                fontsize=10, bbox=dict(fc="white", ec="none", pad=1.5))


def figura_esquema():
    """Esquema de la teoría: ciclo agente-entorno + mapa de conceptos."""
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7.0, 7.85, "Aprendizaje por refuerzo: esquema de la teoría", ha="center",
            fontsize=16, fontweight="bold", color=AZUL)

    # Ciclo agente - entorno
    _caja(ax, (0.4, 4.6), 3.2, 1.6, "AGENTE\n(política π, valor V/Q)", AZUL)
    _caja(ax, (0.4, 1.7), 3.2, 1.6, "ENTORNO\n(medio ambiente)", GUINDA)
    _flecha(ax, (1.2, 4.6), (1.2, 3.3), "acción  aₜ", color=AZUL)
    _flecha(ax, (2.8, 3.3), (2.8, 4.6), "estado sₜ₊₁\nrecompensa rₜ₊₁", color=GUINDA)
    ax.text(2.0, 0.9, "Ciclo: observar → actuar → recibir recompensa → actualizar",
            ha="center", fontsize=9.5, style="italic")

    # Elementos
    ax.text(5.0, 7.15, "Elementos", fontsize=13, fontweight="bold", color=AZUL)
    elementos = [("Estado s", "lo que observa el agente"),
                 ("Acción a", "decisión que puede tomar"),
                 ("Recompensa r", "premio (+) o castigo (−)"),
                 ("Política π(s)", "qué acción elegir en cada s"),
                 ("Valor V(s)", "retorno esperado desde s"),
                 ("Valor Q(s,a)", "retorno esperado de (s, a)")]
    for i, (t, d) in enumerate(elementos):
        y = 6.55 - i * 0.62
        _caja(ax, (5.0, y - 0.22), 1.9, 0.5, t, DORADO, 10)
        ax.text(7.05, y + 0.03, d, va="center", fontsize=9.5)

    # Matemática
    ax.text(5.0, 2.75, "Ideas matemáticas", fontsize=13, fontweight="bold", color=AZUL)
    ax.text(5.0, 2.3, r"Retorno:  $G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \dots$",
            fontsize=11.5)
    ax.text(5.0, 1.75, r"Bellman:  $Q^*(s,a) = r + \gamma \max_{a'} Q^*(s',a')$", fontsize=11.5)
    ax.text(5.0, 1.2, r"Q-learning:  $Q(s,a) \leftarrow Q + \alpha\,[\,r + \gamma \max_{a'}Q(s',a') - Q\,]$",
            fontsize=11.5)
    ax.text(5.0, 0.65, "Proceso de Markov (MDP):  (S, A, P, R, γ) — el futuro solo depende del estado actual",
            fontsize=9.8)

    # Familias
    ax.text(11.4, 7.15, "Familias de algoritmos", fontsize=13, fontweight="bold", color=AZUL)
    fam = [("Basados en valores", "Q-learning, SARSA, DQN"),
           ("Basados en políticas", "REINFORCE, PPO"),
           ("Actor-crítico", "A2C, SAC"),
           ("Basados en modelos", "Dyna-Q, MCTS")]
    for i, (t, d) in enumerate(fam):
        y = 6.3 - i * 1.0
        _caja(ax, (9.9, y - 0.3), 3.6, 0.9, f"{t}\n{d}", AZUL if i % 2 == 0 else GUINDA, 9.5)

    ax.text(11.4, 2.4, "Tipos de refuerzo", fontsize=13, fontweight="bold", color=AZUL)
    _caja(ax, (9.9, 1.1), 1.75, 1.0, "Positivo\nañadir premio\n(+r)", "#2E7D4F", 9)
    _caja(ax, (11.75, 1.1), 1.75, 1.0, "Negativo\nquitar molestia\n(o penalizar −r)", "#B5532A", 9)
    fig.tight_layout()
    return fig


def figura_politicas(Q_orig, Q_new):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, Q, t in zip(axs, (Q_orig, Q_new), ("Original (S3,Quedarse = +3)", "Modificada (S3,Quedarse = +6)")):
        im = ax.imshow(Q.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(ACCIONES)
        ax.set_yticks(range(3)); ax.set_yticklabels(ESTADOS)
        mejor = Q.values.argmax(axis=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{Q.values[i, j]:.1f}", ha="center", va="center",
                        fontweight="bold" if j == mejor[i] else "normal",
                        color="white" if Q.values[i, j] > Q.values.max() * .55 else "black")
            ax.add_patch(plt.Rectangle((mejor[i] - .5, i - .5), 1, 1, fill=False, ec=GUINDA, lw=2.5))
        ax.set_title(t, fontsize=11, color=AZUL)
    fig.suptitle("Q* óptima (recuadro = mejor acción)", fontsize=12)
    fig.tight_layout()
    return fig


def figura_rejilla(env, retornos, longitudes, ruta, k=20):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(retornos, alpha=.25, color=AZUL)
    axs[0].plot(range(k - 1, len(retornos)), media_movil(retornos, k), color=GUINDA, lw=2)
    axs[0].set(title="Retorno por episodio", xlabel="episodio", ylabel="retorno")
    axs[1].plot(longitudes, alpha=.25, color=AZUL)
    axs[1].plot(range(k - 1, len(longitudes)), media_movil(longitudes, k), color=GUINDA, lw=2)
    axs[1].set(title="Pasos hasta limpiar", xlabel="episodio", ylabel="pasos")
    g = np.zeros((env.n, env.n))
    for o in env.obst:
        g[o] = 1
    axs[2].imshow(g, cmap="Greys", vmin=0, vmax=2)
    ys, xs = zip(*ruta)
    axs[2].plot(xs, ys, "-o", color=GUINDA, lw=2)
    axs[2].text(0, 0, "Base", ha="center", va="center", color=AZUL, fontweight="bold")
    axs[2].text(env.meta[1], env.meta[0], "Sucio", ha="center", va="center", color="green", fontweight="bold")
    axs[2].set(title="Ruta aprendida (política greedy)", xticks=range(env.n), yticks=range(env.n))
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 6. Figuras por actividad (25, 27, 28, 29, 30)
# ----------------------------------------------------------------------------
def figura_ciclo_ejemplos():
    """Act. 25: ciclo agente-entorno de cada uno de los tres ejemplos."""
    datos = [
        ("a) Coche autónomo", "Sistema de\nconducción", "Estacionamiento\n(cajones, autos)",
         "acción: volante,\nacelerar, frenar", "estado: posición, sensores\nr: +100 / −50 / −1"),
        ("b) Videojuego", "Personaje\n(IA)", "Nivel del juego\n(monedas, enemigos)",
         "acción: mover,\nsaltar, esperar", "estado: posiciones, vidas\nr: +1 moneda / −10 enemigo"),
        ("c) Recomendador", "Algoritmo\nrecomendador", "Usuario y catálogo\nde películas",
         "acción: recomendar\nla película X", "estado: perfil, historial\nr: +1 la ve / −1 la ignora"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (t, ag, en, acc, resp) in zip(axs, datos):
        ax.set_xlim(-1.6, 7.6); ax.set_ylim(0, 6); ax.axis("off")
        ax.set_title(t, fontsize=13, color=AZUL, fontweight="bold")
        _caja(ax, (0.9, 4.0), 4.2, 1.4, ag, AZUL, 10)
        _caja(ax, (0.9, 0.5), 4.2, 1.4, en, GUINDA, 9.5)
        _flecha(ax, (2.0, 4.0), (2.0, 1.9), color=AZUL)
        _flecha(ax, (4.0, 1.9), (4.0, 4.0), color=GUINDA)
        ax.text(0.95, 2.95, acc, ha="right", va="center", fontsize=8.5, color=AZUL)
        ax.text(4.3, 2.95, resp, ha="left", va="center", fontsize=8.5, color=GUINDA)
    fig.suptitle("Ciclo agente–entorno de cada ejemplo", fontsize=14, color=AZUL, fontweight="bold")
    fig.tight_layout()
    return fig


def figura_q_pasos(T, trayectoria=None, alpha=0.5, gamma=0.9):
    """Act. 27: evolución de los Q actualizados y tabla final."""
    trayectoria = trayectoria or TRAYECTORIA
    Q = q_vacia()
    hist = [Q.copy()]
    for s, a in trayectoria:
        s2, r = T[(s, a)]
        actualizar_q(Q, s, a, r, s2, alpha, gamma)
        hist.append(Q.copy())
    pares = list(dict.fromkeys(trayectoria))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    pal = [AZUL, GUINDA, DORADO, "#2E7D4F"]
    for (s, a), col in zip(pares, pal):
        axs[0].plot(range(len(hist)), [h.loc[s, a] for h in hist], "-o", color=col, lw=2, label=f"Q({s},{a})")
    axs[0].set(title="Evolución de Q en las 6 actualizaciones", xlabel="paso", ylabel="valor Q",
               xticks=range(len(hist)))
    axs[0].grid(alpha=.3); axs[0].legend()
    im = axs[1].imshow(Q.values, cmap="YlGnBu", aspect="auto")
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(ACCIONES)
    axs[1].set_yticks(range(3)); axs[1].set_yticklabels(ESTADOS)
    for i in range(3):
        for j in range(3):
            axs[1].text(j, i, f"{Q.values[i, j]:.2f}", ha="center", va="center",
                        color="white" if Q.values[i, j] > Q.values.max() * .55 else "black")
    axs[1].set_title("Tabla Q final")
    fig.tight_layout()
    return fig


def figura_algoritmo():
    """Act. 28: crecimiento de la tabla Q y árbol de decisión del algoritmo."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    n = np.arange(1, 41)
    celdas = (n + 1.0) ** 4 * 10 ** 4 * 4 * 2
    axs[0].semilogy(n, celdas, color=AZUL, lw=2.5, label="celdas de la tabla Q")
    axs[0].axhline(1e9, color=GUINDA, ls="--", label="≈ límite práctico (memoria y visitas)")
    axs[0].axvline(30, color=DORADO, ls=":", label="cruce real (hasta 30 autos)")
    axs[0].set(title="La tabla Q no escala", xlabel="máx. de autos por carril", ylabel="celdas Q (escala log)")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3, which="both")

    ax = axs[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("¿Qué algoritmo elegir?", fontsize=12)
    _caja(ax, (0.2, 4.6), 2.6, 0.9, "¿Pocos estados\ny acciones?", DORADO, 9)
    _caja(ax, (3.9, 4.6), 2.6, 0.9, "¿Acciones\ncontinuas?", DORADO, 9)
    _caja(ax, (7.3, 4.6), 2.5, 0.9, "¿Se puede\nsimular?", DORADO, 9)
    _caja(ax, (0.2, 2.2), 2.6, 0.9, "Q-learning\ntabular", AZUL, 9)
    _caja(ax, (3.9, 2.2), 2.6, 0.9, "PPO / SAC", AZUL, 9)
    _caja(ax, (7.3, 2.2), 2.5, 0.9, "Dyna-Q /\nbasados en modelo", AZUL, 9)
    _caja(ax, (3.9, 0.2), 2.6, 0.9, "DQN  ✔ (semáforo)", GUINDA, 9.5)
    _flecha(ax, (1.5, 4.6), (1.5, 3.1), "sí")
    _flecha(ax, (2.8, 5.05), (3.9, 5.05), "no")
    _flecha(ax, (5.2, 4.6), (5.2, 3.1), "sí")
    _flecha(ax, (6.5, 5.05), (7.3, 5.05), "no")
    _flecha(ax, (8.5, 4.6), (8.5, 3.1), "sí")
    _flecha(ax, (7.5, 4.6), (6.3, 1.1), "no")
    fig.tight_layout()
    return fig


def figura_umbral():
    """Act. 29: Q*(S3,·) según la recompensa de (S3,Quedarse)."""
    rs = np.arange(0, 8.01, 0.25)
    qq, qi = [], []
    for r in rs:
        Q = q_optima(construir_entorno(r), iteraciones=200)
        qq.append(Q.loc["S3", "Quedarse"]); qi.append(Q.loc["S3", "Izquierda"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(rs, qq, color=GUINDA, lw=2.5, label="Q*(S3, Quedarse)")
    ax.plot(rs, qi, color=AZUL, lw=2.5, label="Q*(S3, Izquierda)  (ciclo S3↔S2)")
    ax.axvline(5.9 / 1.9, color="gray", ls="--")
    ax.text(5.9 / 1.9 + .08, min(qi) + 1, "umbral ≈ 3.105", fontsize=9)
    for r, c in ((3, DORADO), (6, "#2E7D4F")):
        ax.axvline(r, color=c, ls=":")
        ax.text(r + .05, max(qq) * .93, f"r = +{r}", color=c, fontsize=10, fontweight="bold")
    ax.set(title="¿Cuándo conviene quedarse? Efecto de la recompensa", xlabel="recompensa de (S3, Quedarse)",
           ylabel="valor Q óptimo")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    return fig


def figura_skybot():
    """Act. 30: recompensas de SkyBot y cómo ε cambia su conducta."""
    eventos = ["Entrega", "Carga a tiempo", "Paso normal", "Choca", "Cae sin batería"]
    vals = [19, 4, -1, -11, -31]
    cols = ["#2E7D4F", "#2E7D4F", DORADO, GUINDA, GUINDA]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    axs[0].barh(eventos[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        axs[0].text(v + (1 if v >= 0 else -1), i, f"{v:+d}", va="center", ha="left" if v >= 0 else "right")
    axs[0].axvline(0, color="black", lw=.8)
    axs[0].set(title="Recompensa neta por evento (incluye −1 por paso)", xlabel="r")
    axs[0].set_xlim(-38, 26)
    eps = np.linspace(0, 1, 51)
    axs[1].plot(eps, 100 * ((1 - eps) + eps / 3), color=AZUL, lw=2.5, label="teórico: (1−ε) + ε/3")
    rng = np.random.default_rng(0)
    for e in (1.0, 0.5, 0.05):
        sims = [(rng.random() >= e) or (rng.integers(3) == 1) for _ in range(1000)]
        axs[1].plot(e, 100 * np.mean(sims), "o", color=GUINDA, ms=9)
    axs[1].plot([], [], "o", color=GUINDA, label="simulación (1000 decisiones)")
    axs[1].set(title="Al bajar ε, SkyBot elige más la mejor acción", xlabel="ε (exploración)",
               ylabel="% de veces que elige la mejor acción")
    axs[1].invert_xaxis(); axs[1].grid(alpha=.3); axs[1].legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 7. Figuras de la introducción (mapa de la unidad, flujo, fórmulas, parámetros)
# ----------------------------------------------------------------------------
def _panel(ax, xy, w, h, titulo, items, color, tam=10.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=GRIS, ec=color, lw=2))
    ax.add_patch(FancyBboxPatch((xy[0], xy[1] + h - 0.8), w, 0.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec=color, lw=2))
    ax.text(xy[0] + w / 2, xy[1] + h - 0.4, titulo, ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold")
    for i, t in enumerate(items):
        ax.text(xy[0] + 0.25, xy[1] + h - 1.25 - i * 0.48, "•  " + t, va="center", fontsize=tam)


def figura_mapa_unidad():
    """Introducción: mapa de toda la unidad y de las actividades."""
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")
    _caja(ax, (0.3, 6.15), 13.4, 0.85,
          "Aprendizaje por refuerzo: un agente aprende a decidir interactuando con un entorno,\n"
          "con el fin de maximizar la recompensa acumulada", AZUL, 11.5)
    cols = [
        (0.3, "4.1  Definición y conceptos básicos", GUINDA,
         ["Medio ambiente", "Estado y acción", "Recompensa", "Política π", "Valor V(s) y función de valor"],
         "Act. 25 · 26", "VOCABULARIO: identificar y diseñar\nlos elementos de un problema de RL"),
        (4.95, "4.2  Algoritmos de aprendizaje reforzado", AZUL,
         ["Basados en valores (Q-learning, DQN)", "Basados en políticas (REINFORCE, PPO)",
          "Basados en modelos (Dyna-Q)", "Ecuación de Bellman", "Cómo elegir el algoritmo"],
         "Act. 27 · 28", "CÁLCULO: aplicar Bellman y elegir\nel algoritmo adecuado"),
        (9.6, "4.3  Tipos y modelos de refuerzo", GUINDA,
         ["Refuerzo positivo y negativo", "Procesos de Markov (MDP)", "Q-learning paso a paso",
          "Sensibilidad a la recompensa", "Política y adaptación en el tiempo"],
         "Act. 29 · 30", "ANÁLISIS: interpretar cambios y\ndiseñar un agente propio"),
    ]
    for x, t, c, items, act, meta in cols:
        _panel(ax, (x, 2.55), 4.1, 3.3, t, items, c, 9.6)
        _caja(ax, (x + 1.05, 2.0), 2.0, 0.5, act, DORADO, 10)
        ax.text(x + 2.05, 1.05, meta, ha="center", va="center", fontsize=9.5, color=c, fontweight="bold")
    _flecha(ax, (4.45, 4.2), (4.9, 4.2)); _flecha(ax, (9.1, 4.2), (9.55, 4.2))
    ax.text(7.0, 0.25, "Idea central:  Estado → Acción → Recompensa → Nuevo estado → Aprender  (ciclo que se repite)",
            ha="center", fontsize=11, style="italic", color="#333")
    fig.tight_layout()
    return fig


def figura_flujo_qlearning():
    """Introducción: diagrama de flujo del algoritmo Q-learning."""
    fig, ax = plt.subplots(figsize=(14, 6.3))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.3); ax.axis("off")
    pasos = [
        ("1  Inicializar", "Q(s,a) = 0 para\ntodo s, a", AZUL, (0.4, 3.9)),
        ("2  Observar estado", "s = estado actual\ndel entorno", GUINDA, (3.9, 3.9)),
        ("3  Elegir acción", "ε-greedy: azar con prob. ε,\nmejor Q con prob. 1−ε", AZUL, (7.4, 3.9)),
        ("4  Ejecutar", "el entorno devuelve\nr y el nuevo estado s'", GUINDA, (10.9, 3.9)),
        ("5  Calcular objetivo", "y = r + γ · max Q(s',·)", AZUL, (10.9, 1.5)),
        ("6  Actualizar Q", "Q ← Q + α·(y − Q)", GUINDA, (7.4, 1.5)),
        ("7  Avanzar", "s ← s'  ¿fin del\nepisodio?", AZUL, (3.9, 1.5)),
    ]
    for t, d, c, (x, y) in pasos:
        ax.add_patch(FancyBboxPatch((x, y), 3.0, 1.55, boxstyle="round,pad=0.02,rounding_size=0.15", fc=c, ec="none"))
        ax.text(x + 1.5, y + 1.15, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + 1.5, y + 0.5, d, ha="center", va="center", color="white", fontsize=9)
    for p, q in [((3.4, 4.7), (3.9, 4.7)), ((6.9, 4.7), (7.4, 4.7)), ((10.4, 4.7), (10.9, 4.7)),
                 ((12.4, 3.9), (12.4, 3.05)), ((10.9, 2.3), (10.4, 2.3)), ((7.4, 2.3), (6.9, 2.3))]:
        _flecha(ax, p, q)
    _flecha(ax, (5.4, 3.05), (5.4, 3.9), "no: repetir", color=GUINDA)
    ax.text(5.4, 0.95, "sí: terminar el episodio y empezar otro", ha="center", fontsize=9.5, color=AZUL)
    ax.text(7.0, 0.3, "Se repite durante muchos episodios: Q converge a Q* y la política greedy se vuelve óptima.",
            ha="center", fontsize=10.5, style="italic")
    ax.text(7.0, 6.0, "Algoritmo Q-learning (off-policy)", ha="center", fontsize=13, fontweight="bold", color=AZUL)
    fig.tight_layout()
    return fig


_TARJETAS = {
    1: [
        ("1. Retorno descontado",
         r"$G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^{2} r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^{k}\, r_{t+k+1}$",
         "Suma de las recompensas futuras; cada una pesa γᵏ,\npor lo que lo lejano vale menos (0 ≤ γ ≤ 1).",
         "r = (1, 1, 10) y γ = 0.9  →  G = 1 + 0.9·1 + 0.81·10 = 10.0"),
        ("2. Valor de un estado",
         r"$V^{\pi}(s) = \mathbb{E}_{\pi}\,[\,G_t \mid s_t = s\,]$",
         "Retorno que se espera obtener empezando en s\ny siguiendo la política π.",
         "π = «siempre Quedarse», r = 3  →  V(S3) = 3 / (1 − 0.9) = 30"),
        ("3. Valor de una acción",
         r"$Q^{\pi}(s,a) = \mathbb{E}_{\pi}\,[\,G_t \mid s_t = s,\ a_t = a\,]$" "\n\n"
         r"$V^{\pi}(s) = \sum_{a} \pi(a \mid s)\; Q^{\pi}(s,a)$",
         "Como V, pero fijando primero la acción a:\npermite comparar acciones entre sí.",
         "Q*(S3, Izq) = 31.05  >  Q*(S3, Quedarse) = 30.95  →  conviene Izquierda"),
        ("4. Ecuación de Bellman óptima",
         r"$Q^{*}(s,a) = \mathbb{E}\,[\,r + \gamma \max_{a'} Q^{*}(s',a')\,]$",
         "Valor de (s,a) = recompensa inmediata +\nvalor descontado de lo mejor que se puede hacer después.",
         "Q*(S3, Izq) = 5 + 0.9 · V*(S2) = 5 + 0.9 · 28.95 = 31.05"),
    ],
    2: [
        ("5. Actualización de Q-learning",
         r"$Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]$" "\n\n"
         r"$= (1-\alpha)\,Q(s,a) + \alpha \cdot \mathrm{objetivo}$",
         "Acerca Q hacia el objetivo una fracción α\ndel error: es un promedio ponderado viejo/nuevo.",
         "Q = 0 + 0.5 · (5 + 0.9 · 0.5 − 0) = 2.725   (paso 4, Act. 27)"),
        ("6. Error TD (diferencia temporal)",
         r"$\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)$",
         "Qué tan equivocada estaba la estimación. Si δ = 0\nla tabla ya cumple la ecuación de Bellman.",
         "δ = 5 + 0.9 · 0.5 − 0 = 5.45"),
        ("7. Política codiciosa y ε-greedy",
         r"$\pi^{*}(s) = \mathrm{argmax}_{a}\; Q^{*}(s,a)$" "\n\n"
         r"$\pi_{\varepsilon}(a^{*}\!\mid s) = 1 - \varepsilon + \varepsilon/|A|$" "     "
         r"$\pi_{\varepsilon}(a \neq a^{*}\!\mid s) = \varepsilon/|A|$",
         "Elegir la acción de mayor Q (explotar) pero explorar\ncon probabilidad ε para descubrir algo mejor.",
         "ε = 0.2 y |A| = 3  →  mejor acción 86.7 %, cada otra 6.7 %"),
        ("8. Recompensa constante y MDP",
         r"$V = \sum_{k=0}^{\infty} \gamma^{k}\, r = \dfrac{r}{1-\gamma}$" "      "
         r"$\mathrm{MDP} = (S,\,A,\,P,\,R,\,\gamma)$",
         "Serie geométrica: el horizonte efectivo es ≈ 1/(1−γ) pasos.\nMarkov: P(s'|s,a) solo depende de (s, a).",
         "r = 3 → V = 30;   r = 6 → V = 60   (Actividad 29)"),
    ],
}


def figura_formulas(parte=1):
    """Introducción: hojas de fórmulas con significado y ejemplo numérico."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.6))
    for ax, (t, f, sig, ej) in zip(axs.ravel(), _TARJETAS[parte]):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(AZUL if parte == 1 else GUINDA); sp.set_linewidth(2)
        ax.set_facecolor("#FBFBFD")
        ax.text(0.03, 0.92, t, fontsize=11.5, fontweight="bold", color=GUINDA if parte == 1 else AZUL, va="center")
        ax.text(0.5, 0.62, f, fontsize=14, ha="center", va="center", color="#111", linespacing=1.4)
        ax.text(0.03, 0.31, "Significa: " + sig, fontsize=10, va="center", color="#333")
        ax.text(0.03, 0.09, "Ejemplo: " + ej, fontsize=10, va="center", color=DORADO, fontweight="bold")
    fig.tight_layout(pad=1.2)
    return fig


def figura_parametros():
    """Introducción: efecto de γ, α y ε."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    k = np.arange(0, 16)
    for g, c in ((0.5, "#B5532A"), (0.9, AZUL), (0.99, "#2E7D4F")):
        axs[0].plot(k, g ** k, "-o", ms=4, color=c, lw=2, label=f"γ = {g}")
    axs[0].annotate("0.9¹⁰ = 0.35:\nuna recompensa a 10 pasos\nvale 35 % con γ = 0.9", (10, 0.9 ** 10),
                    (6.0, 0.62), arrowprops=dict(arrowstyle="->"), fontsize=8.5)
    axs[0].set(title="γ (descuento): cuánto pesa el futuro", xlabel="pasos hacia el futuro k",
               ylabel="peso γᵏ de la recompensa")
    axs[0].legend(loc="center right"); axs[0].grid(alpha=.3)
    n = np.arange(0, 11)
    for a, c in ((0.1, "#B5532A"), (0.5, AZUL), (1.0, "#2E7D4F")):
        axs[1].plot(n, 5.45 * (1 - (1 - a) ** n), "-o", ms=4, color=c, lw=2, label=f"α = {a}")
    axs[1].axhline(5.45, color="gray", ls="--"); axs[1].text(6.3, 5.62, "objetivo = 5.45", fontsize=9); axs[1].set_ylim(0, 6.4)
    axs[1].set(title="α (tasa de aprendizaje): rapidez para corregir", xlabel="actualizaciones con el mismo objetivo",
               ylabel="valor Q")
    axs[1].legend(); axs[1].grid(alpha=.3)
    eps = [1.0, 0.5, 0.2, 0.05]
    mejor = [(1 - e) + e / 3 for e in eps]
    otras = [e / 3 for e in eps]
    x = np.arange(len(eps))
    axs[2].bar(x, [100 * m for m in mejor], color=AZUL, label="mejor acción (explotar)")
    axs[2].bar(x, [200 * o for o in otras], bottom=[100 * m for m in mejor], color=DORADO, label="las otras 2 (explorar)")
    for i, m in enumerate(mejor):
        axs[2].text(i, 100 * m / 2, f"{100 * m:.0f} %", ha="center", color="white", fontweight="bold")
    axs[2].set_xticks(x); axs[2].set_xticklabels([f"ε = {e}" for e in eps])
    axs[2].set(title="ε (exploración): probabilidad de elegir", ylabel="% de decisiones", ylim=(0, 135))
    axs[2].legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    return fig


def figura_anatomia_update(alpha=0.5, gamma=0.9):
    """Gráfico explicativo: anatomía de UNA actualización de Q-learning (paso 4 de la Act. 27)."""
    q_old, r, max_next = 0.0, 5.0, 0.5
    objetivo = r + gamma * max_next
    delta = objetivo - q_old
    q_new = q_old + alpha * delta
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 15); ax.set_ylim(-0.6, 8.4); ax.axis("off")

    # A) la experiencia
    ax.text(0.2, 8.1, "A) La experiencia (s, a, r, s')", fontsize=12, fontweight="bold", color=GUINDA)
    for i, (n, c) in enumerate((("S1", "#8896B3"), ("S2", AZUL), ("S3", GUINDA))):
        _caja(ax, (0.6 + i * 2.1, 6.55), 1.6, 0.8, n, c, 12)
    ax.add_patch(FancyArrowPatch((5.0, 7.45), (2.9, 7.45), connectionstyle="arc3,rad=0.45", arrowstyle="-|>",
                                 mutation_scale=20, lw=2.5, color=DORADO))
    ax.text(3.95, 7.95, "a = Izquierda,  r = +5", ha="center", fontsize=10, fontweight="bold", color=DORADO)
    ax.text(5.0, 6.2, "agente en s = S3", ha="center", fontsize=8.5)
    ax.text(2.7, 6.2, "llega a s' = S2", ha="center", fontsize=8.5)
    ax.text(7.2, 7.5, "Lo que ya sabe la tabla Q:", fontsize=10, fontweight="bold")
    ax.text(7.2, 7.0, f"Q(S3, Izquierda) = {q_old:g}   (aún no ha probado esta acción)", fontsize=10)
    ax.text(7.2, 6.55, f"max Q(S2, ·) = {max_next:g}   (lo mejor que conoce desde S2)", fontsize=10)

    # B) la fórmula coloreada
    ax.text(0.2, 5.65, "B) La fórmula, pieza por pieza", fontsize=12, fontweight="bold", color=GUINDA)
    toks = [(0.9, "Qnuevo", f"{q_new:g}", "#2E7D4F", "resultado"), (1.9, "=", "", "k", ""),
            (3.0, "Qviejo", f"{q_old:g}", AZUL, "estimación\nactual"), (4.0, "+", "", "k", ""),
            (5.0, "α", f"{alpha:g}", DORADO, "tasa de\naprendizaje"), (5.9, "·[", "", "k", ""),
            (6.8, "r", f"{r:g}", GUINDA, "recompensa\ninmediata"), (7.7, "+", "", "k", ""),
            (8.6, "γ", f"{gamma:g}", DORADO, "factor de\ndescuento"), (9.4, "·", "", "k", ""),
            (10.7, "max Q(s',·)", f"{max_next:g}", AZUL, "mejor valor\ndel siguiente estado"),
            (12.1, "−", "", "k", ""), (13.0, "Qviejo", f"{q_old:g}", AZUL, "estimación\nactual"), (14.0, "]", "", "k", "")]
    for x, s, v, c, lab in toks:
        ax.text(x, 4.95, s, ha="center", va="center", fontsize=13.5, fontweight="bold", color=c)
        if v:
            ax.text(x, 4.4, v, ha="center", va="center", fontsize=13, color=c)
        if lab:
            ax.text(x, 3.8, lab, ha="center", va="center", fontsize=7.8, color=c)
    ax.plot([6.5, 11.9], [3.3, 3.3], color=GUINDA, lw=2)
    ax.text(9.2, 3.05, f"OBJETIVO  y = r + γ·max Q(s',·) = {r:g} + {gamma:g}·{max_next:g} = {objetivo:.2f}",
            ha="center", fontsize=9.8, color=GUINDA, fontweight="bold")
    ax.plot([6.5, 13.8], [2.65, 2.65], color=AZUL, lw=2)
    ax.text(10.15, 2.4, f"ERROR TD  δ = y − Qviejo = {objetivo:.2f} − {q_old:g} = {delta:.2f}",
            ha="center", fontsize=9.8, color=AZUL, fontweight="bold")

    # C) el resultado en una recta numérica
    ax.text(0.2, 1.95, "C) ¿Qué hace la actualización? Mueve Q hacia el objetivo una fracción α del error", fontsize=12,
            fontweight="bold", color=GUINDA)
    x0, esc = 1.0, 2.35                     # 0 -> x0, 6 -> x0 + 6*esc
    ax.plot([x0, x0 + 6 * esc], [0.9, 0.9], color="black", lw=1.5)
    for v in range(0, 7):
        ax.plot([x0 + v * esc] * 2, [0.82, 0.98], color="black", lw=1)
        ax.text(x0 + v * esc, 0.55, str(v), ha="center", fontsize=9)
    X = lambda v: x0 + v * esc
    ax.plot(X(q_old), 0.9, "o", ms=13, color=AZUL); ax.text(X(q_old), 1.25, "Qviejo = 0", ha="center", color=AZUL, fontsize=9.5, fontweight="bold")
    ax.plot(X(objetivo), 0.9, "o", ms=13, color=GUINDA); ax.text(X(objetivo), 1.25, f"objetivo = {objetivo:.2f}", ha="center", color=GUINDA, fontsize=9.5, fontweight="bold")
    ax.plot(X(q_new), 0.9, "o", ms=13, color="#2E7D4F"); ax.text(X(q_new), 1.25, f"Qnuevo = {q_new:.3f}", ha="center", color="#2E7D4F", fontsize=9.5, fontweight="bold")
    ax.annotate("", (X(q_new), 0.1), (X(q_old), 0.1), arrowprops=dict(arrowstyle="-|>", color="#2E7D4F", lw=2.2))
    ax.text((X(q_old) + X(q_new)) / 2, -0.3, f"paso real = α·δ = {alpha:g}·{delta:.2f} = {alpha * delta:.3f}", ha="center",
            fontsize=9.3, color="#2E7D4F", fontweight="bold")
    ax.annotate("", (X(objetivo), 0.1), (X(q_new), 0.1), arrowprops=dict(arrowstyle="-|>", color="#999", lw=2, ls="--"))
    ax.text((X(q_new) + X(objetivo)) / 2, -0.3, "lo que falta (se corrige en futuras visitas)", ha="center", fontsize=8.5, color="#777")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    return fig
