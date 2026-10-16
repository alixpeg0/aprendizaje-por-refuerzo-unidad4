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
