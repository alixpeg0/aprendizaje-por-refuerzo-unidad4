"""Respuestas de las actividades conceptuales (25, 26, 28 y 30).
Fuente única: las usan el notebook y los PDFs."""

# ---------------------------------------------------------------- Actividad 25
A25_COLUMNAS = ["Elemento", "a) Coche autónomo que aprende a estacionarse",
                "b) Videojuego: monedas y enemigos", "c) Recomendador de películas"]
A25_FILAS = [
    ["Agente (quien aprende)", "El sistema de conducción del coche",
     "El personaje controlado por la IA", "El algoritmo recomendador"],
    ["Medio ambiente (entorno)", "Estacionamiento: cajones, otros autos, bordillos, física del vehículo",
     "El nivel del juego: plataformas, monedas, enemigos, reglas", "La plataforma y sus usuarios (catálogo, gustos, hábitos)"],
    ["Estado (s)", "Posición, orientación y velocidad del auto; distancias medidas por sensores",
     "Posición del personaje, de los enemigos y de las monedas; vidas restantes",
     "Perfil del usuario, historial de películas vistas, hora y dispositivo"],
    ["Acción (a)", "Girar el volante, acelerar, frenar, avanzar o retroceder",
     "Moverse izquierda/derecha, saltar, esperar", "Recomendar la película X (o una lista de ellas)"],
    ["Recompensa (r)", "+100 al quedar bien estacionado; −50 por chocar; −1 por cada segundo que pasa",
     "+1 por moneda; +50 por llegar a la meta; −10 por tocar un enemigo",
     "+1 si el usuario la ve completa; +0.5 si da clic; −1 si la ignora o abandona"],
    ["Política (π)", "Regla que decide la maniobra según lo que miden los sensores en cada instante",
     "Regla que decide cuándo saltar o avanzar para tomar monedas esquivando enemigos",
     "Regla que elige qué película mostrar a cada perfil de usuario"],
    ["Valor (V o Q)", "Qué tan prometedora es una posición: alto si ya está alineado y cerca del cajón",
     "Qué tan bueno es estar en cierto punto del nivel: alto cerca de monedas y lejos de enemigos",
     "Satisfacción futura esperada del usuario si se le muestra esa película (no solo el clic inmediato)"],
]
A25_RECOMPENSA = [
    ("a) Coche autónomo",
     "La recompensa le dice al coche qué es estacionarse bien: premia el éxito final y castiga choques y "
     "tardanza. Con ella el auto descubre por ensayo y error que las maniobras suaves y precisas son las "
     "que más puntos acumulan en total."),
    ("b) Videojuego",
     "Las monedas y la meta empujan al personaje a explorar y avanzar, mientras que el castigo por tocar "
     "enemigos le enseña a evitarlos. La recompensa es la única guía: nadie le explica las reglas, "
     "las deduce de lo que gana o pierde."),
    ("c) Recomendador",
     "La recompensa mide si la recomendación fue buena (el usuario la vio o la ignoró). Así el sistema "
     "ajusta sus sugerencias a cada persona y, si la recompensa incluye el largo plazo, evita recomendar "
     "solo lo que genera clics rápidos."),
]

# ---------------------------------------------------------------- Actividad 26
A26_TITULO = "AspiBot, el robot aspiradora que aprende su ruta"
A26_DESCRIPCION = (
    "AspiBot es un robot aspiradora doméstico que no trae el mapa de la casa programado: lo va "
    "descubriendo mientras trabaja. Su propósito es llegar a las zonas sucias en el menor número de "
    "movimientos posible, cuidando su batería y evitando muebles. Al inicio se mueve casi al azar y "
    "choca con frecuencia; cada vez que limpia recibe un premio y cada movimiento o choque le cuesta "
    "puntos. Después de muchos intentos aprende la ruta más corta y segura desde su base hasta la "
    "suciedad, y la repite aunque cambien algunos detalles de la casa.")
A26_FILAS = [
    ("Agente", "AspiBot, robot aspiradora."),
    ("Medio ambiente", "Una casa dividida en una rejilla de 5×5 celdas, con muebles (obstáculos), una base de carga en la esquina y una zona sucia."),
    ("Posibles estados", "Su celda actual (fila, columna), nivel de batería (alto/medio/bajo) y si la zona sucia ya fue limpiada."),
    ("Acciones", "Mover arriba, abajo, izquierda o derecha; aspirar."),
    ("Recompensas / penalizaciones", "+10 al limpiar la zona sucia; −1 por cada paso (gasta batería); −3 por chocar con un mueble o pared; −20 si se queda sin batería lejos de la base."),
    ("Política o estrategia", "ε-greedy sobre Q(s,a): al inicio explora mucho (ε alto) y con el tiempo explota (ε bajo), eligiendo la acción con mayor Q en cada celda."),
    ("¿Cómo sabré que aprendió?", "El retorno por episodio sube y se estabiliza; el número de pasos baja hasta la ruta mínima; ya no choca; y la política greedy (sin exploración) llega a la suciedad siempre por el mismo camino corto."),
]
A26_CONCLUSION = (
    "Sabré que AspiBot aprendió correctamente cuando, sin explorar (ε = 0), llegue a la zona sucia en el "
    "mínimo de pasos de forma consistente, sin choques, y su retorno promedio por episodio se mantenga "
    "estable en un valor alto. En la simulación del notebook el retorno promedio pasa "
    "de ≈ −82 (primeros 20 episodios) a ≈ +4.5 (últimos 20), muy cerca de +5, el máximo posible en esa "
    "rejilla (5 pasos de −1 y la limpieza de +10).")

# ---------------------------------------------------------------- Actividad 28
A28_AGENTE = "Controlador de semáforos de un cruce de avenidas"
A28_FILAS = [
    ("Agente", "Controlador inteligente de un semáforo en un cruce de dos avenidas."),
    ("Descripción del entorno", "Un cruce simulado (por ejemplo en SUMO) con flujo de autos variable según la hora. El estado incluye la longitud de la fila en cada carril, el tiempo de espera acumulado y la fase actual del semáforo; es un espacio muy grande (valores continuos)."),
    ("Acciones posibles", "Mantener la fase actual o cambiar a la siguiente fase (verde para el otro sentido). Son pocas acciones discretas."),
    ("Recompensas esperadas", "Recompensa = −(tiempo total de espera de los autos) en cada intervalo; −5 por cambios de fase demasiado frecuentes; +10 cuando se vacía una fila larga."),
    ("Tipo de algoritmo elegido", "Basado en valores con aproximación por red neuronal: DQN (Deep Q-Network)."),
    ("Justificación", "Con filas de longitud variable el número de estados es enorme, y una tabla Q como la de la Actividad 27 no cabría en memoria. DQN conserva la lógica de Q-learning pero estima Q(s,a) con una red neuronal. Como las acciones son pocas y discretas, un método basado en valores encaja bien (los basados en políticas, como PPO, se justifican sobre todo con acciones continuas). Además usa repetición de experiencias para aprender de forma estable."),
]
A28_COMPARATIVA = [
    ("Q-learning tabular", "Valores", "Pocos estados y acciones discretas", "No: el espacio de estados es demasiado grande"),
    ("DQN", "Valores + red neuronal", "Muchos estados, acciones discretas", "Sí: es la opción elegida"),
    ("PPO / REINFORCE", "Políticas", "Acciones continuas o políticas estocásticas", "Posible, pero más complejo de lo necesario"),
    ("Dyna-Q", "Basado en modelo", "Simulaciones baratas y estados discretos", "Útil si se aprende un modelo del tráfico, pero el modelo es difícil de ajustar"),
]

# ---------------------------------------------------------------- Actividad 30
A30_TITULO = "SkyBot: dron repartidor que aprende a entregar paquetes"
A30_PROPOSITO = (
    "SkyBot es un dron repartidor que debe entregar paquetes en un barrio sin conocer de antemano la "
    "mejor ruta. Por ensayo y error aprende a llegar al punto de entrega rápido, esquivar edificios y "
    "cables, y regresar a cargar antes de que se agote la batería.")
A30_COLUMNAS = ["Agente", "Medio ambiente", "Posibles estados", "Acciones",
                "Recompensas o penalizaciones", "Política (estrategia)", "Cómo sabrás si aprendió correctamente"]
A30_FILA = ["SkyBot, dron repartidor",
            "Barrio con edificios, cables, viento, una estación de carga y puntos de entrega",
            "Posición (x, y, altura), nivel de batería, si lleva paquete, distancia al destino",
            "Subir, bajar, avanzar en 4 direcciones, soltar paquete, volver a la estación",
            "+20 por entregar; +5 por cargar con batería baja; −1 por cada paso; −10 por chocar; −30 si cae por batería",
            "ε-greedy con Q(s,a): explora al inicio y después elige la acción de mayor valor; volver a cargar si la batería < 20 %",
            "Entrega en pocos pasos de forma consistente, sin choques ni caídas, con retorno promedio estable y alto"]
A30_TIPO_Y_ADAPTACION = (
    "SkyBot usa ambos tipos de refuerzo. El refuerzo positivo es el premio que se añade al entregar el "
    "paquete (+20) y al cargar la batería a tiempo; el refuerzo negativo aparece cuando llegar a la "
    "estación elimina el estado aversivo de batería baja, y además se aplican penalizaciones (−10 por "
    "chocar, −1 por paso) que desalientan conductas. Con el tiempo reduce la exploración (ε baja), "
    "actualiza sus valores Q con cada experiencia y su política converge hacia rutas más cortas, "
    "seguras y con recargas oportunas.")

# ---------------------------------------------------------------- Actividad 27 y 29 (texto)
A27_CONCLUSION = [
    "El valor Q más alto es **Q(S3, Izquierda) = 4.7518**. Significa que, con lo aprendido hasta ahora, la mejor "
    "decisión estando en S3 es ir a la izquierda: esa acción da +5 de inmediato y además conduce a S2, un estado "
    "desde el cual el agente ya descubrió que puede volver a S3 con recompensa (Q(S2,Derecha) = 1.9763). Es decir, "
    "un Q alto resume la **recompensa inmediata más el valor futuro descontado** de elegir esa acción.",
    "Observa que el aprendizaje se **propaga hacia atrás**: en el paso 5, Q(S2,Derecha) saltó de 0.5 a 1.9763 porque "
    "el siguiente estado (S3) ya tenía un valor alto (2.725). Con una sola iteración los valores todavía son "
    "pequeños y están incompletos (Q(S1,·) sigue en 0); al repetir muchos episodios, Q converge a los valores "
    "óptimos de la Actividad 29 (por ejemplo Q*(S3,Izquierda) ≈ 31.05).",
]
A29_ANALISIS = [
    "**¿Qué cambió en el comportamiento?** En S3 la acción óptima pasó de **Izquierda** (Q* = 31.05 frente a 30.95 "
    "de Quedarse, casi empatadas) a **Quedarse** (Q* = 60 frente a 54.5). Antes al agente le convenía ir y venir "
    "entre S2 y S3 cobrando +5 y +1; ahora se queda en S3 cobrando +6 en cada paso. En S1 y S2 sigue yendo a la "
    "derecha, pero todos los valores Q suben porque llegar a S3 vale mucho más.",
    "**¿Por qué la nueva recompensa cambió la acción óptima?** Porque el valor de una acción es la recompensa "
    "inmediata más el valor futuro descontado. Quedarse en S3 produce 6 + 0.9·V(S3), es decir V(S3) = 6/(1−0.9) = 60, "
    "mientras que el ciclo S3→S2→S3 rinde solo 5.9 por cada dos pasos (≈ 54.5). Al duplicar la recompensa de "
    "Quedarse, esa opción supera al ciclo. Tras una sola iteración la tabla todavía no lo refleja del todo, pero "
    "con el entrenamiento completo de Q-learning la política converge a la de la solución exacta.",
]
