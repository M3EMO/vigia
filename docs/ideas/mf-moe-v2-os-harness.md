# Propuesta MF-MoE v2: enrutamiento minimax, bits dinámicos y OS Harness (insumo del usuario, 2026-10-06)

> Texto provisto por el usuario. Revisión crítica: `docs/spec/mf-moe-v2-revision.md`.
> Decisión de orden (2026-10-06): 1) Vigía; 2) OS Harness; 3) agente con acceso a nivel kernel.

## 1. Resumen ejecutivo

Las arquitecturas Mixture-of-Experts (MoE) convencionales sufren de rigidez estructural, colapso de
expertos y una incapacidad nativa para medir la certeza de sus respuestas de forma eficiente. Se
propone la arquitectura MF-MoE, un ecosistema modular clásico diseñado para operar en la frontera de
los Sistemas de Modelado de Mundo para Sistemas Operativos (OS World Models). El conocimiento se
organiza en una supermatriz multidimensional indexada por ejes cognitivos ortogonales (Dominios vs.
Operaciones), donde cada celda es un nodo fractal autosimilar capaz de sufrir mitosis (expansión) o
fusión (contracción) de forma fuera de línea (offline) durante el re-entrenamiento continuo.

El sistema incorpora un Harness (arnés de control) asíncrono de bajo nivel que se acopla visualmente
a cualquier aplicación del entorno (Excel, Power BI, editores de código, editores de video) mediante
la captura continua del frame-buffer de la GPU y algoritmos de Cuantización Asimétrica Dinámica (bits
dinámicos por profundidad). El modelo extrae indicadores de confianza fáctica en tiempo real midiendo
la Divergencia de Kullback-Leibler (KL) y la entropía del enrutador interno en una sola pasada,
logrando ventanas de contexto virtuales superiores al millón de tokens mediante indexación jerárquica
y reduciendo los costos de VRAM drásticamente.

## 2. Pilares fundamentales

### A. Supermatriz fractal (el infinito mínimo y huellas digitales)

El conocimiento y los modos de percepción se segmentan en una matriz bidimensional indexada por ejes
ortogonales (eje X: dominios; eje Y: naturaleza operacional). Cada celda es un nodo fractal autosimilar:

- **Bifurcación dinámica (mitosis):** si una coordenada experimenta saturación por entropía de datos
  acumulada durante el re-entrenamiento continuo, se divide en sub-expertos clonando pesos e
  inyectando ruido gaussiano infinitesimal (ε ~ N(0, 1e-5)) para romper la simetría.
- **Morfología por sector (huellas digitales arquitectónicas):** el tejido fractal adopta topologías
  de árbol profundo y estrecho en entornos jerárquicos (legal, editores de código) y topologías planas
  y transicionales en sectores volátiles (finanzas). El conocimiento se convierte en la forma del
  circuito, no en datos memorizados.
- **Contracción evolutiva (weight merging):** si dos expertos registran gradientes con similitud de
  coseno superior a θ_merging = 0,92, se fusionan con un promedio ponderado de sus tensores,
  liberando memoria de GPU de forma permanente.

### B. Percepción híbrida y OS Harness (arnés de control)

El sistema se conecta a las aplicaciones del sistema operativo en tres niveles de libertad:

1. **Libertad máxima (APIs/DOM):** si la ventana activa es una aplicación web o software con API
   expuesta, el arnés se acopla directo al código y la latencia de procesamiento visual es cero.
2. **Accesibilidad estructural:** lee los árboles de objetos de aplicaciones que exponen
   automatización de interfaz (por ejemplo UI Automation en Windows).
3. **Caja negra (stream continuo del frame-buffer):** en software cerrado o antiguo, el arnés captura
   el flujo de imágenes de la GPU en RAM; un modelo de visión liviano genera bounding boxes en
   milisegundos y el arnés ejecuta clics o eventos de teclado virtuales con llamadas a nivel kernel.

### C. Cuantización asimétrica dinámica (bits dinámicos por profundidad)

- **Superficie fractal (nivel 1):** pesos ternarios [−1, 0, 1] (1,58 bits) para patrones comunes,
  sintaxis plana o interfaces estáticas.
- **Hojas profundas (nivel 3):** ante alta complejidad analítica (fórmula anidada de Excel, consulta
  SQL en Power BI), promoción local de bits a micro-expertos en FP16 o FP32.
- **Batch compute asimétrico:** el harness ordena los tokens por precisión requerida y agrupa los
  ternarios en registros de 32 bits con kernels de Triton para procesarlos en lotes simétricos.

### D. Sabiduría de las masas, contexto extendido y confianza endógena

- **Compresión de contexto jerárquica:** con contextos de más de un millón de tokens, el historial se
  fragmenta e indexa en las celdas de la matriz; la GPU solo atiende el micro-contexto de la
  coordenada activa, con costo operativo O(1) en RAM.
- **Confianza por divergencia KL:** cada enfoque emite su distribución sobre el siguiente token; la
  capa de agregación calcula la KL cruzada en una sola pasada. Si los expertos divergen, la confianza
  cae a cero y el arnés congela la salida, fuerza un bucle de razonamiento (chain-of-thought) o
  deriva la tarea a un entorno seguro antes de responder.

## 3. Ventajas esperadas

1. **Detección inmediata de alucinaciones** sin modelos jueces externos.
2. **Eficiencia de hardware:** un modelo con habilidades de nivel Frontier en una sola GPU de 80 GB.
3. **Universalidad operacional:** automatizador universal de tareas sin que las empresas expongan
   el código de sus herramientas.
4. **Transferencia de conocimiento estructural (ADN de IA):** el núcleo abierto protege los datos del
   cliente mientras el desarrollador recopila telemetría anonimizada de gradientes geométricos para
   inicializar futuros modelos base.
