# Qué dispara la división de un experto

Type: grilling
Status: resolved
Blocked by: 01

## Question

¿La división la dispara la duda del router (señal de Vigía), la pérdida del experto o una combinación? Hay que fijar umbral, ventana, cómo se rompe la simetría de los dos clones (objeción 7) y si los clones comparten el router global o tienen un micro-router.

## Answer

Resuelto con el usuario el 2026-10-14 (grilling).

- Señal, tres variantes como condiciones: duda del router (entropía del router sin sesgo, promediada sobre los tokens que recibe cada experto; condición principal), pérdida de esos tokens (comparador) y la combinación de las dos (divide cuando ambas son altas). Las tres con la misma cantidad máxima de divisiones.
- Umbral relativo a la capa, estilo z-score de SEMA: cada 500 pasos, un experto se divide si su señal supera en 2 desvíos a la media de su capa. Máximo una división por capa en cada revisión, hasta 12 expertos ruteados por capa.
- Nacimiento del hijo: copia los pesos del padre más un ruido chico; su fila del router nace con la media de los estados ocultos de los tokens de alta duda del padre (idea de DynMoE). Se mide la divergencia padre-hijo en los primeros pasos (objeción 7). El clon con solo ruido queda como comparación posible, a decidir después de ver el primero.
- Router: global. El hijo suma una fila al router de su capa, y su sesgo de balanceo arranca igual al del padre. El sensor de Vigía sigue igual.
- Experimento aparte (pedido del usuario): micro-router jerárquico por padre, y la comparación router global contra jerárquico. Anotado en el mapa, fuera de la fase 1.
