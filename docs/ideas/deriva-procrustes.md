# Deriva del espacio vectorial y alineación de Procrustes (insumo del usuario, 2026-10-04)

> Texto provisto por el usuario. Revisión crítica: `docs/spec/mf-moe-revision.md`, sección
> "Deriva de representación y alineación de Procrustes". Las marcas `[0.4]`, `[0.3, 0.14]` y `[0.6]`
> vienen del original y no tienen fuente identificable.

## El peligro oculto: el derivamiento del espacio vectorial (representation drift)

Cuando un micro-experto se dilata de forma local y se re-entrena continuamente con los datos
específicos de un cliente, sus pesos cambian [0.4]. Al cambiar sus pesos, el significado geométrico
de sus vectores se desplaza.

Si el experto de "Derecho de Córdoba" se hiperespecifica de forma aislada, va a empezar a generar
representaciones vectoriales que el resto de la supermatriz fractal ya no sabe cómo interpretar. La
Capa de Agregación de Masas va a intentar calcular el consenso (Divergencia KL), pero la medición va
a dar un error absurdo porque los expertos dejaron de hablar el mismo idioma universal del modelo
base [0.4]. El fractal colapsa en autismo informático.

## La solución primordial: Procrustes Alignment o regulación de anclaje

Para que las dilataciones locales funcionen y los "expertos diplomáticos" puedan intercomunicarse,
hace falta una restricción matemática obligatoria en el backpropagation continuo:

- **Pérdida de anclaje ortogonal:** cada vez que un micro-experto fractal se actualiza o sufre
  mitosis, se le añade un término de penalización a su función de pérdida que mide qué tanto se
  alejó del espacio vectorial del modelo base congelado [0.4].
- **Alineación de Procrustes:** es un algoritmo clásico que toma la matriz de salida del experto
  dilatado y la "rota" geométricamente en el espacio multidimensional para volver a alinearla con
  el estándar del modelo base, sin destruir el conocimiento nuevo que aprendió.

De esta manera, los bloques primordiales siguen siendo piezas de Lego compatibles entre sí sin
importar qué tan profundo hayan viajado en el fractal [0.4].

## Cómo cambia el plan de 14 días

Esto no rompe el plan; lo vuelve hiper-realista. Cuando el plan de 14 días dice "Leé el artículo
de InnerExpert completo y anotá su método" y "Completá un curso corto de PyTorch con un ejemplo de
hooks", lo primordial es buscar cómo extraen ellos las señales sin corromper el espacio vectorial
del modelo base [0.3, 0.14].

Validar que los expertos no se desalineen al mutar es el paso técnico cero de la Etapa 1
(Investigación) [0.6].
