# Dónde vive el código del experimento

Type: grilling
Status: resolved

## Question

¿El código del experimento vive dentro del repo Vigía (por ejemplo `research/mf_moe/`) o en un repo aparte? Cambia dependencias, tests, licencia y cómo se publica junto al paper.

## Answer

Resuelto con el usuario el 2026-10-14 (grilling).

- Repo aparte `mf-moe`, en `~/Proyectos/mf-moe`, que instala Vigía como dependencia para el sensor de duda. El producto Vigía no recibe código de investigación.
- Privado en GitHub hasta publicar el paper; licencia Apache 2.0, igual que el núcleo de Vigía.
- Crear el repo local y el remoto privado es un paso de construcción: se hace al cerrar la spec, y crear el remoto en GitHub pide confirmación del usuario.
