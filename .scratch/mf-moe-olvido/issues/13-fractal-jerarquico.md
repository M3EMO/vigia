# MoE fractal jerárquico: ¿un árbol de expertos con mitosis encuentra estructura y olvida menos?

Type: grilling
Status: open
Blocked by: —

## Question

La fase 1 refutó la versión plana: dividir solo agregaba un experto a la lista del router, y padre e hijo
competían por los mismos tokens. Objeción del usuario (2026-10-20): la idea fractal es que el modelo encuentre
formas nuevas de subdividir y especificar a los expertos, no que gane capacidad. Pedido anterior del usuario
(2026-10-14, en el mapa): micro-router jerárquico por padre y comparación con el router global. ¿Un MoE con
ruteo en árbol y división por mitosis con partición se especializa, transfiere y olvida menos que un MoE fijo
con repetición?

## Qué le faltaba a la versión plana

1. Ruteo jerárquico: el router elige una raíz y baja por compuertas hasta una hoja.
2. Mitosis con partición: el padre reparte su territorio, y los tokens viejos siguen yendo a la hija vieja.
3. Medir la especialización: si las hojas se alinean con las tareas, no solo el olvido.

Dato de la fase 1: la versión plana tampoco aprende más rápido (transferencia, pasos en fila / pasos sola,
menor es mejor: fijo+repetición 1,12; división sin fusión 1,14; principal 1,36; recuperación 1,67).

## Prototipo (mf-moe, `mf_moe/tree.py`)

- `TreeRouter`: top-3 entre 6 raíces con el sesgo de balanceo de la fase 1; cada nodo interno tiene una
  compuerta lineal de dos hijos (top-1). El peso del token es el de la raíz; la compuerta recibe gradiente por
  paso directo (p / p.detach() vale 1), así la mitosis no cambia la salida.
- Mitosis (`tree.split`): la hoja que duda (mismo disparador de la fase 1, z 3,5 cada 500 pasos, hasta 12 hojas)
  queda como hija vieja y nace una hija nueva (clon con ruido). La compuerta parte los tokens del padre por la
  mitad a lo largo de la dirección de la duda; los dudosos van a la nueva. El árbol puede seguir dividiéndose
  (una hija puede volver a hacer mitosis).
- `freeze_old`: la hija vieja queda congelada y guarda lo aprendido.
- El árbol vive en buffers y entra al checkpoint; `tree.resize` rearma las formas para retomar.
- `tree.specialization`: información mutua normalizada I(hoja; tarea) / H(tarea) por capa; ahora toda corrida
  de la fase 1 la reporta.
- Tests (CPU): sin divisiones el árbol rutea igual que el router plano; la mitosis no cambia la salida y parte
  los tokens del padre (30-70 %, los dudosos a la nueva); la hija vieja congelada no se mueve y la compuerta
  aprende; el retome reconstruye el árbol; corren de punta a punta y retoman a mitad de tarea.

## Propuesta de experimento (spec, sin correr)

- Condiciones nuevas: `jerárquico/duda` y `jerárquico/duda+congelar`, 6 → 12 hojas, con repetición, sin fusión.
- Datos y pares: las 12 tareas de la fase 1, mismos órdenes, semillas 0-2 × con/sin nombre (12 corridas). Se
  parean con las corridas ya hechas de fijo+repetición y de división sin fusión: no hace falta repetir controles.
- Línea de base de especialización: las corridas viejas no la registraron; se suman 2 corridas de división sin
  fusión y 2 de fijo+repetición (semilla 0) con el código nuevo.
- Predicciones si la idea fractal es cierta: (a) especialización del árbol mayor que la del plano; (b)
  transferencia menor que la del control; (c) BWT desde lo aprendido de `jerárquico/duda+congelar` mayor que el
  del control.
- Veredicto: bootstrap pareado IC 95 % con 6 pares, como la fase 1.
- Costo: ~3 h por corrida (~4,6 unidades), 16 corridas ≈ 74 unidades. La primera corrida hace de piloto: si
  falla o tarda más del doble que una corrida plana, se frena.

## Preguntas abiertas para el usuario

1. Datos: ¿tareas aritméticas de la fase 1 (barato, con familias implícitas: suma → binsuma, invsuma, ordsuma,
   sumamult; mult → modmult, digmult) o idiomas emparentados (finés, estonio, húngaro; turco, azerí, kazajo),
   que necesitan un modelo con tokenizer de texto?
2. ¿Las 4 corridas de línea de base de especialización entran, o alcanza con comparar contra el árbol sin
   congelar?

## Comments

- 2026-10-20, prototipo construido y con tests (mf-moe, ver commit del día). Ninguna corrida en GPU todavía.
