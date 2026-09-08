# prueba_gpt

Bienvenido al repositorio donde las ideas entran como pruebas y, con un poco de suerte, salen como código.

## ¿Qué es esto?

Un laboratorio oficial de cosas que quizá funcionen.

Estado actual:

- Si compila, celebramos.
- Si falla, era una prueba.
- Si funciona a la primera, sospechamos.
- Si nadie sabe por qué funciona, no se toca.

## Comparación QUBO: DP vectorizado vs SMVC

El repositorio incluye ahora dos métodos adaptados de `SergioITCL/QUDO-tensor-network-solver`, rama `notebook_to_script`:

- `vectorized_programming_solver.py`: programación dinámica vectorizada exacta para problemas QUDO/QUBO con interacciones locales de rango `k`.
- `smvc.py`: Sparse Matrix Vector Contraction, método basado en contracciones de matrices dispersas.
- `smvc_nodes.py`: construcción de los nodos dispersos de SMVC.
- `qubo_problem_generator.py`: generador reproducible de problemas utilizado por los experimentos del proyecto del paper.
- `qubo_common.py`: evaluación del objetivo, estimación de `tau` y modelo común de resultados.
- `run_qubo_comparison.py`: genera una instancia y ejecuta ambos métodos sobre exactamente el mismo problema.

Instala las dependencias:

```bash
poetry install
```

Ejecuta una comparación por defecto (`n=20`, `k=2`, binario, seed 7):

```bash
poetry run python run_qubo_comparison.py
```

También puedes controlar los parámetros:

```bash
poetry run python run_qubo_comparison.py --n 30 --k 3 --dits 2 --seed 170
```

O utilizar instancias con interacciones fijas:

```bash
poetry run python run_qubo_comparison.py --n 30 --k 3 --seed 170 --instance-type fixed
```

El script muestra para cada algoritmo:

- solución obtenida;
- valor de la función objetivo;
- tiempo de ejecución;
- gap absoluto y relativo de SMVC respecto al óptimo obtenido por programación dinámica vectorizada.

La programación dinámica es exacta para esta estructura local. SMVC es el método de contracción que se compara contra ese óptimo.

La atribución del código adaptado está en `THIRD_PARTY_NOTICES.md`.

## Calculadora web

El repo también incluye una calculadora con backend en Python y visualizador HTML.

Archivos principales:

- `calculator.py`: servidor HTTP y API de cálculo, sin dependencias externas.
- `index.html`: interfaz web para sumar, restar, multiplicar, dividir, calcular potencias y módulos.

Para arrancarla:

```bash
git clone https://github.com/SergioITCL/prueba_gpt.git
cd prueba_gpt
python calculator.py
```

Después abre:

```text
http://127.0.0.1:8000
```

Dos números entran. Un resultado sale. Normalmente.

## Procedimiento estándar de ingeniería avanzada

1. Ejecuta algo.
2. Lee el error.
3. Busca el error.
4. Cambia una línea.
5. Introduce dos errores nuevos.
6. Repite hasta que parezca estable.

## Contribuciones

Las contribuciones son bienvenidas, especialmente las que reducen el número de `TODO` sin aumentar misteriosamente el número de `FIXME`.

## Garantía

Ninguna. Pero el README ha quedado bastante profesional para ser un repositorio de pruebas.
