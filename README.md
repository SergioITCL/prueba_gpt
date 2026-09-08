# prueba_gpt

Repositorio de pruebas dividido por proyectos para que la raíz no parezca un cajón de cables.

## Estructura

```text
prueba_gpt/
├── calculator/
│   ├── calculator.py
│   └── index.html
├── games/
│   └── flappy_game.py
├── qubo/
│   ├── __init__.py
│   ├── THIRD_PARTY_NOTICES.md
│   ├── qubo_common.py
│   ├── qubo_problem_generator.py
│   ├── run_qubo_comparison.py
│   ├── smvc.py
│   ├── smvc_nodes.py
│   ├── vectorized_programming_solver.py
│   └── solvers/
│       ├── __init__.py
│       ├── benchmark_smvc.py
│       └── smvc_optimized/
│           ├── __init__.py
│           └── solver.py
├── pyproject.toml
└── README.md
```

## Instalación

Desde la raíz del repositorio:

```powershell
git pull
poetry install
```

Si Poetry avisa de que `pyproject.toml changed significantly since poetry.lock was last generated`, tienes un `poetry.lock` local antiguo. Regénéralo:

```powershell
Remove-Item .\poetry.lock -ErrorAction SilentlyContinue
poetry lock
poetry install
```

Puedes comprobar NumPy y SciPy con:

```powershell
poetry run python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

## Calculadora web

La calculadora vive en `calculator/` y mantiene juntos el backend Python y la interfaz HTML.

```powershell
poetry run python .\calculator\calculator.py
```

Después abre `http://127.0.0.1:8000`.

## Juego estilo Flappy Bird

```powershell
poetry run python .\games\flappy_game.py
```

Controles: espacio, flecha arriba o clic para volar.

## QUBO/QUDO

Todo el código relacionado con QUBO/QUDO está agrupado en `qubo/`.

Componentes principales:

- `vectorized_programming_solver.py`: programación dinámica vectorizada exacta para problemas locales de rango `k`.
- `smvc.py`: implementación SMVC original usada como referencia.
- `smvc_nodes.py`: nodos de la implementación SMVC de referencia.
- `qubo_problem_generator.py`: generador reproducible de instancias del proyecto del paper.
- `qubo_common.py`: función objetivo, estimación de `tau` y modelo común de resultados.
- `run_qubo_comparison.py`: compara programación dinámica exacta y SMVC.
- `solvers/smvc_optimized/`: versión optimizada de SMVC.
- `solvers/benchmark_smvc.py`: benchmark entre el SMVC original y el optimizado.

### SMVC optimizado

La versión optimizada mantiene la misma regla de decisión de SMVC, pero reduce el coste de ejecución mediante:

- eliminación de matrices densas temporales durante la reconstrucción de la solución;
- eliminación de `itertools.product` del camino crítico;
- tablas de estados y transiciones en base `d` cacheadas;
- contracción directa de vectores sin materializar la mayoría de operadores dispersos;
- evaluación vectorizada de energías locales con NumPy;
- exponentiales estabilizadas numéricamente mediante desplazamiento por el máximo;
- reducción mediante `numpy.einsum` para evitar temporales innecesarios.

La implementación original se conserva para poder validar resultados y medir aceleración.

Benchmark por defecto:

```powershell
poetry run python -m qubo.solvers.benchmark_smvc
```

Ejemplo más exigente:

```powershell
poetry run python -m qubo.solvers.benchmark_smvc --n 200 --k 5 --dits 2 --seed 170 --repetitions 10
```

El benchmark informa del tiempo medio, aceleración y comprueba que ambas variantes devuelven la misma solución para la instancia utilizada.

Uso directo desde Python:

```python
from qubo.solvers.smvc_optimized import solver_smvc_optimized

result = solver_smvc_optimized(
    q_matrix,
    q_row,
    dits=2,
    n_neighbors=3,
)
```

### Comparación QUBO: DP vectorizado vs SMVC

Comparación por defecto, con `n=20`, `k=2`, `dits=2` y seed 7:

```powershell
poetry run python .\qubo\run_qubo_comparison.py
```

Con parámetros propios:

```powershell
poetry run python .\qubo\run_qubo_comparison.py --n 30 --k 3 --dits 2 --seed 170
```

Con el generador de interacciones fijas:

```powershell
poetry run python .\qubo\run_qubo_comparison.py --n 30 --k 3 --seed 170 --instance-type fixed
```

La programación dinámica actúa como referencia exacta. El script informa del gap de SMVC respecto a ese óptimo.

## Filosofía del repositorio

1. Cada experimento tiene su carpeta.
2. La raíz contiene solo configuración y documentación.
3. Si funciona a la primera, se documenta antes de que deje de hacerlo.
4. Si nadie sabe por qué funciona, se añaden tests antes de tocarlo.
