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
│   ├── THIRD_PARTY_NOTICES.md
│   ├── qubo_common.py
│   ├── qubo_problem_generator.py
│   ├── run_qubo_comparison.py
│   ├── smvc.py
│   ├── smvc_nodes.py
│   └── vectorized_programming_solver.py
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

Ejecuta:

```powershell
poetry run python .\calculator\calculator.py
```

Después abre:

```text
http://127.0.0.1:8000
```

## Juego estilo Flappy Bird

El juego está separado en `games/`:

```powershell
poetry run python .\games\flappy_game.py
```

Controles: espacio, flecha arriba o clic para volar.

## Comparación QUBO: DP vectorizado vs SMVC

Todo el código relacionado con QUBO/QUDO está agrupado en `qubo/`.

Incluye:

- `vectorized_programming_solver.py`: programación dinámica vectorizada exacta para problemas locales de rango `k`.
- `smvc.py`: Sparse Matrix Vector Contraction.
- `smvc_nodes.py`: construcción de los nodos dispersos utilizados por SMVC.
- `qubo_problem_generator.py`: generador reproducible de instancias del proyecto del paper.
- `qubo_common.py`: función objetivo, estimación de `tau` y modelo común de resultados.
- `run_qubo_comparison.py`: ejecuta ambos algoritmos sobre exactamente la misma instancia y compara coste, tiempo y gap.
- `THIRD_PARTY_NOTICES.md`: atribución del código adaptado desde `SergioITCL/QUDO-tensor-network-solver`, rama `notebook_to_script`.

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
