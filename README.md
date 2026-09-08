# prueba_gpt

Bienvenido al repositorio donde las ideas entran como pruebas y, con un poco de suerte, salen como código.

## ¿Qué es esto?

Un laboratorio oficial de cosas que quizá funcionen.

Estado actual:

- Si compila, celebramos.
- Si falla, era una prueba.
- Si funciona a la primera, sospechamos.
- Si nadie sabe por qué funciona, no se toca.

## Calculadora web

Ahora el repo incluye una calculadora con backend en Python y visualizador HTML.

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
