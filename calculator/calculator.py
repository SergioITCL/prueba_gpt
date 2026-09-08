#!/usr/bin/env python3
"""Calculadora web mínima sin dependencias externas."""

from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

HOST = "127.0.0.1"
PORT = 8000
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("No se puede dividir entre cero. Ni siquiera con actitud.")
    return a / b


def power(a: float, b: float) -> float:
    return a**b


def modulo(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("El módulo entre cero tampoco existe. Buen intento.")
    return a % b


OPERATIONS: dict[str, Callable[[float, float], float]] = {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide,
    "**": power,
    "%": modulo,
}


def calculate(a: float, b: float, operation: str) -> float:
    """Ejecuta una operación permitida y devuelve un resultado finito."""
    if operation not in OPERATIONS:
        raise ValueError(f"Operación no soportada: {operation}")

    result = OPERATIONS[operation](a, b)
    if isinstance(result, complex) or not math.isfinite(result):
        raise ValueError("El resultado se ha ido demasiado lejos para esta calculadora.")
    return result


class CalculatorHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(404, "Aquí no hay nada. La calculadora vive en /")
            return

        try:
            body = INDEX_FILE.read_bytes()
        except FileNotFoundError:
            self.send_error(500, "Falta index.html. La interfaz ha escapado.")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/calculate":
            self._send_json({"error": "Endpoint no encontrado."}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 10_000:
                raise ValueError("Petición vacía o demasiado grande.")

            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            a = float(payload["a"])
            b = float(payload["b"])
            operation = str(payload["operation"])

            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Los números deben ser finitos.")

            result = calculate(a, b, operation)
            self._send_json({"result": result})
        except (KeyError, TypeError, json.JSONDecodeError):
            self._send_json({"error": "Datos inválidos."}, status=400)
        except (ValueError, OverflowError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[calculadora] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), CalculatorHandler)
    print(f"Calculadora lista en http://{HOST}:{PORT}")
    print("Ctrl+C para cerrar el chiringuito.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCalculadora apagada. Los números pueden descansar.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
