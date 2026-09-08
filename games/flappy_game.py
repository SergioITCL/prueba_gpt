#!/usr/bin/env python3
"""Mini juego inspirado en Flappy Bird usando solo tkinter."""

from __future__ import annotations

import random
import tkinter as tk
from dataclasses import dataclass

WIDTH = 480
HEIGHT = 640
GROUND_HEIGHT = 70
BIRD_X = 120
BIRD_RADIUS = 16
GRAVITY = 0.55
FLAP_VELOCITY = -8.8
PIPE_WIDTH = 78
PIPE_GAP = 170
PIPE_SPEED = 3.4
SPAWN_EVERY = 105
FRAME_MS = 16


@dataclass
class PipePair:
    x: float
    gap_y: float
    top_id: int
    bottom_id: int
    scored: bool = False


class FlappyGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PyFlappy")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=WIDTH,
            height=HEIGHT,
            bg="#9adcf7",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.running = False
        self.game_over = False
        self.frame = 0
        self.score = 0
        self.best_score = 0
        self.bird_y = HEIGHT / 2
        self.bird_velocity = 0.0
        self.pipes: list[PipePair] = []

        self._draw_background()
        self.bird_id = self.canvas.create_oval(0, 0, 0, 0, fill="#ffd84d", outline="#222", width=2)
        self.eye_id = self.canvas.create_oval(0, 0, 0, 0, fill="white", outline="#222")
        self.pupil_id = self.canvas.create_oval(0, 0, 0, 0, fill="#222", outline="#222")
        self.beak_id = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, fill="#ff8b3d", outline="#222")

        self.score_id = self.canvas.create_text(
            WIDTH / 2,
            48,
            text="0",
            font=("Segoe UI", 28, "bold"),
            fill="white",
        )
        self.message_id = self.canvas.create_text(
            WIDTH / 2,
            HEIGHT / 2 - 40,
            text="ESPACIO o clic para empezar\n\nNo estampes al pájaro.",
            font=("Segoe UI", 17, "bold"),
            fill="#17324d",
            justify="center",
        )
        self.best_id = self.canvas.create_text(
            WIDTH / 2,
            HEIGHT - 35,
            text="Mejor: 0",
            font=("Segoe UI", 12, "bold"),
            fill="#17324d",
        )

        self._draw_bird()

        self.root.bind("<space>", self._on_action)
        self.root.bind("<Up>", self._on_action)
        self.canvas.bind("<Button-1>", self._on_action)

        self._game_loop()

    def _draw_background(self) -> None:
        self.canvas.create_oval(25, 80, 155, 125, fill="white", outline="")
        self.canvas.create_oval(100, 65, 220, 125, fill="white", outline="")
        self.canvas.create_oval(305, 145, 420, 185, fill="white", outline="")
        self.canvas.create_oval(365, 125, 475, 185, fill="white", outline="")

        ground_y = HEIGHT - GROUND_HEIGHT
        self.canvas.create_rectangle(0, ground_y, WIDTH, HEIGHT, fill="#d7c27a", outline="")
        self.canvas.create_rectangle(0, ground_y, WIDTH, ground_y + 12, fill="#76b947", outline="")

    def _draw_bird(self) -> None:
        x1 = BIRD_X - BIRD_RADIUS
        y1 = self.bird_y - BIRD_RADIUS
        x2 = BIRD_X + BIRD_RADIUS
        y2 = self.bird_y + BIRD_RADIUS
        self.canvas.coords(self.bird_id, x1, y1, x2, y2)

        self.canvas.coords(
            self.eye_id,
            BIRD_X + 3,
            self.bird_y - 10,
            BIRD_X + 14,
            self.bird_y + 1,
        )
        self.canvas.coords(
            self.pupil_id,
            BIRD_X + 9,
            self.bird_y - 7,
            BIRD_X + 13,
            self.bird_y - 3,
        )
        self.canvas.coords(
            self.beak_id,
            BIRD_X + 14,
            self.bird_y - 2,
            BIRD_X + 28,
            self.bird_y + 4,
            BIRD_X + 14,
            self.bird_y + 8,
        )

    def _on_action(self, _event: object | None = None) -> None:
        if self.game_over:
            self._reset()
            self._start()
            return

        if not self.running:
            self._start()

        self.bird_velocity = FLAP_VELOCITY

    def _start(self) -> None:
        self.running = True
        self.game_over = False
        self.canvas.itemconfigure(self.message_id, text="")

    def _reset(self) -> None:
        for pipe in self.pipes:
            self.canvas.delete(pipe.top_id)
            self.canvas.delete(pipe.bottom_id)

        self.pipes.clear()
        self.frame = 0
        self.score = 0
        self.bird_y = HEIGHT / 2
        self.bird_velocity = 0.0
        self.canvas.itemconfigure(self.score_id, text="0")
        self._draw_bird()

    def _spawn_pipe(self) -> None:
        margin = 95
        ground_y = HEIGHT - GROUND_HEIGHT
        min_gap_y = margin + PIPE_GAP / 2
        max_gap_y = ground_y - margin - PIPE_GAP / 2
        gap_y = random.uniform(min_gap_y, max_gap_y)
        x = WIDTH + PIPE_WIDTH

        top_bottom = gap_y - PIPE_GAP / 2
        bottom_top = gap_y + PIPE_GAP / 2

        top_id = self.canvas.create_rectangle(
            x,
            0,
            x + PIPE_WIDTH,
            top_bottom,
            fill="#58b947",
            outline="#2f6e2d",
            width=3,
        )
        bottom_id = self.canvas.create_rectangle(
            x,
            bottom_top,
            x + PIPE_WIDTH,
            ground_y,
            fill="#58b947",
            outline="#2f6e2d",
            width=3,
        )
        self.pipes.append(PipePair(x=x, gap_y=gap_y, top_id=top_id, bottom_id=bottom_id))

    def _move_pipes(self) -> None:
        for pipe in self.pipes:
            pipe.x -= PIPE_SPEED
            self.canvas.move(pipe.top_id, -PIPE_SPEED, 0)
            self.canvas.move(pipe.bottom_id, -PIPE_SPEED, 0)

            if not pipe.scored and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.scored = True
                self.score += 1
                self.best_score = max(self.best_score, self.score)
                self.canvas.itemconfigure(self.score_id, text=str(self.score))
                self.canvas.itemconfigure(self.best_id, text=f"Mejor: {self.best_score}")

        alive: list[PipePair] = []
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH >= -10:
                alive.append(pipe)
            else:
                self.canvas.delete(pipe.top_id)
                self.canvas.delete(pipe.bottom_id)
        self.pipes = alive

    def _collides(self) -> bool:
        bird_left = BIRD_X - BIRD_RADIUS + 3
        bird_right = BIRD_X + BIRD_RADIUS - 3
        bird_top = self.bird_y - BIRD_RADIUS + 3
        bird_bottom = self.bird_y + BIRD_RADIUS - 3
        ground_y = HEIGHT - GROUND_HEIGHT

        if bird_top <= 0 or bird_bottom >= ground_y:
            return True

        for pipe in self.pipes:
            pipe_left = pipe.x
            pipe_right = pipe.x + PIPE_WIDTH

            if bird_right < pipe_left or bird_left > pipe_right:
                continue

            gap_top = pipe.gap_y - PIPE_GAP / 2
            gap_bottom = pipe.gap_y + PIPE_GAP / 2
            if bird_top < gap_top or bird_bottom > gap_bottom:
                return True

        return False

    def _finish_game(self) -> None:
        self.running = False
        self.game_over = True
        self.best_score = max(self.best_score, self.score)
        self.canvas.itemconfigure(self.best_id, text=f"Mejor: {self.best_score}")
        self.canvas.itemconfigure(
            self.message_id,
            text=(
                f"GAME OVER\n\nPuntuación: {self.score}\n"
                "Espacio o clic para reintentar"
            ),
        )

    def _update(self) -> None:
        self.frame += 1
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        if self.frame == 1 or self.frame % SPAWN_EVERY == 0:
            self._spawn_pipe()

        self._move_pipes()
        self._draw_bird()

        if self._collides():
            self._finish_game()

    def _game_loop(self) -> None:
        if self.running:
            self._update()
        self.root.after(FRAME_MS, self._game_loop)


def main() -> None:
    root = tk.Tk()
    FlappyGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
