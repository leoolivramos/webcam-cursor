import time

import numpy as np
from pynput.mouse import Button, Controller

from .detector import HandDetector, Landmark
from .drawing import draw_circle, draw_landmarks
from .screen import get_virtual_screen

_CLICK_DIST = 0.65


class GestureController:
    """Traduz landmarks da mao em acoes de mouse."""

    def __init__(self, sensitivity: float = 3.0, smoothing: int = 3):
        self._mouse = Controller()
        self._sensitivity = sensitivity
        self._smoothing = smoothing

        virt_x, virt_y, virt_w, virt_h = get_virtual_screen()
        self._virt_x = virt_x
        self._virt_y = virt_y
        self._screen_w = virt_x + virt_w
        self._screen_h = virt_y + virt_h

        self._pre_x: float = 0.0
        self._pre_y: float = 0.0
        self._buf_x: list[float] = []
        self._buf_y: list[float] = []
        self._initialized = False

        self._now_click = 0
        self._pre_click = 0
        self._right_click = 0
        self._pre_right = 0
        self._double_click = 0
        self._scroll_active = 0
        self._click_hold = 0
        self._k = 0
        self._h = 0
        self._click_start = float("inf")
        self._double_start = float("inf")

    def process(self, landmarks: list[Landmark], image: np.ndarray,
                image_width: int, image_height: int) -> None:
        """Processa landmarks e executa acoes de mouse. Desenha feedback na imagem."""
        draw_landmarks(image, landmarks, image_width, image_height)

        lm = landmarks
        base = self._safe_distance(lm[0], lm[1])
        spread = self._safe_distance(lm[8], lm[12]) / base
        click_dist = self._safe_distance(lm[4], lm[6]) / base

        if not self._initialized:
            self._pre_x = lm[8].x
            self._pre_y = lm[8].y
            self._buf_x = [lm[8].x] * self._smoothing
            self._buf_y = [lm[8].y] * self._smoothing
            self._initialized = True

        self._buf_x.append(lm[8].x)
        self._buf_y.append(lm[8].y)
        if len(self._buf_x) > self._smoothing:
            self._buf_x.pop(0)
            self._buf_y.pop(0)

        now_x = sum(self._buf_x) / self._smoothing
        now_y = sum(self._buf_y) / self._smoothing
        dx = self._sensitivity * (now_x - self._pre_x) * image_width + 0.5
        dy = self._sensitivity * (now_y - self._pre_y) * image_height + 0.5
        self._pre_x, self._pre_y = now_x, now_y

        px, py = self._mouse.position
        dx = max(self._virt_x - px, min(self._screen_w - px, dx))
        dy = max(self._virt_y - py, min(self._screen_h - py, dy))

        self._now_click = 1 if click_dist < _CLICK_DIST else 0

        if self._now_click == 1:
            draw_circle(image, lm[8].x * image_width, lm[8].y * image_height, 20, (0, 250, 250))

        if np.abs(dx) > 5 and np.abs(dy) > 5:
            self._k = 0

        if self._now_click == 1 and np.abs(dx) < 5 and np.abs(dy) < 5:
            if self._k == 0:
                self._click_start = time.perf_counter()
                self._k += 1
            if time.perf_counter() - self._click_start > 1.5:
                self._right_click = 1
                draw_circle(image, lm[8].x * image_width, lm[8].y * image_height, 20, (0, 0, 250))
        else:
            self._right_click = 0

        is_scroll = lm[8].y - lm[5].y > -0.06
        if spread >= _CLICK_DIST and not is_scroll:
            try:
                self._mouse.move(dx, dy)
            except Exception:
                pass
            draw_circle(image, lm[8].x * image_width, lm[8].y * image_height, 8, (250, 0, 0))

        if self._now_click == 1 and self._now_click != self._pre_click:
            if self._h == 1:
                self._h = 0
            else:
                self._mouse.press(Button.left)

        if self._now_click == 0 and self._now_click != self._pre_click:
            self._mouse.release(Button.left)
            self._k = 0
            if self._double_click == 0:
                self._double_start = time.perf_counter()
                self._double_click += 1
            if 10 * (time.perf_counter() - self._double_start) > 5 and self._double_click == 1:
                self._mouse.click(Button.left, 2)
                self._double_click = 0

        if self._right_click == 1 and self._right_click != self._pre_right:
            self._mouse.press(Button.right)
            self._mouse.release(Button.right)
            self._h = 1

        if is_scroll:
            self._mouse.scroll(0, -dy / 50)
            draw_circle(image, lm[8].x * image_width, lm[8].y * image_height, 20, (0, 0, 0))

        self._pre_click = self._now_click
        self._pre_right = self._right_click

    def reset(self) -> None:
        self._initialized = False

    @staticmethod
    def _safe_distance(a: Landmark, b: Landmark) -> float:
        return float(np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))) + 1e-6
