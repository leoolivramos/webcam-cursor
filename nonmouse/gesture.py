import math
import time

import cv2
import numpy as np
from pynput.mouse import Button, Controller

from .detector import HandDetector, Landmark
from .drawing import draw_circle, draw_landmarks
from .screen import get_virtual_screen

_PINCH_ON = 0.35
_PINCH_OFF = 0.48


class GestureController:
    """Traduz landmarks da mao em acoes de mouse com alta precisao e estabilidade."""

    def __init__(self, sensitivity: float = 30.0, smoothing: int = 3):
        self._mouse = Controller()
        self._sensitivity = sensitivity
        self._smoothing = smoothing

        virt_x, virt_y, virt_w, virt_h = get_virtual_screen()
        self._virt_x = virt_x
        self._virt_y = virt_y
        self._screen_w = virt_x + virt_w
        self._screen_h = virt_y + virt_h

        self._curr_x: float = 0.0
        self._curr_y: float = 0.0
        self._pre_x: float = 0.0
        self._pre_y: float = 0.0
        self._pre_raw_x: float = 0.0
        self._pre_raw_y: float = 0.0
        self._initialized = False

        self._now_click = 0
        self._pre_click = 0
        self._click_start = float("inf")

        # Trava de seguranca do mouse
        self._locked: bool = True
        self._unlocked_hand_id: str | None = None
        self._last_action_time: float = 0.0
        self._inactivity_timeout: float = 10.0  # 10 segundos sem acao trave o mouse
        self._rotation_history: dict[str, list[tuple[float, float]]] = {}

    def trigger(self, hand_id: str = "Right") -> None:
        """Destrava o mouse para a mao especificada que realizou a rotacao."""
        self._locked = False
        self._unlocked_hand_id = hand_id
        self._last_action_time = time.perf_counter()

    def lock(self) -> None:
        """Trava o mouse."""
        self._locked = True
        self._unlocked_hand_id = None

    def is_locked(self) -> bool:
        return self._locked

    def get_unlocked_hand(self) -> str | None:
        return self._unlocked_hand_id

    def get_remaining_unlocked_time(self) -> float:
        if self._locked:
            return 0.0
        elapsed = time.perf_counter() - self._last_action_time
        return max(0.0, self._inactivity_timeout - elapsed)

    @staticmethod
    def _palm_up_score(lm: list[Landmark]) -> float:
        """Pontua a probabilidade de a mão estar com a palma levantada.
        Valores positivos indicam dedos mais altos que o pulso (palma voltada para a câmera).
        """
        wrist = lm[0]
        tip_ys = [lm[i].y for i in (8, 12, 16, 20)]
        knuckle_ys = [lm[i].y for i in (5, 9, 13, 17)]
        avg_tip_y = float(np.mean(tip_ys)) if tip_ys else wrist.y
        avg_knuckle_y = float(np.mean(knuckle_ys)) if knuckle_ys else wrist.y
        return float((wrist.y - avg_tip_y) + 0.35 * (avg_knuckle_y - avg_tip_y))

    def select_active_hand(
        self,
        hands_with_label: list[tuple[list[Landmark], str]],
    ) -> tuple[list[Landmark], str] | None:
        """Seleciona a única mão ativa, priorizando a palma levantada e mantendo a mão já desbloqueada."""
        if not hands_with_label:
            return None

        if self._unlocked_hand_id is not None:
            preferred = [
                (lms, label)
                for lms, label in hands_with_label
                if label == self._unlocked_hand_id
            ]
            if preferred:
                return max(preferred, key=lambda pair: self._palm_up_score(pair[0]))

        palm_candidates = [
            (lms, label)
            for lms, label in hands_with_label
            if self._palm_up_score(lms) > 0.02
        ]
        if palm_candidates:
            return max(palm_candidates, key=lambda pair: self._palm_up_score(pair[0]))

        return None

    @staticmethod
    def calc_hand_angle(lm: list[Landmark]) -> float:
        """Calcula o angulo de orientacao da mao (vetor pulso lm[0] ate base do medio lm[9])."""
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return math.atan2(dy, dx)

    def check_rotation_gesture(self, hand_id: str, lm: list[Landmark], now: float) -> bool:
        """Detecta se a mao efetuou movimento de rotacao/torcao acentuado."""
        angle = self.calc_hand_angle(lm)
        if hand_id not in self._rotation_history:
            self._rotation_history[hand_id] = []

        history = self._rotation_history[hand_id]
        history.append((now, angle))

        # Manter historico do ultimo 1.0 segundo
        self._rotation_history[hand_id] = [(t, a) for t, a in history if now - t <= 1.0]
        history = self._rotation_history[hand_id]

        if len(history) < 4:
            return False

        # Calcula a amplitude de rotacao acumulada no intervalo
        total_rot = 0.0
        for i in range(1, len(history)):
            prev_a = history[i - 1][1]
            curr_a = history[i][1]
            diff = (curr_a - prev_a + math.pi) % (2 * math.pi) - math.pi
            total_rot += abs(diff)

        # Se girou acumulado > 0.75 rad (aprox 43 graus) em ate 1 segundo
        if total_rot > 0.75:
            self._rotation_history[hand_id] = []
            return True
        return False

    def process(self, landmarks: list[Landmark] | None = None, image: np.ndarray | None = None,
                image_width: int = 0, image_height: int = 0, hand_id: str = "Right") -> None:
        """Processa landmarks e executa acoes de mouse sem tremores e com alta precisao."""
        now = time.perf_counter()

        # VERIFICA TIMEOUT DE INATIVIDADE NO INÍCIO/CONFIGURAÇÃO (INDEPENDENTE DE TER MÃO OU NÃO)
        if not self._locked:
            if now - self._last_action_time > self._inactivity_timeout:
                self.lock()

        # Se não houver mão no frame (landmarks é None ou lista vazia)
        if not landmarks:
            if image is not None:
                if self._locked:
                    try:
                        cv2.putText(image, "MOUSE TRAVADO - Gire a mao para destravar", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except Exception:
                        pass
                else:
                    remaining = self.get_remaining_unlocked_time()
                    try:
                        cv2.putText(image, f"MOUSE DESTRAVADO [{self._unlocked_hand_id}] - Trava em: {remaining:.1f}s", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception:
                        pass
            return

        if image is not None:
            draw_landmarks(image, landmarks, image_width, image_height)
        lm = landmarks

        # 0. DETECCAO DE ROTACAO PARA DESTRAVAR
        rotation_detected = self.check_rotation_gesture(hand_id, lm, now)
        if rotation_detected:
            self.trigger(hand_id)

        # SE ESTIVER TRAVADO, NAO MOVER E EXIBIR INDICADOR VISUAL DE TRAVA
        if self._locked:
            if image is not None:
                try:
                    cv2.putText(image, "MOUSE TRAVADO - Gire a mao para destravar", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception:
                    pass
            return

        # SE ESTIVER DESTRAVADO, GARANTE QUE APENAS A MAO QUE DESTRAVOU PODE MEXER
        if hand_id != self._unlocked_hand_id:
            if image is not None:
                try:
                    cv2.putText(image, f"Mao ignorada (apenas {self._unlocked_hand_id} controla)", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                except Exception:
                    pass
            return

        # Tamanho de referencia da mao (pulso lm[0] ate a base do medio lm[9])
        hand_size = self._safe_distance(lm[0], lm[9])

        # Distancia entre o polegar (lm[4]) e o indicador (ponta lm[8] ou junta lm[6])
        pinch_dist = min(
            self._safe_distance(lm[4], lm[8]),
            self._safe_distance(lm[4], lm[6])
        ) / hand_size

        raw_x = lm[8].x
        raw_y = lm[8].y

        if not self._initialized:
            self._curr_x = raw_x
            self._curr_y = raw_y
            self._pre_x = raw_x
            self._pre_y = raw_y
            self._pre_raw_x = raw_x
            self._pre_raw_y = raw_y
            self._initialized = True

        # 1. HISTERESE DE CLIQUE (Evita disparos falsos e tremores de clique)
        if self._now_click == 0:
            if pinch_dist < _PINCH_ON:
                self._now_click = 1
        else:
            if pinch_dist > _PINCH_OFF:
                self._now_click = 0

        # 2. FILTRO ADAPTATIVO DE MOVIMENTO (Elimina tremor quando a mao esta parada)
        dist_moved = float(math.hypot(raw_x - self._pre_raw_x, raw_y - self._pre_raw_y))
        self._pre_raw_x, self._pre_raw_y = raw_x, raw_y

        # Atualiza timestamp de acao quando houver movimento significativo ou clique
        if dist_moved > 0.003 or self._now_click == 1 or rotation_detected:
            self._last_action_time = now

        # Deadzone: Pequenos ruídos de câmera não movem o cursor
        if dist_moved < 0.0015:
            raw_x = self._curr_x
            raw_y = self._curr_y
            alpha = 1.0
        else:
            # Alpha dinamico: suave para movimentos lentos, rapido para movimentos bruscos
            alpha = float(np.clip(dist_moved * 30.0, 0.08, 0.95))

        self._curr_x += alpha * (raw_x - self._curr_x)
        self._curr_y += alpha * (raw_y - self._curr_y)

        # 3. TRAVA DE POSICAO DURANTE O CLIQUE (Click Precision Lock)
        # Quando a pessoa clica, evita que a trepidação dos dedos desloque o ponteiro
        if self._now_click == 1:
            dx, dy = 0.0, 0.0
            if image is not None:
                draw_circle(image, lm[8].x * image_width, lm[8].y * image_height, 20, (0, 250, 250))
        else:
            dx = self._sensitivity * (self._curr_x - self._pre_x) * image_width
            dy = self._sensitivity * (self._curr_y - self._pre_y) * image_height

        self._pre_x, self._pre_y = self._curr_x, self._curr_y

        try:
            pos = self._mouse.position
            px, py = pos if pos is not None else (0, 0)
        except Exception:
            px, py = 0, 0
        dx = max(self._virt_x - px, min(self._screen_w - px, dx))
        dy = max(self._virt_y - py, min(self._screen_h - py, dy))

        # 5. MOVIMENTACAO DO CURSOR
        if self._now_click == 0:
            if np.hypot(dx, dy) > 0.5:
                try:
                    self._mouse.move(dx, dy)
                except Exception:
                    pass
            if image is not None:
                draw_circle(image, lm[8].x * image_width, lm[8].y * image_height, 8, (250, 0, 0))

        # Ações do botão esquerdo do mouse (Pressionar / Soltar)
        if self._now_click == 1 and self._now_click != self._pre_click:
            self._mouse.press(Button.left)

        if self._now_click == 0 and self._now_click != self._pre_click:
            self._mouse.release(Button.left)

        self._pre_click = self._now_click

        if not self._locked:
            if time.perf_counter() - self._last_action_time > self._inactivity_timeout:
                self.lock()

        # Exibe status destravado com contador do tempo restante de inatividade no final do process
        if image is not None:
            if self._locked:
                try:
                    cv2.putText(image, "MOUSE TRAVADO - Gire a mao para destravar", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception:
                    pass
            else:
                remaining = self.get_remaining_unlocked_time()
                try:
                    cv2.putText(image, f"MOUSE DESTRAVADO [{hand_id}] - Trava em: {remaining:.1f}s", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception:
                    pass

    def reset(self) -> None:
        self._initialized = False

    @staticmethod
    def _safe_distance(a: Landmark, b: Landmark) -> float:
        return float(np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))) + 1e-6

