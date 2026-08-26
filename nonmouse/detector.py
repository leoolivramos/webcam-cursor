import os
import urllib.request
from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(_MODELS_DIR, "hand_landmarker.task")


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


def _ensure_model() -> None:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Baixando modelo mediapipe em {MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
        print("Download concluido.")


def _distance(a: Landmark, b: Landmark) -> float:
    return float(np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y])))


class HandDetector:
    """Encapsula o HandLandmarker do mediapipe com suporte a IMAGE e VIDEO."""

    def __init__(self, mode: str = "IMAGE", confidence: float = 0.8, num_hands: int = 2):
        _ensure_model()
        running_mode = (
            mp_vision.RunningMode.VIDEO
            if mode == "VIDEO"
            else mp_vision.RunningMode.IMAGE
        )
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._detector = HandLandmarker.create_from_options(options)
        self._mode = mode

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int = 0) -> list[list[Landmark]]:
        """Retorna lista de maos, cada mao e uma lista de 21 Landmarks."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self._mode == "VIDEO":
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._detector.detect(mp_image)
        return [
            [Landmark(p.x, p.y, p.z) for p in hand]
            for hand in result.hand_landmarks
        ]

    def get_closest_hand(self, hands: list[list[Landmark]]) -> list[Landmark] | None:
        """Retorna a mao mais proxima da camera (maior tamanho aparente no frame)."""
        if not hands:
            return None
        return max(hands, key=lambda hand: _distance(hand[0], hand[9]))

    def distance(self, a: Landmark, b: Landmark) -> float:
        return _distance(a, b)

    def close(self) -> None:
        self._detector.close()
