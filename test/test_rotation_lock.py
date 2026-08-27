import math
import time
import numpy as np

from nonmouse.detector import Landmark
from nonmouse.gesture import GestureController


def create_hand(angle_rad: float = -math.pi / 2, wrist_x: float = 0.5, wrist_y: float = 0.8) -> list[Landmark]:
    """Cria conjunto de 21 landmarks de teste orientados no angulo especificado."""
    length = 0.2
    wrist = Landmark(wrist_x, wrist_y, 0.0)
    mcp_x = wrist_x + length * math.cos(angle_rad)
    mcp_y = wrist_y + length * math.sin(angle_rad)
    
    landmarks = [wrist] + [Landmark(wrist_x, wrist_y, 0.0)] * 20
    landmarks[9] = Landmark(mcp_x, mcp_y, 0.0)
    landmarks[8] = Landmark(mcp_x, mcp_y - 0.05, 0.0)
    landmarks[5] = Landmark(mcp_x - 0.02, mcp_y, 0.0)
    landmarks[4] = Landmark(wrist_x - 0.05, wrist_y - 0.05, 0.0)
    return landmarks


def test_initial_state_is_locked():
    controller = GestureController()
    assert controller.is_locked() is True
    assert controller.get_unlocked_hand() is None


def test_explicit_trigger_unlocks():
    controller = GestureController()
    controller.trigger(hand_id="Right")
    assert controller.is_locked() is False
    assert controller.get_unlocked_hand() == "Right"


def test_rotation_gesture_triggers_unlock():
    controller = GestureController()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    angles = np.linspace(-math.pi / 2, 0, 8)
    for a in angles:
        hand = create_hand(angle_rad=float(a))
        controller.process(hand, dummy_img, 640, 480, hand_id="Right")
        time.sleep(0.02)

    assert controller.is_locked() is False
    assert controller.get_unlocked_hand() == "Right"


def test_other_hand_ignored_when_unlocked():
    controller = GestureController()
    controller.trigger(hand_id="Right")

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    left_hand = create_hand(angle_rad=-math.pi / 2, wrist_x=0.2, wrist_y=0.5)

    controller.process(left_hand, dummy_img, 640, 480, hand_id="Left")
    assert controller.get_unlocked_hand() == "Right"


def test_inactivity_timeout_relocks():
    controller = GestureController()
    controller._inactivity_timeout = 0.2
    controller.trigger(hand_id="Right")

    assert controller.is_locked() is False
    time.sleep(0.3)

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    hand = create_hand()
    controller.process(hand, dummy_img, 640, 480, hand_id="Right")

    assert controller.is_locked() is True
    assert controller.get_unlocked_hand() == "None" or controller.get_unlocked_hand() is None


def test_no_hand_inactivity_relocks():
    controller = GestureController()
    controller._inactivity_timeout = 0.2
    controller.trigger(hand_id="Right")

    assert controller.is_locked() is False
    time.sleep(0.3)

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Processa sem nenhuma mão no frame (landmarks = None)
    controller.process(None, dummy_img, 640, 480)

    assert controller.is_locked() is True
    assert controller.get_unlocked_hand() is None


if __name__ == "__main__":
    test_initial_state_is_locked()
    print("[PASS] test_initial_state_is_locked")
    test_explicit_trigger_unlocks()
    print("[PASS] test_explicit_trigger_unlocks")
    test_rotation_gesture_triggers_unlock()
    print("[PASS] test_rotation_gesture_triggers_unlock")
    test_other_hand_ignored_when_unlocked()
    print("[PASS] test_other_hand_ignored_when_unlocked")
    test_inactivity_timeout_relocks()
    print("[PASS] test_inactivity_timeout_relocks")
    test_no_hand_inactivity_relocks()
    print("[PASS] test_no_hand_inactivity_relocks")
    print("ALL TESTS PASSED SUCCESSFULLY!")
