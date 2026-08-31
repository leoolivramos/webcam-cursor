import time
import tkinter as tk

import cv2

from nonmouse.camera import open_camera
from nonmouse.detector import HandDetector
from nonmouse.gesture import GestureController


def setup_ui() -> tuple[int, int, float]:
    root = tk.Tk()
    root.title("NonMouse — Setup")
    root.geometry("300x320")

    cam_var = tk.IntVar()
    place_var = tk.IntVar()
    sens_var = tk.IntVar(value=30)

    tk.Label(root, text="Camera").grid(row=0, column=0, columnspan=6, pady=(10, 0))
    for i in range(3):
        tk.Radiobutton(root, text=f"Device{i}", value=i,
                       variable=cam_var).grid(row=1, column=i * 2)

    tk.Label(root, text="How to place").grid(row=2, column=0, columnspan=6, pady=(10, 0))
    for i, label in enumerate(["Normal", "Above", "Behind"]):
        tk.Radiobutton(root, text=label, value=i,
                       variable=place_var).grid(row=3, column=i * 2)

    tk.Label(root, text="Sensitivity").grid(row=4, column=0, columnspan=6, pady=(10, 0))
    tk.Scale(root, orient="h", from_=1, to=100,
             variable=sens_var).grid(row=5, column=0, columnspan=6)

    tk.Button(root, text="Continue",
              command=root.destroy).grid(row=6, column=0, columnspan=6, pady=15)
    root.mainloop()

    return cam_var.get(), place_var.get(), sens_var.get() / 10


def main(cam_index: int, mode: int, sensitivity: float) -> None:
    cap = open_camera(cam_index, width=1280, height=720)
    cfps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    ran = max(cfps // 10, 1)

    detector = HandDetector(mode="IMAGE", confidence=0.7, num_hands=1)
    controller = GestureController(sensitivity=sensitivity, smoothing=ran)

    window = "NonMouse"
    cv2.namedWindow(window)

    try:
        while cap.isOpened():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if mode == 1:
                frame = cv2.flip(frame, 0)
            elif mode == 2:
                frame = cv2.flip(frame, 1)
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_with_label = detector.detect_with_handedness(rgb)

            if hands_with_label:
                for hand_lms, hand_id in hands_with_label:
                    controller.process(hand_lms, frame, w, h, hand_id=hand_id)
            else:
                controller.reset()
                controller.process(None, frame, w, h)
                cv2.putText(frame, "Aguardando mao...", (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            elapsed = time.perf_counter() - t0
            fps = str(int(1 / elapsed)) if elapsed > 0 else "?"
            cv2.putText(frame, f"FPS:{fps}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow(window, cv2.resize(frame, dsize=None, fx=0.4, fy=0.4))
            if (cv2.waitKey(1) & 0xFF == 27) or \
                    cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) == 0:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_index, mode, sensitivity = setup_ui()
    main(cam_index, mode, sensitivity)
