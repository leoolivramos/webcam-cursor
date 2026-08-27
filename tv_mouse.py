import argparse
import time

import cv2

from nonmouse.camera import ThreadedCamera
from nonmouse.detector import HandDetector
from nonmouse.gesture import GestureController

parser = argparse.ArgumentParser(description="NonMouse — modo otimizado sem interface grafica")
parser.add_argument("--camera", type=int, default=0, help="Indice da camera (padrao: 0)")
parser.add_argument("--kando", type=float, default=15.0, help="Sensibilidade do cursor (padrao: 30)")
parser.add_argument("--smoothing", type=int, default=3, help="Suavizacao do cursor (padrao: 3)")
parser.add_argument("--headless", action="store_true", help="Desativa a janela de video")
args = parser.parse_args()


def main() -> None:
    cam = ThreadedCamera(index=args.camera, width=1280, height=720).start()
    detector = HandDetector(mode="VIDEO", confidence=0.4, num_hands=4)
    controller = GestureController(sensitivity=args.kando, smoothing=args.smoothing)   

    print("NonMouse iniciado. Pressione Ctrl+C para encerrar.")

    try:    
        while True:
            t0 = time.perf_counter()
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(t0 * 1000)

            hands_with_label = detector.detect_with_handedness(rgb, timestamp_ms)
            if hands_with_label:
                for hand_lms, hand_id in hands_with_label:
                    controller.process(hand_lms, frame, w, h, hand_id=hand_id)
            else:
                controller.reset()
                controller.process(None, frame, w, h)

            if not args.headless:
                elapsed = time.perf_counter() - t0
                fps = str(int(1 / elapsed)) if elapsed > 0 else "?"
                cv2.putText(frame, f"FPS:{fps}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("NonMouse TV", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
