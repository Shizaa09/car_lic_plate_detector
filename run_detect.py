import sys
from pathlib import Path
from typing import Optional, Union

import cv2
from ultralytics import YOLO


def choose_mode() -> str:
    print("Choose mode:")
    print("  1) Image")
    print("  2) Video")
    print("  3) Webcam")
    choice = input("Enter 1/2/3: ").strip()
    if choice not in {"1", "2", "3"}:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    return choice


def prompt_path(prompt: str) -> Path:
    user_input = input(prompt).strip().strip('"')
    path = Path(user_input)
    if not path.exists() or not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)
    return path


def resolve_weights(project_root: Path) -> Path:
    weights = project_root / "best.pt"
    if weights.exists():
        return weights
    alt = project_root / "weights" / "best_colab.pt"
    if alt.exists():
        return alt
    print("No weights found. Place your trained weights as 'best.pt' in the project root.")
    sys.exit(1)


def run_image(model: YOLO, image_path: Path) -> None:
    results = model.predict(
        source=str(image_path), device="cpu", save=False, conf=0.25, imgsz=640, verbose=False
    )
    if not results:
        print("No result returned by model.")
        sys.exit(1)
    frame = results[0].plot()
    cv2.imshow(f"Detections - {image_path.name}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    out_path = image_path.with_name(image_path.stem + "_det.jpg")
    cv2.imwrite(str(out_path), frame)
    print(f"Saved: {out_path}")


def open_writer_like(cap: cv2.VideoCapture, out_path: Path) -> Optional[cv2.VideoWriter]:
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    except Exception:
        return None


def run_video(model: YOLO, source: Union[int, Path]) -> None:
    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None
    out_path: Optional[Path] = None

    if isinstance(source, Path):
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"Failed to open video: {source}")
            sys.exit(1)
        out_path = source.with_name(source.stem + "_det.mp4")
        writer = open_writer_like(cap, out_path)

    print("Press 'q' to quit.")
    for result in model.predict(
        source=str(source) if isinstance(source, Path) else source,
        stream=True,
        device="cpu",
        imgsz=640,
        conf=0.25,
        verbose=False,
    ):
        frame = result.plot()

        if writer is None and cap is None:
            # Webcam: create writer on first frame
            out_dir = Path("runs") / "predict_video_cpu"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "webcam_det.mp4"
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Detections (press q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if out_path is not None:
        print(f"Saved annotated video to: {out_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    weights = resolve_weights(project_root)
    model = YOLO(str(weights))

    mode = choose_mode()
    if mode == "1":
        img = prompt_path("Enter image path: ")
        run_image(model, img)
    elif mode == "2":
        vid = prompt_path("Enter video path: ")
        run_video(model, vid)
    else:
        # Webcam
        index_str = input("Enter webcam index (default 0): ").strip() or "0"
        if not index_str.isdigit():
            print("Invalid webcam index.")
            sys.exit(1)
        run_video(model, int(index_str))


if __name__ == "__main__":
    main()


