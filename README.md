# Car License Plate Detector (YOLOv8, CPU-friendly)

This project runs a trained YOLOv8 model to detect car license plates. It includes a single interactive script that supports image, video, or webcam input and is optimized for CPU-only systems.

## Project layout
- `best.pt` — your trained weights (place this in the project root)
- `run_detect.py` — unified interactive detection script (image/video/webcam)
- `datasets/plates_yolo/` — dataset used during development (optional for inference)
- `.venv/` — local virtual environment (created on your machine)

## Quick start on Windows (CPU)
1) Open Command Prompt in the project folder:
```
cd /d D:\car_lic_plate_detector
```
2) Create and activate a virtual environment:
```
python -m venv .venv
.venv\Scripts\activate
```
3) Install CPU-only PyTorch (official CPU wheels):
```
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```
4) Install Ultralytics and OpenCV:
```
pip install ultralytics opencv-python
```
5) Ensure your weights file exists:
- Preferred: `best.pt` in the project root
- Alternative: `weights/best_colab.pt`

6) Run detection (interactive):
```
python run_detect.py
```
- Choose `1` for image, `2` for video, or `3` for webcam
- For image/video, paste the full path when prompted
- A window will display results. Press any key (`image`) or `q` (`video`/`webcam`) to close
- Outputs are saved next to your input as `<name>_det.jpg` or `<name>_det.mp4`

## Running on another PC
Repeat the same steps on the target machine:

firstly extract the zip file to LocalDisk D.
then, run these commands in command prompt one by one.
```
cd /d D:\car_lic_plate_detector
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python
python run_detect.py
```
Copy `best.pt` with the project if it’s not already present.

## Linux/macOS (optional)
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python
python run_detect.py
```

## Troubleshooting
- `ModuleNotFoundError: No module named 'cv2'` → `pip install opencv-python` in the venv
- No GUI window (remote/locked-down environments) → ask for a save-only version that skips `cv2.imshow`
- Slow on CPU → reduce `imgsz` inside the script (e.g., 512 or 416)

## Notes
- Training was done in Google Colab (GPU). Only `best.pt` is needed here for inference.
