"""
PlateNet AI - License Plate Detection System
============================================

This module provides a comprehensive license plate detection system using YOLOv8.
It supports detection on images, videos, and live webcam feeds with real-time
visualization and output saving capabilities.

Author: [Your Name]
Date: [Current Date]
Version: 1.0.0

Dependencies:
    - ultralytics: YOLOv8 model implementation
    - opencv-python: Computer vision operations
    - pathlib: Cross-platform path handling
    - typing: Type hints for better code documentation

Usage:
    python run_detect.py
    
Features:
    - Multi-modal detection (image, video, webcam)
    - Real-time visualization with OpenCV
    - Automatic output file generation
    - Flexible model weight resolution
    - User-friendly interactive interface
"""

import sys
from pathlib import Path
from typing import Optional, Union

import cv2
from ultralytics import YOLO


def choose_mode() -> str:
    """
    Interactive mode selection for the detection system.
    
    Presents the user with three detection modes:
    - Image: Process a single image file
    - Video: Process a video file
    - Webcam: Real-time detection from webcam feed
    
    Returns:
        str: The selected mode ("1", "2", or "3")
        
    Raises:
        SystemExit: If an invalid choice is made
        
    Note:
        This function handles user input validation and provides
        clear error messaging for invalid selections.
    """
    print("Choose mode:")
    print("  1) Image")
    print("  2) Video")
    print("  3) Webcam")
    
    # Get user input and strip whitespace
    choice = input("Enter 1/2/3: ").strip()
    
    # Validate user input - only accept valid mode selections
    if choice not in {"1", "2", "3"}:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    return choice


def prompt_path(prompt: str) -> Path:
    """
    Prompt user for a file path and validate its existence.
    
    This function handles user input for file paths, including proper
    handling of quoted paths and validation of file existence.
    
    Args:
        prompt (str): The prompt message to display to the user
        
    Returns:
        Path: A validated Path object pointing to an existing file
        
    Raises:
        SystemExit: If the specified file doesn't exist or isn't a file
        
    Note:
        The function strips both whitespace and quotes from user input
        to handle various input formats gracefully.
    """
    # Get user input, strip whitespace and quotes for flexibility
    user_input = input(prompt).strip().strip('"')
    path = Path(user_input)
    
    # Validate that the path exists and is a file (not a directory)
    if not path.exists() or not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)
    
    return path


def resolve_weights(project_root: Path) -> Path:
    """
    Resolve the path to the trained model weights file.
    
    This function searches for trained model weights in multiple locations
    with a fallback hierarchy to ensure compatibility with different
    deployment scenarios.
    
    Args:
        project_root (Path): The root directory of the project
        
    Returns:
        Path: Path to the found weights file
        
    Raises:
        SystemExit: If no valid weights file is found
        
    Search Order:
        1. best.pt in project root (primary location)
        2. weights/best_colab.pt (Google Colab compatibility)
        
    Note:
        The function prioritizes the standard 'best.pt' location but
        provides fallback support for Colab-trained models.
    """
    # Primary weights location - standard YOLO output
    weights = project_root / "best.pt"
    if weights.exists():
        return weights
    
    # Alternative location for Colab-trained models
    alt = project_root / "weights" / "best_colab.pt"
    if alt.exists():
        return alt
    
    # No weights found - provide clear error message
    print("No weights found. Place your trained weights as 'best.pt' in the project root.")
    sys.exit(1)


def run_image(model: YOLO, image_path: Path) -> None:
    """
    Perform license plate detection on a single image.
    
    This function processes an image through the YOLO model, displays the
    results with bounding boxes, and saves the annotated image to disk.
    
    Args:
        model (YOLO): The trained YOLO model for license plate detection
        image_path (Path): Path to the input image file
        
    Raises:
        SystemExit: If the model fails to process the image
        
    Note:
        The function uses CPU processing for compatibility and sets
        confidence threshold to 0.25 for balanced detection sensitivity.
        Output images are saved with "_det" suffix to avoid overwriting originals.
    """
    # Run inference on the image with optimized parameters
    results = model.predict(
        source=str(image_path),  # Convert Path to string for YOLO
        device="cpu",            # Use CPU for broad compatibility
        save=False,              # Don't auto-save (we'll handle it manually)
        conf=0.25,               # Confidence threshold for detections
        imgsz=640,               # Input image size (YOLO standard)
        verbose=False            # Suppress verbose output
    )
    
    # Validate that the model returned results
    if not results:
        print("No result returned by model.")
        sys.exit(1)
    
    # Generate annotated image with bounding boxes and labels
    frame = results[0].plot()
    
    # Display the results in a window
    cv2.imshow(f"Detections - {image_path.name}", frame)
    cv2.waitKey(0)  # Wait for key press to close window
    cv2.destroyAllWindows()
    
    # Save the annotated image with detection suffix
    out_path = image_path.with_name(image_path.stem + "_det.jpg")
    cv2.imwrite(str(out_path), frame)
    print(f"Saved: {out_path}")


def open_writer_like(cap: cv2.VideoCapture, out_path: Path) -> Optional[cv2.VideoWriter]:
    """
    Create a VideoWriter with properties matching the input VideoCapture.
    
    This utility function creates an output video writer that matches the
    properties (resolution, frame rate) of the input video capture.
    This ensures consistent video output quality and compatibility.
    
    Args:
        cap (cv2.VideoCapture): Input video capture object
        out_path (Path): Path where the output video will be saved
        
    Returns:
        Optional[cv2.VideoWriter]: VideoWriter object if successful, None if failed
        
    Note:
        The function uses MP4V codec for broad compatibility and creates
        parent directories as needed. Falls back gracefully on errors.
    """
    try:
        # Extract video properties from the input capture
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default to 25 FPS if unavailable
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use MP4V codec for broad compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create and return the video writer
        return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    except Exception:
        # Return None if video writer creation fails
        return None


def run_video(model: YOLO, source: Union[int, Path]) -> None:
    """
    Perform license plate detection on video or webcam feed.
    
    This function handles both video file processing and live webcam detection.
    It processes frames in real-time, displays results, and optionally saves
    the annotated video to disk.
    
    Args:
        model (YOLO): The trained YOLO model for license plate detection
        source (Union[int, Path]): Video file path or webcam index (0, 1, etc.)
        
    Raises:
        SystemExit: If video file cannot be opened
        
    Note:
        For video files, output is saved with "_det" suffix.
        For webcam, output is saved to runs/predict_video_cpu/webcam_det.mp4
        Press 'q' to quit during processing.
    """
    # Initialize video capture and writer variables
    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None
    out_path: Optional[Path] = None

    # Handle video file input
    if isinstance(source, Path):
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"Failed to open video: {source}")
            sys.exit(1)
        
        # Create output path and video writer for file input
        out_path = source.with_name(source.stem + "_det.mp4")
        writer = open_writer_like(cap, out_path)

    # Display user instructions
    print("Press 'q' to quit.")
    
    # Process video/webcam stream with YOLO model
    for result in model.predict(
        source=str(source) if isinstance(source, Path) else source,
        stream=True,        # Enable streaming for real-time processing
        device="cpu",       # Use CPU for compatibility
        imgsz=640,          # Standard YOLO input size
        conf=0.25,          # Confidence threshold
        verbose=False,      # Suppress verbose output
    ):
        # Generate annotated frame with bounding boxes
        frame = result.plot()

        # Handle webcam case: create writer on first frame
        if writer is None and cap is None:
            # Set up output directory and path for webcam recording
            out_dir = Path("runs") / "predict_video_cpu"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "webcam_det.mp4"
            
            # Get frame dimensions and create video writer
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))

        # Write frame to output video if writer is available
        if writer is not None:
            writer.write(frame)

        # Display the annotated frame
        cv2.imshow("Detections (press q to exit)", frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up resources
    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Notify user of saved output
    if out_path is not None:
        print(f"Saved annotated video to: {out_path}")


def main() -> None:
    """
    Main entry point for the PlateNet AI detection system.
    
    This function orchestrates the entire detection workflow:
    1. Resolves the project structure and model weights
    2. Loads the trained YOLO model
    3. Prompts user for detection mode
    4. Executes the appropriate detection function
    
    The function handles all three detection modes:
    - Image processing: Single image detection and annotation
    - Video processing: Video file detection with output saving
    - Webcam processing: Real-time live detection
    
    Raises:
        SystemExit: For various error conditions (invalid input, missing files)
        
    Note:
        All detection operations use CPU processing for maximum compatibility.
        The system automatically creates output directories and files as needed.
    """
    # Get the project root directory (where this script is located)
    project_root = Path(__file__).resolve().parent
    
    # Resolve and load the trained model weights
    weights = resolve_weights(project_root)
    model = YOLO(str(weights))

    # Get user's preferred detection mode
    mode = choose_mode()
    
    # Execute detection based on user choice
    if mode == "1":
        # Image detection mode
        img = prompt_path("Enter image path: ")
        run_image(model, img)
    elif mode == "2":
        # Video detection mode
        vid = prompt_path("Enter video path: ")
        run_video(model, vid)
    else:
        # Webcam detection mode
        index_str = input("Enter webcam index (default 0): ").strip() or "0"
        if not index_str.isdigit():
            print("Invalid webcam index.")
            sys.exit(1)
        run_video(model, int(index_str))


# Script execution entry point
if __name__ == "__main__":
    main()


