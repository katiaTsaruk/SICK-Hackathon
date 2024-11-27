from abc import ABC, abstractmethod
import numpy as np
import cv2
import os


class NoMoreImages(Exception):
    pass
class ImageSensor(ABC):
    @abstractmethod
    def get_next_image(self) -> np.ndarray:
        pass

    @abstractmethod   
    def frame_size(self) -> tuple:
        pass

    @abstractmethod   
    def fps(self)-> float:
        pass

class VideoFileImageSensor(ImageSensor):
    def __init__(self, video_path: str):
        # Initialize instance variables but do not open resources here
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        # Open the video file resource
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Video File not found or could not be opened")
        return self

    def get_next_image(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Video capture not initialized. Use the 'with' statement.")
        _, frame = self.cap.read()
        if frame is None:
            raise NoMoreImages()
        return frame

    def frame_size(self) -> tuple:
        if self.cap is None:
            raise RuntimeError("Video capture not initialized. Use the 'with' statement.")
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (frame_height, frame_width)

    def fps(self) -> float:
        if self.cap is None:
            raise RuntimeError("Video capture not initialized. Use the 'with' statement.")
        return self.cap.get(cv2.CAP_PROP_FPS)

    def __exit__(self, exc_type, exc_value, traceback):
        # Release the video resource
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        # Ensure resources are released if not already
        if self.cap:
            self.cap.release()



class DirectoryImageSensor(ImageSensor):
    def __init__(self, directory_path: str):
        """
        Initialize the DirectoryImageSensor with the path to a directory containing images.

        :param directory_path: Path to the directory containing image files.
        """
        self.directory_path = directory_path
        self.image_files = []
        self.index = 0

    def __enter__(self):
        """
        Populate the list of image files in the directory and reset the index.

        :return: self
        """
        if not os.path.isdir(self.directory_path):
            raise RuntimeError(f"Directory not found: {self.directory_path}")

        # Get sorted list of image files (e.g., jpg, png, bmp)
        self.image_files = sorted(
            [
                os.path.join(self.directory_path, f)
                for f in os.listdir(self.directory_path)
                if os.path.isfile(os.path.join(self.directory_path, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
        )

        if not self.image_files:
            raise RuntimeError("No image files found in the directory.")

        self.index = 0
        return self

    def get_next_image(self) -> np.ndarray:
        """
        Retrieve the next image in the directory as a NumPy array.

        :return: The next image as a NumPy array.
        :raises NoMoreImages: If all images have been read.
        """
        if self.index >= len(self.image_files):
            raise NoMoreImages()

        image_path = self.image_files[self.index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        self.index += 1
        return image

    def frame_size(self) -> tuple:
        """
        Return the dimensions of the first image in the directory.

        :return: Tuple containing (height, width) of the image.
        """
        if not self.image_files:
            raise RuntimeError("No image files loaded. Use the 'with' statement to initialize.")

        image = cv2.imread(self.image_files[0], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {self.image_files[0]}")

        return image.shape[:2]  # (height, width)

    def fps(self) -> float:
        """
        Return frames per second. For a directory sensor, return a default value (e.g., 0.0).

        :return: Frames per second (default: 0.0).
        """
        return 0.0

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Cleanup resources (if necessary).
        """
        self.image_files = []
        self.index = 0

    def __del__(self):
        """
        Ensure resources are cleaned up if not already done.
        """
        self.image_files = []
        self.index = 0