import cv2
import os
from image_acquisition.image_sensor import ImageSensor

def video_file_writer_matching_image_sensor(img_src:ImageSensor, output_file_name:str):
    return VideoFileWriter(img_src.frame_size()[0], img_src.frame_size()[1], img_src.fps(), output_file_name)

class VideoFileWriter:
    def __init__(self, height, width, fps, output_file_name):
        self.height = height
        self.width = width
        self.fps = fps
        self.output_file_name = output_file_name
        self.writer = None

    def __enter__(self):
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_file_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
        self.writer = cv2.VideoWriter(self.output_file_name, fourcc, self.fps, (self.width, self.height))
        return self

    def write(self, img):
        if self.writer is not None:
            self.writer.write(img)
        else:
            raise RuntimeError("Video writer not initialized. Did you forget to use the context manager?")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release the video writer when done
        if self.writer is not None:
            self.writer.release()
            self.writer = None