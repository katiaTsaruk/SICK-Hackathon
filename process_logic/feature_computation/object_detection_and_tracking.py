from feature_computation.object_tracking import Tracker
from typing import List
from feature_computation.object_detection import ONNXObjectDetector
import numpy as np

class onnx_network_and_trackers:
    #handles a neuran network defined as onnx files and a collection of Trackers
    #an image given to this is fed to the neural network and the result is fed to all trackers
    #the main purpose is to represent this one to many relation
    def __init__(self, onnx_file_path:str, trackers: List[Tracker]):
        self.neural_network = None
        self.onnx_file_path = onnx_file_path
        self.trackers = trackers

    def __enter__(self):
        self.neural_network = ONNXObjectDetector(onnx_model_path=self.onnx_file_path)
        self.neural_network.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.neural_network is not None:
            self.neural_network.__exit__(exc_type, exc_value, traceback)
            self.neural_network = None  # Clear the reference to M

    def update_trackers(self, img:np.ndarray):
        detections = self.neural_network.detect(img)
        for t in self.trackers:
            t.update(detections)

