import onnxruntime as ort
import numpy as np
import cv2

class ONNXObjectDetector:
    def __init__(self, onnx_model_path):
        # Load the ONNX model
        options = ort.SessionOptions()
        options.enable_profiling = True
        self.session = ort.InferenceSession(onnx_model_path,options = options, providers=['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.target_size = (416, 416)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        prof_file = self.session.end_profiling()
        print("onnx_session profile: ", prof_file)

    def preprocess_image(self, input_image:np.ndarray):
        if input_image.shape[0] > input_image.shape[1]:
            new_height = self.target_size[0]
            new_width = int(self.target_size[1] * (input_image.shape[1] / input_image.shape[0]))
        else:
            new_width = self.target_size[1]
            new_height = int(self.target_size[0] * (input_image.shape[0] / input_image.shape[1]))

        inv_height_factor = input_image.shape[0] / new_height
        inv_width_factor = input_image.shape[1] / new_width

        resized_image = cv2.resize(input_image, (new_width, new_height))

        padded_image = np.full(
            (self.target_size[1], self.target_size[0], input_image.shape[2]), 114, dtype=np.uint8
        )
        padded_image[:new_height, :new_width, :] = resized_image

        ret = np.transpose(padded_image, (2, 0, 1))
        ret = np.expand_dims(ret, axis=0)
        ret = np.ascontiguousarray(ret, dtype=np.float32)

        return ret, inv_height_factor, inv_width_factor

    def detect(self, image):
        input_tensor, height_factor, width_factor = self.preprocess_image(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        boxes, scores = outputs[0], outputs[1]

        x_indices = [0, 2]
        y_indices = [1, 3]
        boxes[..., x_indices] *= width_factor
        boxes[..., y_indices] *= height_factor

        classes = np.argmax(scores, axis=2)
        classes = classes.reshape(1, classes.size, 1)
        max_scores = np.max(scores, axis=2)
        max_scores = max_scores.reshape(1, classes.size, 1)
        boxes = np.concatenate([boxes, max_scores, classes], axis=-1).squeeze()

        return boxes