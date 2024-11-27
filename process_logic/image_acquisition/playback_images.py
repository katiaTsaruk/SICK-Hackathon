import image_acquisition.image_sensor as image_sensor
import cv2

def _default_callback(img):
     # Show the frame
    cv2.imshow("Video Playback", img)

    # Check if 'q' is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise image_sensor.NoMoreImages

def run_playback( image_source:image_sensor.ImageSensor, callback = _default_callback):
    while True:
        try:
            image = image_source.get_next_image()
            callback(image)
        except image_sensor.NoMoreImages:
            break
    