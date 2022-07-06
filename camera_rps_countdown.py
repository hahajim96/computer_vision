import cv2
import time
from keras.models import load_model
import numpy as np


def get_prediction(TIMER=3):
    # SET THE COUNTDOWN TIMER
    # for simplicity we set it to 3
    # We can also take this as input
    TIMER = int(TIMER)

    # load the model
    model = load_model("keras_model.h5")

    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:

        # Read and display each frame
        ret, img = cap.read()
        cv2.imshow("a", img)

        # check for the key pressed
        k = cv2.waitKey(125)

        # set the key for the countdown
        # to begin. Here we set q
        # if key pressed is q
        if k == ord("q"):
            prev = time.time()

            while TIMER >= 0:
                ret, img = cap.read()

                # Display countdown on each frame
                # specify the font and draw the
                # countdown using puttext
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    img, str(TIMER), (200, 250), font, 7, (0, 255, 255), 4, cv2.LINE_AA
                )
                cv2.imshow("a", img)
                cv2.waitKey(125)

                # current time
                cur = time.time()

                # Update and keep track of Countdown
                # if time elapsed is one second
                # than decrease the counter
                if cur - prev >= 1:
                    prev = cur
                    TIMER = TIMER - 1
            else:
                ret, img = cap.read()

                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                resized_frame = cv2.resize(
                    img, (224, 224), interpolation=cv2.INTER_AREA
                )
                image_np = np.array(resized_frame)
                normalized_image = (
                    image_np.astype(np.float32) / 127.0
                ) - 1  # Normalize the image
                data[0] = normalized_image
                prediction = model.predict(data)

                # Display the clicked frame for 2
                # sec.You can increase time in
                # waitKey also
                cv2.imshow("a", img)

                # time for which image displayed
                cv2.waitKey(2000)

                # Save the frame
                cv2.imwrite("camera.jpg", img)

                # HERE we can reset the Countdown timer
                # if we want more Capture without closing
                # the camera
        # Press Esc to exit
        elif k == 27:
            break
    print(prediction)
    event_probability = np.argmax(prediction)

    # close the camera
    cap.release()

    # close all the opened windows
    cv2.destroyAllWindows()

    return event_probability
