import os
import cv2
import imageio
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

"""
Fire Detection on GIF / Video frames using trained CNN model.
Outputs an annotated GIF with predicted class and probability.
"""

# ---------------------------------------------------------
# GIF FIRE DETECTION (MAIN FUNCTION YOU NEED)
# ---------------------------------------------------------
def run_gif_fire_detection(gif_path, output_gif_path, model_path, preprocess, image_size):
    classes = ['fire', 'no_fire', 'start_fire']

    print("Input GIF :", gif_path)
    print("Output GIF:", output_gif_path)
    print("Model     :", model_path)

    model = load_model(model_path)

    reader = imageio.get_reader(gif_path)
    writer = imageio.get_writer(output_gif_path, fps=10)

    for frame in reader:
        # ---- normalize channels ----
        if frame.ndim == 2:  # grayscale
            frame = np.stack((frame,) * 3, axis=-1)
        elif frame.shape[-1] == 4:  # RGBA
            frame = frame[:, :, :3]

        # ---- prepare image for model ----
        img = cv2.resize(frame, image_size)
        x = np.expand_dims(img, axis=0)
        x = preprocess(x)

        probs = model.predict(x, verbose=0)[0]
        idx = np.argmax(probs)
        label = f"{classes[idx]} : {probs[idx] * 100:.2f}%"

        # ---- CRITICAL FIX (OpenCV compatibility) ----
        frame = np.ascontiguousarray(frame)
        frame = frame.astype(np.uint8)

        # ---- draw label ----
        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        writer.append_data(frame)

    writer.close()
    print("✅ Annotated GIF saved successfully")


# ---------------------------------------------------------
# OPTIONAL: Convert GIF → MP4 (NOT REQUIRED FOR GIF OUTPUT)
# ---------------------------------------------------------
def convert_gif_to_mp4(gif_path, mp4_path, fps=24):
    print("Converting GIF to MP4...")

    reader = imageio.get_reader(gif_path)
    writer = imageio.get_writer(mp4_path, fps=fps)

    for frame in reader:
        if frame.ndim == 2:
            frame = np.stack((frame,) * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        frame = frame.astype(np.uint8)
        writer.append_data(frame)

    writer.close()
    print("GIF converted to MP4:", mp4_path)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_gif = os.path.join(BASE_DIR, "my_videos", "test_nofire.gif")
    output_gif = os.path.join(BASE_DIR, "output_videos", "annotated_test_nofire.gif")
    model_path = os.path.join(BASE_DIR, "final_fire_model.h5")

    # create output folder if missing
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    run_gif_fire_detection(
        gif_path=input_gif,
        output_gif_path=output_gif,
        model_path=model_path,
        preprocess=preprocess_input,
        image_size=(224, 224)
    )
