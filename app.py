import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import shutil
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from PIL import Image

app = FastAPI()

# Load Models
foot_model = load_model('./foot_model.h5')
disease_model = YOLO('./dis_model.pt')


# def save_image_to_media(src_path: str) -> str:
#     os.makedirs("media", exist_ok=True)

#     timestamp = int(datetime.utcnow().timestamp() * 1000)
#     dest_filename = f"{timestamp}.png"
#     dest_path = os.path.join("media", dest_filename)

#     shutil.copyfile(src_path, dest_path)
#     return dest_path
def save_image_to_media(src_path: str) -> str:
    try:
        os.makedirs("media", exist_ok=True)

        # Generate the destination file path
        timestamp = int(datetime.utcnow().timestamp() * 1000)
        dest_filename = f"{timestamp}.png"
        dest_path = os.path.join("media", dest_filename)

        # Open the image and resize to 224x224
        with Image.open(src_path) as img:
            img = img.convert("RGB")  # Ensure compatibility (if source image has alpha channel)
            img = img.resize((224, 224), Image.Resampling.LANCZOS) # Resize with good quality

            # Save the image to the destination path
            img.save(dest_path, format='PNG')

        return dest_path

    except Exception as e:
        print(f"Error occurred: {e}")
        return ""


def convert_image(img_path, img_height=224, img_width=224):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_is_foot(img_path):
    img_array = convert_image(img_path)
    prediction = foot_model.predict(img_array)
    if prediction[0][0] > 0.5:
        return False
    else:
        return True

def predict_is_abnormal(img_path, conf_threshold=0.25):
    results = disease_model.predict(source=img_path, conf=conf_threshold)
    for result in results:
        classes = result.boxes.cls if result.boxes else []
        for cls in classes:
            if int(cls) == 1:
                return False  # Class 1 detected
    return True  # No class 1 found

app.mount("/media", StaticFiles(directory="media"), name="media")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    savedPath = ''
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if predict_is_foot(file_location):
            savedPath = save_image_to_media(file_location)
            if predict_is_abnormal(file_location):
                result = "Abnormal Foot"
            else:
                result = "Normal Foot"
        else:
            result = "Foot Not Found"
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(file_location)

    return {"result": result,"savedPath":savedPath    }
@app.get("/ping")
async def ping():
    return {"message": "Pong! Server is up and running."}
