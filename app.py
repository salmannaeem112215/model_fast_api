from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import shutil
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
app = FastAPI()

# Load Models
foot_model = load_model('./foot_model.h5')
disease_model = YOLO('./dis_model.pt')

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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if predict_is_foot(file_location):
            if predict_is_abnormal(file_location):
                result = "Abnormal Feet"
            else:
                result = "Normal Feet"
        else:
            result = "Feet Not Found"
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(file_location)

    return {"result": result}
