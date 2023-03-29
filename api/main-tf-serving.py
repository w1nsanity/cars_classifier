from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import json

app = FastAPI()

enpdoint = 'http://localhost:8501/v1/models/cars_model:predict'

CLASS_NAMES = ['GAZ 3102', 'GAZ 3110', 'GAZ 31105', 'LADA 1111', 'LADA 2101',
               'LADA 2102', 'LADA 2103', 'LADA 2104', 'LADA 2105', 'LADA 2106',
               'LADA 2107', 'LADA 2108', 'LADA 2109', 'LADA 21099', 'LADA 2110',
               'LADA 2111', 'LADA 2112', 'LADA 2113', 'LADA 2114', 'LADA 2115',
               'LADA 2120', 'LADA GRANTA', 'LADA KALINA', 'LADA LARGUS', 'LADA NIVA',
               'LADA PRIORA', 'LADA VESTA', 'LADA XRAY', 'MOSCVICH 2140', 'MOSCVICH 2141', 'ZAPOROZHECH']

@app.get('/ping')
async def ping():
    return 'hello im here'

def read_file_as_image(data) -> np.ndarray:
    new_size = (256, 256)
    image = np.array(Image.open(BytesIO(data)).resize(new_size))
    return image

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    json_data = {
        'instances': img_batch.tolist()
    }
    
    response = requests.post(enpdoint, json=json.dumps(json_data))
    prediction = np.array(response.json()['predictions'][0])
    
    pred_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return {
        'pred_class': pred_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port='8000')