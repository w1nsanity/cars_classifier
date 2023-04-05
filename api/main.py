from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    'http://localhost',
    'http://localhost:3000',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL = tf.keras.models.load_model('../models/1')

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
    
    predictions = MODEL.predict(img_batch)
    
    pred_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'pred_class': pred_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port='8000')