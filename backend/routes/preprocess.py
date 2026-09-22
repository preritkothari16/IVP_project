import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from ..core.preprocess import preprocess_image, to_uint8

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])


def img_to_b64(img_array: np.ndarray) -> str:
    pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


@router.post("/single")
async def preprocess_single(file: UploadFile = File(...)):
    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert('RGB'))
    result = preprocess_image(img)
    return {
        'original': img_to_b64(result['original']),
        'resized': img_to_b64(result['resized']),
        'filtered': img_to_b64(result['filtered']),
        'enhanced': img_to_b64(result['enhanced']),
        'clean_mask': img_to_b64(result['clean_mask']),
        'masked_output': img_to_b64(result['masked_output']),
    }
