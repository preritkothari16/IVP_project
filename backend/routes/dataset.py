import os
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.get("/classes")
def list_classes(dataset_root: str = Query(...)):
    classes = []
    if not os.path.isdir(dataset_root):
        return {'error': 'Folder not found', 'classes': []}
    for name in sorted(os.listdir(dataset_root)):
        class_dir = os.path.join(dataset_root, name)
        if not os.path.isdir(class_dir):
            continue
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        classes.append({'name': name, 'count': len(images)})
    total = sum(c['count'] for c in classes)
    return {'total_images': total, 'num_classes': len(classes), 'classes': classes}


@router.get("/sample-images")
def sample_images(dataset_root: str = Query(...), class_name: str = Query(...), limit: int = Query(5)):
    class_dir = os.path.join(dataset_root, class_name)
    if not os.path.isdir(class_dir):
        return {'error': 'Class not found', 'images': []}
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    images = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(extensions)])
    return {'images': images[:limit], 'total': len(images)}


@router.get("/image")
def serve_image(dataset_root: str = Query(...), class_name: str = Query(...), filename: str = Query(...)):
    img_path = os.path.join(dataset_root, class_name, filename)
    if not os.path.isfile(img_path):
        return {'error': 'Image not found'}
    return FileResponse(img_path)
