"""Project paths — all directories relative to snap_project root."""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CASCADES_DIR = os.path.join(PROJECT_ROOT, 'cascades')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

TRAINING_CSV = os.path.join(DATA_DIR, 'training.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

FACE_CASCADE = os.path.join(CASCADES_DIR, 'haarcascade_frontalface_default.xml')
MODEL_WEIGHTS = os.path.join(MODELS_DIR, 'my_model')

FILTER_IMAGES = [
    'sunglasses.png',
    'sunglasses_6.png',
    'sunglasses_5.jpg',
    'sunglasses_4.png',
    'sunglasses_3.jpg',
    'sunglasses_2.png',
]
