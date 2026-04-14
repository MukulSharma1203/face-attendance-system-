import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config

logger = logging.getLogger("embeddings")


@lru_cache(maxsize=1)
def load_model() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(config.FRAME_WIDTH, config.FRAME_HEIGHT))
    logger.info("InsightFace model loaded on CPU")
    return app


def align_face(image: np.ndarray, face_object) -> np.ndarray:
    bbox = face_object.bbox.astype(int)
    x1, y1, x2, y2 = bbox.tolist()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((112, 112, 3), dtype=np.uint8)
    return cv2.resize(crop, (112, 112))


def get_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    model = load_model()
    faces = model.get(image, max_num=config.MAX_FACES_PER_FRAME)
    if not faces:
        return None

    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = getattr(largest, "normed_embedding", None)
    if emb is None:
        emb = getattr(largest, "embedding", None)
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


def get_all_embeddings(image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    face_data = get_all_face_data(image)
    return [(item["bbox"], item["embedding"]) for item in face_data]


def get_all_face_data(image: np.ndarray) -> List[Dict]:
    model = load_model()
    faces = model.get(image, max_num=config.MAX_FACES_PER_FRAME)
    out: List[Dict] = []

    for face in faces:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            continue

        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        lmk68 = getattr(face, "landmark_3d_68", None)
        if lmk68 is not None:
            lmk68 = np.asarray(lmk68, dtype=np.float32)

        out.append(
            {
                "bbox": (x1, y1, x2, y2),
                "embedding": np.asarray(emb, dtype=np.float32),
                "landmark_3d_68": lmk68,
            }
        )

    return out
