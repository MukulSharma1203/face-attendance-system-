import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

import config


def setup_logging() -> None:
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a.astype(np.float32)
    b = vec_b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    sim = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def load_embeddings() -> Dict[int, np.ndarray]:
    if not os.path.exists(config.EMBEDDINGS_PATH):
        return {}

    with open(config.EMBEDDINGS_PATH, "rb") as f:
        raw = pickle.load(f)

    data: Dict[int, np.ndarray] = {}
    for user_id, emb in raw.items():
        data[int(user_id)] = np.asarray(emb, dtype=np.float32)
    return data


def save_embeddings(embeddings_dict: Dict[int, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(config.EMBEDDINGS_PATH), exist_ok=True)
    serializable = {int(k): np.asarray(v, dtype=np.float32) for k, v in embeddings_dict.items()}
    with open(config.EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(serializable, f)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def find_best_match(
    query_embedding: np.ndarray,
    embeddings_dict: Dict[int, np.ndarray],
) -> Tuple[Optional[int], float]:
    best_user_id: Optional[int] = None
    best_score = 0.0

    for user_id, known_emb in embeddings_dict.items():
        score = cosine_similarity(query_embedding, known_emb)
        if score > best_score:
            best_score = score
            best_user_id = user_id

    if best_score >= config.SIMILARITY_THRESHOLD:
        return best_user_id, best_score

    return None, 0.0
