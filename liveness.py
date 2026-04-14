import logging
import os
import pickle
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("liveness")


def _uniform_lbp(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    center = gray[1 : h - 1, 1 : w - 1]
    codes = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        gray[0 : h - 2, 0 : w - 2],
        gray[0 : h - 2, 1 : w - 1],
        gray[0 : h - 2, 2:w],
        gray[1 : h - 1, 2:w],
        gray[2:h, 2:w],
        gray[2:h, 1 : w - 1],
        gray[2:h, 0 : w - 2],
        gray[1 : h - 1, 0 : w - 2],
    ]

    for i, nbr in enumerate(neighbors):
        codes |= ((nbr >= center) << i).astype(np.uint8)

    return codes


def extract_lbp_features(face_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96))
    lbp = _uniform_lbp(gray)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    return hist.astype(np.float32)


def analyze_texture_frequency(face_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96)).astype(np.float32)

    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(fft) + 1.0)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    low = magnitude[cy - 8 : cy + 8, cx - 8 : cx + 8].mean()
    high = magnitude.mean() - low
    ratio = float(high / (low + 1e-6))

    return np.array([float(low), float(high), ratio], dtype=np.float32)


@lru_cache(maxsize=1)
def load_liveness_model() -> Optional[object]:
    if not os.path.exists(config.LIVENESS_MODEL_PATH):
        logger.warning("Liveness model not found. Falling back to rule-based heuristic.")
        return None

    try:
        with open(config.LIVENESS_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded liveness model from disk")
        return model
    except Exception as exc:
        logger.warning("Failed to load liveness model (%s). Using heuristic fallback.", exc)
        return None


def _heuristic_liveness(face_crop: np.ndarray, threshold: float) -> Tuple[bool, float]:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    specular_ratio = float((v > 245).sum() / (v.size + 1e-6))

    lbp = extract_lbp_features(face_crop)
    lbp_entropy = float(-np.sum(lbp * np.log(lbp + 1e-10)))

    freq = analyze_texture_frequency(face_crop)
    freq_ratio = float(freq[2])

    blur_norm = min(1.0, blur_score / 150.0)
    entropy_norm = min(1.0, lbp_entropy / 5.0)
    specular_penalty = max(0.0, min(1.0, specular_ratio * 8.0))
    freq_penalty = max(0.0, min(1.0, abs(freq_ratio - 0.9)))

    confidence = 0.55 * blur_norm + 0.35 * entropy_norm - 0.05 * specular_penalty - 0.05 * freq_penalty
    confidence = float(max(0.0, min(1.0, confidence)))

    return confidence >= threshold, confidence


def check_liveness(face_crop: np.ndarray, threshold: Optional[float] = None) -> Tuple[bool, float]:
    if face_crop is None or face_crop.size == 0:
        return False, 0.0

    effective_threshold = config.LIVENESS_THRESHOLD if threshold is None else float(threshold)

    model = load_liveness_model()

    lbp_features = extract_lbp_features(face_crop)
    freq_features = analyze_texture_frequency(face_crop)
    feature_vec = np.concatenate([lbp_features, freq_features], axis=0).reshape(1, -1)

    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(feature_vec)[0][1])
            else:
                pred = float(model.predict(feature_vec)[0])
                proba = pred
            is_live = proba >= effective_threshold
            return is_live, proba
        except Exception as exc:
            logger.warning("Liveness model inference failed (%s). Using heuristic.", exc)

    heuristic_threshold = min(effective_threshold, config.HEURISTIC_LIVENESS_THRESHOLD)
    return _heuristic_liveness(face_crop, threshold=heuristic_threshold)
