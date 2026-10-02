import os
import time
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def tile_slices(height, width, tile_size):
    out = []
    idx = 0
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            out.append((idx, y0, y1, x0, x1))
            idx += 1
    return out


def tile_scores(map2d, image_hw, tile_size):
    """Resize a priority map to image size and average within encryption tiles."""
    from PIL import Image
    h, w = image_hw
    arr = np.asarray(map2d, dtype=np.float32)
    im = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255)).resize((w, h), Image.Resampling.BILINEAR)
    r = np.asarray(im, dtype=np.float32) / 255.0
    scores = []
    for idx, y0, y1, x0, x1 in tile_slices(h, w, tile_size):
        block = r[y0:y1, x0:x1]
        scores.append((idx, float(block.mean())))
    return scores


def select_top(scores, fraction):
    n = len(scores)
    k = max(1, min(n, int(np.ceil(float(fraction) * n))))
    ranked = sorted(scores, key=lambda z: z[1], reverse=True)
    return set(idx for idx, _ in ranked[:k])


def select_random(n_tiles, fraction, rng):
    k = max(1, min(n_tiles, int(np.ceil(float(fraction) * n_tiles))))
    return set(int(x) for x in rng.choice(n_tiles, size=k, replace=False))


def priority_metrics(scores, selected, top_fraction=0.20):
    values = np.asarray([s for _, s in scores], dtype=np.float64)
    ids = [idx for idx, _ in scores]
    selected_mask = np.asarray([idx in selected for idx in ids], dtype=bool)
    total = float(values.sum())
    mass = float(values[selected_mask].sum() / total) if total > 1e-12 else float(selected_mask.mean())
    k = max(1, int(np.ceil(top_fraction * len(ids))))
    top_ids = set(idx for idx, _ in sorted(scores, key=lambda z: z[1], reverse=True)[:k])
    recall = len(top_ids.intersection(selected)) / float(len(top_ids))
    protected = float(selected_mask.mean())
    efficiency = mass / protected if protected > 0 else 0.0
    return protected, mass, float(recall), float(efficiency)


def encrypt_selected_tiles(image_rgb, selected, tile_size, key=None, aad=b"SRACR-Med-v7.1"):
    arr = np.asarray(image_rgb, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    tiles = tile_slices(h, w, tile_size)
    key = key or AESGCM.generate_key(bit_length=256)
    aes = AESGCM(key)
    records = {}
    encrypted_payload_bytes = 0

    t0 = time.perf_counter()
    for idx, y0, y1, x0, x1 in tiles:
        if idx not in selected:
            continue
        plain = arr[y0:y1, x0:x1].tobytes()
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plain, aad)
        records[idx] = (nonce, ct, (y0, y1, x0, x1), arr[y0:y1, x0:x1].shape)
        encrypted_payload_bytes += len(ct)
        # The benchmark representation masks encrypted regions in memory.
        arr[y0:y1, x0:x1] = 0
    enc_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    recovered = arr.copy()
    for idx, (nonce, ct, bounds, shape) in records.items():
        y0, y1, x0, x1 = bounds
        plain = aes.decrypt(nonce, ct, aad)
        recovered[y0:y1, x0:x1] = np.frombuffer(plain, dtype=np.uint8).reshape(shape)
    dec_ms = (time.perf_counter() - t1) * 1000.0

    overhead_bytes = len(records) * 12 + len(records) * 16  # nonce + compact per-tile metadata; GCM tag is already in ciphertext
    return {
        "recovered": recovered,
        "encryption_ms": float(enc_ms),
        "decryption_ms": float(dec_ms),
        "encrypted_payload_bytes": int(encrypted_payload_bytes),
        "metadata_overhead_est_bytes": int(overhead_bytes),
        "encrypted_tiles": len(records),
        "total_tiles": len(tiles),
    }
