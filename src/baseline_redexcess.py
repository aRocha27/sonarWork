import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Det:
    image: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float            # proxy (0-1)
    mean_redex: float       # 0-255
    cls: str                # low/medium/high


def compute_red_excess(bgr: np.ndarray) -> np.ndarray:
    """
    red_excess = max(0, R - max(G,B))
    Input: BGR uint8
    Output: uint8 [0..255]
    """
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    mx = np.maximum(g, b)
    red_ex = r - mx
    red_ex[red_ex < 0] = 0
    return red_ex.astype(np.uint8)


def threshold_mask(red_ex: np.ndarray,
                   method: str = "percentile",
                   p: float = 99.5,
                   k: float = 2.0) -> np.ndarray:
    """
    Produz máscara binária (0/255) a partir de red_ex.
    method:
      - "percentile": T = percentile(red_ex>0, p)
      - "meanstd":    T = mean + k*std  (em red_ex>0)
      - "otsu":       Otsu em red_ex (mas geralmente pior em sonar)
    """
    nz = red_ex[red_ex > 0]
    if nz.size == 0:
        return np.zeros_like(red_ex, dtype=np.uint8)

    if method == "percentile":
        T = np.percentile(nz, p)
    elif method == "meanstd":
        T = float(nz.mean() + k * nz.std())
    elif method == "otsu":
        # Otsu precisa de imagem 8-bit; aplica em tudo
        _, mask = cv2.threshold(red_ex, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    else:
        raise ValueError("method must be: percentile | meanstd | otsu")

    mask = (red_ex >= T).astype(np.uint8) * 255
    return mask


def clean_mask(mask: np.ndarray,
               open_iter: int = 0,
               close_iter: int = 1,
               dilate_iter: int = 1,
               ksize: int = 3) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = mask
    if open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iter)
    if dilate_iter > 0:
        m = cv2.dilate(m, k, iterations=dilate_iter)
    return m



def find_blobs(mask: np.ndarray,
               min_area: int = 40,
               max_area: Optional[int] = None) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = []
    H, W = mask.shape[:2]
    if max_area is None:
        max_area = int(0.20 * H * W)  # evita blobs gigantes (ruído colado)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        keep.append(c)
    return keep


def classify_by_blob_mean(red_ex: np.ndarray,
                          contours: List[np.ndarray]) -> Tuple[List[Tuple[float, str]], Tuple[float, float]]:
    """
    Classifica por mean(red_ex) dentro de cada blob.
    Thresholds por imagem (p33,p66) nos pixels das regiões detectadas.
    Retorna:
      - lista (mean, cls) por contorno
      - (p33, p66)
    """
    # valores nas regiões detectadas
    union = np.zeros_like(red_ex, dtype=np.uint8)
    cv2.drawContours(union, contours, -1, 255, thickness=-1)
    vals = red_ex[union > 0]
    if vals.size == 0:
        return [(0.0, "low") for _ in contours], (0.0, 0.0)

    p33, p66 = np.percentile(vals, [33, 66])

    out = []
    for c in contours:
        blob = np.zeros_like(red_ex, dtype=np.uint8)
        cv2.drawContours(blob, [c], -1, 255, thickness=-1)
        v = red_ex[blob > 0]
        if v.size:
            v_sorted = np.sort(v)
            top = v_sorted[int(0.90 * len(v_sorted)):]   # top 10%
            m = float(top.mean()) if top.size else float(v.mean())
        else:
             m = 0.0
        if m < p33:
            cls = "low"
        elif m < p66:
            cls = "medium"
        else:
            cls = "high"
        out.append((m, cls))

    return out, (float(p33), float(p66))


def run_baseline_on_image(img_path: str,
                          thr_method: str = "percentile",
                          p: float = 99.5,
                          k: float = 2.0,
                          median_ksize: int = 3,
                          min_area: int = 40) -> Tuple[List[Det], np.ndarray]:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    red_ex = compute_red_excess(bgr)

    if median_ksize >= 3:
        red_ex = cv2.medianBlur(red_ex, median_ksize)

    mask = threshold_mask(red_ex, method=thr_method, p=p, k=k)
    mask = clean_mask(mask, open_iter=0, close_iter=1, dilate_iter=1, ksize=3)

    contours = find_blobs(mask, min_area=min_area)

    (means_and_cls, _) = classify_by_blob_mean(red_ex, contours)

    dets: List[Det] = []
    out = bgr.copy()

    for c, (m, cls) in zip(contours, means_and_cls):
        x, y, w, h = cv2.boundingRect(c)
        x1, y1, x2, y2 = x, y, x + w, y + h

        score = float(np.clip(m / 255.0, 0, 1))

        dets.append(Det(
            image=os.path.basename(img_path),
            x1=x1, y1=y1, x2=x2, y2=y2,
            score=score,
            mean_redex=m,
            cls=cls
        ))

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"{cls} | s={score:.2f} | m={m:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ordenar por score desc (opcional)
    dets.sort(key=lambda d: d.score, reverse=True)

    return dets, out


def run_folder(input_dir: str,
               out_dir: str,
               csv_path: str,
               thr_method: str = "percentile",
               p: float = 99.5,
               k: float = 2.0,
               median_ksize: int = 3,
               min_area: int = 40,
               exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")):

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(exts):
            continue
        img_path = os.path.join(input_dir, name)

        dets, vis = run_baseline_on_image(
            img_path,
            thr_method=thr_method,
            p=p, k=k,
            median_ksize=median_ksize,
            min_area=min_area
        )

        cv2.imwrite(os.path.join(out_dir, name), vis)

        for d in dets:
            rows.append({
                "image": d.image,
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "score": d.score,
                "mean_red_excess": d.mean_redex,
                "class": d.cls
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[baseline] wrote {len(df)} detections to {csv_path}")
    print(f"[baseline] wrote annotated images to {out_dir}")
