import argparse
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def iter_images(source: Path):
    if source.is_file():
        yield source
        return
    for p in sorted(source.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def to_intensity_float01(img_path: Path, mode: str = "hsv_v") -> np.ndarray:
    im = Image.open(img_path).convert("RGB")

    if mode == "luma":
        g = im.convert("L")
        return np.asarray(g, dtype=np.float32) / 255.0
    
    if mode == "red":
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr[:, :, 0].astype(np.float32)

    if mode == "red_excess":
        arr = np.asarray(im, dtype=np.float32) / 255.0
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        excess = r - np.maximum(g, b)
        excess = np.clip(excess, 0.0, 1.0)

        # normalização por imagem (evita imagens "mais quentes" dominarem)
        p99 = np.percentile(excess, 99)
        if p99 > 1e-6:
            excess = np.clip(excess / p99, 0.0, 1.0)
        return excess.astype(np.float32)

    # hsv_v
    hsv = im.convert("HSV")
    hsv_np = np.asarray(hsv, dtype=np.float32)  # H,S,V em 0..255
    v = hsv_np[:, :, 2] / 255.0
    return v.astype(np.float32)


def clip_normalize_patch(patch: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Opcional: clipping p1/p99 e normalização (dentro do patch)
    Ajuda a estabilizar thresholds quando imagens variam de brilho.
    """
    if patch.size == 0:
        return patch
    if not clip:
        return patch

    p1 = np.percentile(patch, 1)
    p99 = np.percentile(patch, 99)
    if p99 <= p1:
        return np.clip(patch, 0.0, 1.0)
    patch = np.clip(patch, p1, p99)
    patch = (patch - p1) / (p99 - p1 + 1e-8)
    return patch

def intensity_score(int01: np.ndarray, x1, y1, x2, y2,method="p95", topk=0.05, do_clip=True, strong_thr=0.20, w_mean=0.7, w_frac=0.3) -> float:
    h, w = int01.shape

    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w, x2)))
    y2 = int(max(0, min(h, y2)))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = int01[y1:y2, x1:x2]
    patch = clip_normalize_patch(patch, clip=do_clip)

    if patch.size == 0:
        return 0.0
    if method == "mean_frac":
        m = float(np.mean(patch))
        frac = float(np.mean(patch >= float(strong_thr)))
        return float(w_mean) * m + float(w_frac) * frac

    if method == "p90":
        return float(np.percentile(patch, 90))
    if method == "p95":
        return float(np.percentile(patch, 95))
    if method == "topk_mean":
        flat = patch.reshape(-1)
        k = max(1, int(len(flat) * float(topk)))
        # pega nos k maiores
        top = np.partition(flat, -k)[-k:]
        return float(np.mean(top))

    # default
    return float(np.percentile(patch, 95))

def class_from_score(score: float, t_low: float, t_high: float):
    """
    score em [0,1]
      score >= t_high  => high
      score >= t_low   => medium
      else             => low
    """
    if score >= t_high:
        return "highReflection"
    elif score >= t_low:
        return "mediumReflection"
    else:
        return "lowReflection"

def draw_boxes(img_path: Path, dets, out_path: Path):
    """
    dets: lista de dicts com {x1,y1,x2,y2,label,conf,score}
    """
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    # tenta fonte default; se não houver, usa a do PIL
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for d in dets:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        label = d["label"]
        conf = d["conf"]
        score = d["score"]

        draw.rectangle([x1, y1, x2, y2], outline="white", width=2)
        txt = f"{label} | conf={conf:.2f} | s={score:.2f}"
        # caixa do texto
        tw, th = draw.textbbox((0,0), txt, font=font)[2:]
        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 4, y1], fill="black")
        draw.text((x1 + 2, max(0, y1 - th - 2)), txt, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best_obj_base200.pt (YOLO 1 classe: object)")
    ap.add_argument("--source", required=True, help="pasta com imagens ou uma imagem")
    ap.add_argument("--out", default="outputs/pred_hybrid")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument("--t_low", type=float, default=0.35)
    ap.add_argument("--t_high", type=float, default=0.60)

    ap.add_argument("--int_mode", choices=["hsv_v", "luma", "red", "red_excess"], default="red_excess")
    ap.add_argument("--score", choices=["p95", "p90", "topk_mean", "mean_frac"], default="mean_frac")
    ap.add_argument("--topk", type=float, default=0.05, help="só usado em topk_mean (ex: 0.05 = top 5%)")
    ap.add_argument("--no_clip", action="store_true", help="desliga clipping p1/p99 no patch")

    ap.add_argument("--strong_thr", type=float, default=0.20, help="limiar de pixel forte para frac_strong (0..1)")
    ap.add_argument("--w_mean", type=float, default=0.7)
    ap.add_argument("--w_frac", type=float, default=0.3)


    args = ap.parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    csv_path = outdir / "detections_hybrid.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "final_class", "conf_yolo", "score", "x1", "y1", "x2", "y2"])

        for img_path in iter_images(source):
            int01 = to_intensity_float01(img_path, mode=args.int_mode)

            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False
            )

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            dets_for_draw = []
            for b in r.boxes:
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

                score = intensity_score(
                    int01, x1, y1, x2, y2,
                    method=args.score,
                    topk=args.topk,
                    do_clip=(not args.no_clip),
                    strong_thr=args.strong_thr,
                    w_mean=args.w_mean,
                    w_frac=args.w_frac
                )

                final_class = class_from_score(score, args.t_low, args.t_high)
                writer.writerow([img_path.name, final_class, conf, score, x1, y1, x2, y2])

                dets_for_draw.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "label": final_class, "conf": conf, "score": score
                })

            out_img = outdir / "images" / img_path.name
            draw_boxes(img_path, dets_for_draw, out_img)

    print(f"[OK] Imagens anotadas (híbridas) em: {outdir / 'images'}")
    print(f"[OK] CSV guardado em: {csv_path}")
    print("[DICA] Ajusta --t_low/--t_high e experimenta --score p95/p90/topk_mean.")

if __name__ == "__main__":
    main()
