# main.py
import os
import time
import argparse

from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR
from preprocessing import load_data, preprocess
from detection import detect_objects
from visualization import draw_detections
from export import save_results


def find_png_for_base(input_dir: str, base: str):
    """
    Finds a PNG that belongs to datapoint base.
    Priority:
      1) exact: base + ".png"
      2) any png that starts with base + "_"
      3) any png that starts with base
    """
    exact = os.path.join(input_dir, base + ".png")
    if os.path.exists(exact):
        return exact

    for f in os.listdir(input_dir):
        if f.lower().endswith(".png") and f.startswith(base + "_"):
            return os.path.join(input_dir, f)

    for f in os.listdir(input_dir):
        if f.lower().endswith(".png") and f.startswith(base):
            return os.path.join(input_dir, f)

    return None


def run_watch_loop(input_dir: str, output_dir: str, poll_s: float = 2.0):
    print("=== Sonar Object Detection System ===")
    print(f"[WATCH] Input : {input_dir}")
    print(f"[WATCH] Output: {output_dir}")
    print(f"[WATCH] Poll  : {poll_s:.1f}s")
    print("Press Ctrl+C to stop.\n")

    processed = set()
    os.makedirs(output_dir, exist_ok=True)

    try:
        while True:
            sonar_files = [f for f in os.listdir(input_dir) if f.lower().endswith("_sonar.npy")]

            if not sonar_files:
                print("[INFO] Waiting for *_sonar.npy files...")
                time.sleep(poll_s)
                continue

            for sonar_file in sonar_files:
                base = sonar_file.replace("_sonar.npy", "")
                npy_path = os.path.join(input_dir, sonar_file)

                if base in processed:
                    continue

                png_path = find_png_for_base(input_dir, base)  # may be None

                try:
                    image, data = load_data(png_path, npy_path)
                    image, data = preprocess(image, data)

                    detections = detect_objects(image, data)
                    overlay = draw_detections(image, detections)

                    out_subdir = os.path.join(output_dir, base)
                    save_results(base, out_subdir, detections, overlay)

                    processed.add(base)
                    print(f"[OK] {base}: {len(detections)} object(s) | png={'yes' if png_path else 'no'}")

                except Exception as e:
                    print(f"[ERR] {base}: {e}")

            time.sleep(poll_s)

    except KeyboardInterrupt:
        print("\n[SAFE EXIT] Stopped by user (Ctrl+C). Goodbye!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.out)

    if not os.path.isdir(input_dir):
        print(f"[FATAL] Input folder does not exist: {input_dir}")
        raise SystemExit(1)

    run_watch_loop(input_dir, output_dir, args.poll)


if __name__ == "__main__":
    main()
