from baseline_redexcess import run_folder

if __name__ == "__main__":
    run_folder(
    input_dir="data/images/test",
    out_dir="outputs/pred_baseline_redexcess/images",
    csv_path="outputs/pred_baseline_redexcess/detections_baseline.csv",
    thr_method="percentile",
    p=98.0,
    median_ksize=3,
    min_area=10
)
