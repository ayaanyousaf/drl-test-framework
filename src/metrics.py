import csv
import os

def export_metrics_csv(metrics, export_dir):
    """
    Automatically exports per-episode metrics to CSV.
    Gets metrics from evaluate function.
    """
    
    path = f"{export_dir}/metrics.csv"

    if not metrics or not isinstance(metrics, list): 
        print("No episode metrics found.")
        return

    with open(path, "w", newline="") as file: 
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"Exported per-episode metrics to: {path}")
