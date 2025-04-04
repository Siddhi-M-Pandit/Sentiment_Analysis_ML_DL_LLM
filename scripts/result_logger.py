import pandas as pd
import os

def log_results(model_name, dataset_name, 
                accuracy, precision, recall, f1, 
                tuning_time=None, train_time=None, inference_time=None,  latency=None,
                cpu_usage=None, mem_usage=None,
                gpu_usage=None, gpu_mem_usage=None, gpu_temp=None, power=None,
                num_features=None, hyperparams=None,
                path="../results/results.csv"):
    
    # Append model evaluation metrics to a CSV for final comparison
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = {
        'Model': model_name,
        'Dataset': dataset_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'Tuning Time (s)': round(tuning_time, 4) if tuning_time else None,
        'Training Time (s)': round(train_time, 4) if train_time else None,
        'Inference Time per Sample (s)': round(inference_time, 6) if inference_time else None,
        'Latency (ms)': round(latency, 2) if latency else None,
        'CPU Usage (%)': round(cpu_usage, 2) if cpu_usage is not None else None,
        'Memory Usage (%)': round(mem_usage, 2) if mem_usage is not None else None,
        'GPU Usage (%)': round(gpu_usage, 2) if gpu_usage is not None else None,
        'GPU Memory Usage (%)': round(gpu_mem_usage, 2) if gpu_mem_usage is not None else None,
        'GPU Temp (C)': round(gpu_temp, 2) if gpu_temp is not None else None,
        'Power Consumption (W)': round(power, 2) if power is not None else None,
        'Features': num_features,
        'Hyperparameters': str(hyperparams) if hyperparams else None
    }

    row_df = pd.DataFrame([row])

    # Append or create file
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, row_df], ignore_index=True)
    else:
        df = row_df

    df.to_csv(path, index=False)
    print(f"Logged: {model_name} on {dataset_name} to {path}")
