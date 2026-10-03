import time
import torch
from sklearn.metrics import accuracy_score, f1_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_cpu_inference_time(model, sample_input, num_runs=50):
    model.eval()
    model.to('cpu')
    sample_input = sample_input.to('cpu')
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)
            
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / num_runs) * 1000  # ms
    return avg_latency

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return acc, macro_f1