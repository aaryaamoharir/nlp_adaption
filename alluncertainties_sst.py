import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np

# =========================
# Load SST-2 dataset + model
# =========================
dataset = load_dataset("glue", "sst2")
val_dataset = dataset["validation"]

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Softmax helper
def compute_probs(logits):
    return F.softmax(logits, dim=-1).squeeze()

# =========================
# Proper uncertainty metrics (all return HIGHER values for MORE uncertainty)
# =========================
def entropy_uncertainty(logits):
    """Standard predictive entropy - higher = more uncertain"""
    probs = compute_probs(logits)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12))
    return entropy.item()

def max_probability_uncertainty(logits):
    """1 - max probability - higher = more uncertain"""
    probs = compute_probs(logits)
    return (1 - torch.max(probs)).item()

def prediction_margin_uncertainty(logits):
    """1 - (max_prob - second_max_prob) - higher = more uncertain"""
    probs = compute_probs(logits)
    top2 = torch.topk(probs, 2).values
    margin = top2[0] - top2[1]
    return (1 - margin).item()

def variance_based_uncertainty(logits):
    """Variance of predictions - higher = more uncertain"""
    probs = compute_probs(logits)
    variance = torch.var(probs)
    return variance.item()

def temperature_scaling_uncertainty(logits, T=2.0):
    """Temperature-scaled entropy - higher = more uncertain"""
    scaled_logits = logits / T
    probs = F.softmax(scaled_logits, dim=-1).squeeze()
    entropy = -torch.sum(probs * torch.log(probs + 1e-12))
    return entropy.item()

def logit_variance_uncertainty(logits):
    """Variance of logits - higher = more uncertain"""
    return torch.var(logits).item()

def hidden_state_norm_uncertainty(hidden_states):
    """Normalized hidden state variance - higher = more uncertain"""
    # Use the mean token representation (excluding padding)
    mean_hidden = torch.mean(hidden_states, dim=0) if len(hidden_states.shape) > 1 else hidden_states
    # Compute variance across dimensions
    uncertainty = torch.var(mean_hidden).item()
    return uncertainty

# =========================
# Precompute uncertainties
# =========================
print("Computing uncertainties for all SST-2 validation samples...")
all_results = []

for i, example in enumerate(val_dataset):
    if i % 200 == 0:
        print(f"Processed {i} samples...")
    
    sentence = example["sentence"]
    true_label = example["label"]

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        # Get the last layer hidden states and handle dimensions properly
        hidden_states = outputs.hidden_states[-1].squeeze(0)  # Remove batch dimension
        if len(hidden_states.shape) > 1:
            hidden_states = torch.mean(hidden_states, dim=0)  # Average over sequence length

    predicted_label = torch.argmax(logits, dim=-1).item()
    correct = (predicted_label == true_label)

    uncertainties = {
        'Entropy': entropy_uncertainty(logits),
        'Max Probability': max_probability_uncertainty(logits),
        'Prediction Margin': prediction_margin_uncertainty(logits),
        'Variance Based': variance_based_uncertainty(logits),
        'Temperature Scaling': temperature_scaling_uncertainty(logits),
        'Logit Variance': logit_variance_uncertainty(logits),
        'Hidden State Uncertainty': hidden_state_norm_uncertainty(hidden_states)
    }

    all_results.append((uncertainties, correct))

print(f"Processed {len(all_results)} total samples.")

# =========================
# Use percentile-based thresholds for better analysis
# =========================
print("\nComputing percentile-based thresholds...")

# Get all uncertainty values for each metric to compute percentiles
all_uncertainties = {metric: [] for metric in all_results[0][0].keys()}
for uncertainties, _ in all_results:
    for metric, value in uncertainties.items():
        all_uncertainties[metric].append(value)

# Convert to numpy arrays and compute percentiles
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
thresholds_by_metric = {}

for metric in all_uncertainties.keys():
    values = np.array(all_uncertainties[metric])
    thresholds_by_metric[metric] = np.percentile(values, percentiles)

# =========================
# Evaluate at each percentile threshold
# =========================
print("\n" + "="*120)
print("SST-2 UNCERTAINTY ANALYSIS RESULTS")
print("Higher uncertainty scores = more uncertain predictions")
print("="*120)

for i, percentile in enumerate(percentiles):
    print(f"\n=== {percentile}th Percentile Threshold ===")
    
    for metric in all_uncertainties.keys():
        thresh = thresholds_by_metric[metric][i]
        
        # Count samples above this threshold (most uncertain samples)
        correct_high_uncertainty = 0
        incorrect_high_uncertainty = 0
        total_high_uncertainty = 0
        
        # Count samples below this threshold (most confident samples)  
        correct_low_uncertainty = 0
        incorrect_low_uncertainty = 0
        total_low_uncertainty = 0
        
        for uncertainties, correct in all_results:
            if uncertainties[metric] >= thresh:  # High uncertainty
                total_high_uncertainty += 1
                if correct:
                    correct_high_uncertainty += 1
                else:
                    incorrect_high_uncertainty += 1
            else:  # Low uncertainty
                total_low_uncertainty += 1
                if correct:
                    correct_low_uncertainty += 1
                else:
                    incorrect_low_uncertainty += 1
        
        # Calculate accuracies
        high_acc = correct_high_uncertainty / total_high_uncertainty if total_high_uncertainty > 0 else 0
        low_acc = correct_low_uncertainty / total_low_uncertainty if total_low_uncertainty > 0 else 0
        
        print(f"{metric:20s} | Thresh: {thresh:.4f} | "
              f"High Unc: {high_acc:.3f} (C:{correct_high_uncertainty:3d}/I:{incorrect_high_uncertainty:3d}) | "
              f"Low Unc: {low_acc:.3f} (C:{correct_low_uncertainty:3d}/I:{incorrect_low_uncertainty:3d}) | "
              f"Diff: {low_acc - high_acc:+.3f}")

# =========================
# Summary statistics
# =========================
print(f"\n" + "="*120)
print("SUMMARY - Expected behavior for SST-2:")
print("- Higher uncertainty should correlate with lower accuracy")
print("- 'Low Unc' accuracy should be higher than 'High Unc' accuracy")  
print("- Positive 'Diff' values indicate the metric is working correctly")
print("- SST-2 is binary sentiment classification (positive/negative)")
print("="*120)

# Print overall accuracy for reference
total_correct = sum(1 for _, correct in all_results if correct)
overall_accuracy = total_correct / len(all_results)
print(f"\nOverall model accuracy on SST-2 validation: {overall_accuracy:.4f} ({total_correct}/{len(all_results)})")

# Print dataset info
print(f"Dataset: SST-2 validation set")
print(f"Model: {model_name}")
print(f"Task: Binary sentiment classification")
