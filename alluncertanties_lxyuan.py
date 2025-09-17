#wrote the initial code on my own and used the uncertainty metrics from input adaption project and then asked claude to generate code for the percentile based thresholds instead of fixed values
#these are the outputs from the 95% percentile threshold 
#Entropy              | Thresh: 1.0801 | High Unc: 0.352 (C: 255/I: 469) | Low Unc: 0.553 (C:7593/I:6148) | Diff: +0.200
#Max Probability      | Thresh: 0.5968 | High Unc: 0.362 (C: 262/I: 462) | Low Unc: 0.552 (C:7586/I:6155) | Diff: +0.190
#Prediction Margin    | Thresh: 0.9683 | High Unc: 0.349 (C: 253/I: 471) | Low Unc: 0.553 (C:7595/I:6146) | Diff: +0.203
#Variance Based       | Thresh: 0.3058 | High Unc: 0.913 (C: 661/I:  63) | Low Unc: 0.523 (C:7187/I:6554) | Diff: -0.390
#Temperature Scaling  | Thresh: 1.0939 | High Unc: 0.351 (C: 254/I: 470) | Low Unc: 0.553 (C:7594/I:6147) | Diff: +0.202
#Logit Variance       | Thresh: 7.6717 | High Unc: 0.909 (C: 658/I:  66) | Low Unc: 0.523 (C:7190/I:6551) | Diff: -0.386
#Hidden State Uncertainty | Thresh: 0.3090 | High Unc: 0.883 (C: 639/I:  85) | Low Unc: 0.525 (C:7209/I:6532) | Diff: -0.358
# I = incorrect, C = correct and dataset has 3 labels (positive, neutral, and negative)
# high unc: is uncertainty >= threshold and low unc: is uncertainty < threshold
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np

# =========================
# Load dataset + model
# =========================
dataset = load_dataset("tyqiangz/multilingual-sentiments", "all")
val_dataset = dataset["test"]   # use test set (labeled)

model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
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
print("Computing uncertainties for all samples...")
all_results = []

for i, example in enumerate(val_dataset):
    if i % 1000 == 0:
        print(f"Processed {i} samples...")
    
    sentence = example["text"]
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
print("\n" + "="*80)
print("UNCERTAINTY ANALYSIS RESULTS")
print("Higher uncertainty scores = more uncertain predictions")
print("="*80)

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
              f"High Unc: {high_acc:.3f} (C:{correct_high_uncertainty:4d}/I:{incorrect_high_uncertainty:4d}) | "
              f"Low Unc: {low_acc:.3f} (C:{correct_low_uncertainty:4d}/I:{incorrect_low_uncertainty:4d}) | "
              f"Diff: {low_acc - high_acc:+.3f}")

# =========================
# Summary statistics
# =========================
print(f"\n" + "="*80)
print("SUMMARY - Expected behavior:")
print("- Higher uncertainty should correlate with lower accuracy")
print("- 'Low Unc' accuracy should be higher than 'High Unc' accuracy")  
print("- Positive 'Diff' values indicate the metric is working correctly")
print("="*80)

# Print overall accuracy for reference
total_correct = sum(1 for _, correct in all_results if correct)
overall_accuracy = total_correct / len(all_results)
print(f"\nOverall model accuracy: {overall_accuracy:.4f} ({total_correct}/{len(all_results)})")
