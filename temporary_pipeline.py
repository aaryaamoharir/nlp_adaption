import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet

# found wordnet for synonyms 
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonym(word):
    synsets = wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                return lemma.name().replace('_', ' ')
    return word

#applys transformations 
def transform_text(text):
    
    # expand contractions
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    
    # normalize punctuation and whitespace 
    text = re.sub(r'[!?.]{2,}', '.', text)
    
    text = ' '.join(text.split())
    
    # replace synonyms 
    words = text.split()
    new_words = [get_synonym(w) for w in words]
    return ' '.join(new_words)

#sst-2 dataset (validation only) and model 
dataset = load_dataset("glue", "sst2")
val_dataset = dataset["validation"]
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# using softmax uncertainty 
def top1_uncertainty(probs):
    return 1 - torch.max(probs).item()

#run inference 
uncertainty_threshold = 0.25
results = []

prev_correct_high_uncertainty = 0
new_correct_from_transform = 0

for example in val_dataset:
    sentence = example["sentence"]
    true_label = example["label"]

    # og prediction 
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze()
    pred_label = torch.argmax(probs).item()
    uncertainty = top1_uncertainty(probs)

    # Track previously correct high-uncertainty predictions
    if uncertainty > uncertainty_threshold and pred_label == true_label:
        prev_correct_high_uncertainty += 1

    # if uncert then apply transformations 
    transformed_sentence = sentence
    transformed_pred_label = pred_label
    transformed_probs = probs
    transformed_uncertainty = uncertainty

    if uncertainty > uncertainty_threshold:
        transformed_sentence = transform_text(sentence)
        inputs_t = tokenizer(transformed_sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits_t = model(**inputs_t).logits
        transformed_probs = F.softmax(logits_t, dim=-1).squeeze()
        transformed_pred_label = torch.argmax(transformed_probs).item()
        transformed_uncertainty = top1_uncertainty(transformed_probs)

        # Track newly corrected predictions
        if pred_label != true_label and transformed_pred_label == true_label:
            new_correct_from_transform += 1

    results.append({
        "original_sentence": sentence,
        "transformed_sentence": transformed_sentence,
        "true_label": true_label,
        "original_pred_label": pred_label,
        "original_probs_neg": probs[0].item(),
        "original_probs_pos": probs[1].item(),
        "original_uncertainty": uncertainty,
        "transformed_pred_label": transformed_pred_label,
        "transformed_probs_neg": transformed_probs[0].item(),
        "transformed_probs_pos": transformed_probs[1].item(),
        "transformed_uncertainty": transformed_uncertainty
    })

# saves to csv just incase and also prints out statistics 
df = pd.DataFrame(results)
df.to_csv("sst2_transformed_inference.csv", index=False)
print("Results saved to 'sst2_transformed_inference.csv'")
total = len(results)
correct_original = sum(r["true_label"] == r["original_pred_label"] for r in results)
correct_transformed = sum(r["true_label"] == r["transformed_pred_label"] for r in results)

print(f"Original accuracy: {correct_original / total:.4f}")
print(f"Transformed accuracy: {correct_transformed / total:.4f}")
print(f"Previously correct high-uncertainty inputs: {prev_correct_high_uncertainty}")
print(f"New correct predictions after transformation: {new_correct_from_transform}")

