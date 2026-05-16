# Emotion Detection in Social Media with RAG Justification

**Bachelor Semester Project 5 — University of Luxembourg**  
**Student:** João Bernardo Sousa Faria  
**Supervisor:** Salima Lamsiyah

---

## Overview
A system that detects emotions (anger, joy, sadness, etc.) in Reddit comments 
and explains its predictions using Retrieval-Augmented Generation (RAG), 
making the model transparent and interpretable instead of a black box.

---

## How it works
1. **Emotion Classification** — Fine-tuned RoBERTa on the GoEmotions dataset 
   for multi-label emotion classification (28 emotion categories)
2. **Threshold Optimization** — Per-label decision thresholds to handle class 
   imbalance, significantly improving F1 scores
3. **RAG Justification** — Retrieves semantically similar posts from the 
   training set and uses them as evidence to explain predictions

---

## Results

| Setting | DEV Micro-F1 | DEV Macro-F1 | TEST Micro-F1 | TEST Macro-F1 |
|---|---|---|---|---|
| Fixed threshold (0.5) | 0.510 | 0.469 | 0.501 | 0.453 |
| Tuned per-label thresholds | 0.598 | 0.540 | 0.597 | 0.521 |

> The GoEmotions baseline documented F1-score is 0.46 — our optimized model surpasses it.

---

## Tech Stack
- Python, PyTorch, HuggingFace Transformers
- RoBERTa (transformer-based classifier)
- Sentence-BERT (semantic similarity)
- GoEmotions Dataset (58,000 Reddit comments, 28 emotion labels)

---

## Project Structure
```
BSP_S5/
├── BSP_S5.ipynb          # Main notebook
├── dataset/              # GoEmotions dataset and preprocessing
├── results/              # Evaluation results
```
