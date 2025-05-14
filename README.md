# 🧠 Medical QA System Using Mistral & FAISS

This project builds a pipeline for Question Answering (QA) using:

- 🧾 Mistral-7B-Instruct (4-bit quantized) for answer generation
- 🔍 Sentence-BERT + FAISS for document retrieval
- 🧪 BLEU-based evaluation

## 📁 Structure

- `notebooks/`: Original Kaggle notebooks
- `src/`: Modular code for each pipeline step
- `data/`: Raw and processed JSON/CSV files
- `outputs/`: Evaluation results

## 💡 Workflow

1. Generate QA from papers (`1_qa_generation.ipynb`)
2. Fine-tune Mistral (`2_mistral_finetuning.ipynb`)
3. Evaluate performance (`3_inference_and_eval.ipynb`)

## 🚀 Quickstart

```bash
git clone https://github.com/your-username/qa-medical-papers.git
cd qa-medical-papers
pip install -r requirements.txt
python main.py
