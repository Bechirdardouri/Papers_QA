

# 🧠 Papers\_QA: Medical Paper Question Answering System

**Papers\_QA** is an end-to-end question answering pipeline tailored for extracting insights from a corpus of medical research papers in reproductive medicine.

This project combines:

* 🧾 **Mistral-7B-Instruct** (4-bit quantized) for natural language answer generation
* 🔍 **Sentence-BERT** with **FAISS** for dense retrieval
* 🧪 **BLEU score** and retrieval accuracy for evaluation

---

## 📁 Project Structure

```
📦 Papers_QA
├── 1_qa_generation.ipynb         # QA pair generation from JSON papers
├── 3_inference.ipynb             # Retrieval + QA inference and evaluation
├── train_data(2).csv             # Training QA dataset
├── medqa-training(1).ipynb       # Experimental training notebook
├── README.md                     # Project overview
├── LICENSE                       # MIT License
```

> ⚠️ `2_mistral_training.ipynb` was removed for cleanup — training handled externally or in `medqa-training`.

---

## 💡 Workflow Overview

1. **Question Generation**
   Extract QA pairs from JSON-formatted medical papers using prompt engineering.

   > Notebook: `1_qa_generation.ipynb`

2. **Model Fine-tuning**
   (Optional) Fine-tune Mistral-7B-Instruct on domain-specific QA data.

   > Notebook: `medqa-training(1).ipynb`

3. **Retrieval & Inference**

   * Use Sentence-BERT embeddings + FAISS to retrieve relevant context
   * Generate answers with Mistral
   * Evaluate using BLEU and retrieval correctness

   > Notebook: `3_inference.ipynb`

---

## 🚀 Quickstart

```bash
git clone https://github.com/Bechirdardouri/Papers_QA.git
cd Papers_QA
pip install -r requirements.txt  # (prepare this file manually if needed)
```

> For GPU-based inference with quantized models, use a runtime like **Kaggle**, **Colab**, or a local setup with CUDA.

---

## 📝 Notes

* Dataset: 50 medical research papers in JSON, manually curated.
* QA generation uses multiple prompt strategies for robustness.
* Retrieval is evaluated based on whether the correct context is selected before answering.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

