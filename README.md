# 🧠 Papers_QA: Medical Paper Question Answering System

**Papers_QA** is an end-to-end question answering pipeline designed to extract insights from a corpus of medical research papers, with a focus on reproductive medicine.

This project combines:

- 🧾 **Mistral-7B-Instruct** (4-bit quantized) for answer generation
- 🔍 **Sentence-BERT** with **FAISS** for dense passage retrieval
- 🧪 **BLEU score** and retrieval accuracy for evaluation

---

## 📁 Project Structure

📦 Papers_QA
├── data/
│   └── generated/
│       └── train_data.csv          # Automatically generated QA pairs
├── notebooks/
│   ├── 1_qa_generation.ipynb       # Generate QAs from medical papers
│   ├── 3_inference.ipynb           # Retrieval + inference + BLEU evaluation
│   └── medqa_training.ipynb        # Fine-tuning Mistral (optional)
├── docs/
│   └── MedQA_Documentation.pdf     # Project overview/report
├── src/                            # Placeholder for reusable code modules
├── requirements.txt                # Project dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file


---

## 💡 Workflow Overview

1. **Question Generation**
   - Extract QA pairs from JSON-formatted medical papers using prompt engineering.
   - _Notebook: `notebooks/1_qa_generation.ipynb`_

2. **Model Fine-tuning (Optional)**
   - Fine-tune the Mistral-7B-Instruct model on domain-specific data.
   - _Notebook: `notebooks/medqa_training.ipynb`_

3. **Retrieval & Inference**
   - Use Sentence-BERT to embed passages and FAISS for nearest neighbor search.
   - Generate answers with Mistral based on retrieved contexts.
   - Evaluate results with BLEU and retrieval correctness.
   - _Notebook: `notebooks/3_inference.ipynb`_

---

## 🚀 Quickstart

```bash
git clone https://github.com/Bechirdardouri/Papers_QA.git
cd Papers_QA
pip install -r requirements.txt
jupyter notebook

    ✅ For GPU inference with quantized models, use environments like Colab, Kaggle, or a local CUDA setup.

📊 Data Notes

    The training data (train_data.csv) is automatically generated using the QA generation notebook.

    The source corpus includes 50 manually curated JSON-format medical papers.

    Prompt strategies are diversified to ensure robustness of generated QA pairs.

📄 License

This project is licensed under the MIT License.
See the LICENSE file for full details.
