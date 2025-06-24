# 🧠 Papers_QA: Medical Paper Question Answering System

**Papers_QA** is an end-to-end AI-powered pipeline that automates the extraction of meaningful insights from a collection of medical research papers, specifically in the field of reproductive medicine. The system leverages large language models, semantic similarity search, and robust evaluation metrics to generate, retrieve, and assess question-answer (QA) pairs.

This project aims to support medical researchers, students, and professionals by providing quick access to structured information extracted from unstructured scientific literature.

---

## 🎯 Objective

The core objective of Papers_QA is to:
- Automatically generate high-quality question-answer pairs from medical research articles.
- Use state-of-the-art embedding and retrieval systems to fetch the most relevant passages.
- Enable natural language understanding and reasoning over dense, domain-specific content.
- Provide a reproducible and extensible platform for QA research in the medical domain.

---

## 🚀 Key Features

- 🧾 **Mistral-7B-Instruct (4-bit quantized)**: A large language model used for generating detailed answers from context passages.
- 🔍 **Sentence-BERT + FAISS**: Combines semantic sentence embeddings and efficient vector indexing for high-quality context retrieval.
- 🧪 **BLEU Scoring & Retrieval Accuracy**: Evaluates answer quality and checks whether the model selects the correct context from the document corpus.
- 📊 **Synthetic Training Data Generation**: Automatically produces a QA dataset from real research papers using custom prompt engineering strategies.
- 🧠 **Support for Fine-tuning, Inference & Evaluation**: Modular notebooks for experimenting with model training, evaluation pipelines, and iterative QA improvement.

---

## 📁 Project Structure

```plaintext
📦 Papers_QA
├── data/
│   └── generated/
│       └── train_data.csv          # QA pairs generated from medical papers
├── notebooks/
│   ├── 1_qa_generation.ipynb       # Generate QA pairs using LLM + prompts
│   ├── 3_inference.ipynb           # Retrieve context, answer questions, evaluate
│   └── medqa_training.ipynb        # Optional fine-tuning of Mistral
├── docs/
│   └── MedQA_Documentation.pdf     # Overview of architecture, method, and results
├── src/
│   └── (Coming soon) Python modules for reuse and integration
├── requirements.txt                # Python dependencies (transformers, faiss, etc.)
├── LICENSE                         # Open-source license (MIT)
└── README.md                       # Project overview and instructions
