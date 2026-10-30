# Supervised Learning for Intent-Based Call Routing

## Project Overview
In modern Contact Center as a Service (CCaaS) environments, initial call routing is frequently plagued by static touch-tone IVR menus or manual intake, causing misrouted interactions, inflated Average Handle Time (AHT), and customer friction. This project builds a supervised machine learning classification framework to automatically map unstructured customer utterances to precise intent categories at the point of entry.

## Research Question
Can a supervised machine learning classification framework—leveraging natural language text preprocessing, TF-IDF feature vectorization, and multi-class classification algorithms—accurately predict customer intent from raw text transcripts to automate inbound call and chat routing within a CCaaS environment?

## Dataset Information
- **Primary Source:** Bitext Training Dataset for Chatbots and Virtual Assistants[cite: 2] (comprising domain-specific customer support utterances mapped across detailed intent classes and high-level categories).
- **Input Feature ($X$):** Raw customer message text strings.
- **Target Label ($y$):** Discrete intent categories (e.g., `payment_issue`, `cancel_subscription`, `track_order`, `reset_password`).

## Methodology & Pipeline
1. **Exploratory Data Analysis (EDA):** Inspected class balance across intent categories to evaluate skew and analyzed token length distributions to establish vocabulary boundaries.
2. **Text Preprocessing:** Case folding, punctuation removal, URL stripping, and whitespace normalization.
3. **Feature Vectorization:** Extracted numerical feature representations using `TfidfVectorizer` with unigram and bigram ranges (`ngram_range=(1, 2)`), sub-linear term frequency scaling, and frequency cutoffs (`min_df=2`, `max_df=0.95`) to eliminate sparse noise.
4. **Supervised Modeling:** Evaluated three baseline algorithms:
   - Multinomial Naive Bayes (`MultinomialNB`)
   - Logistic Regression (`LogisticRegression`)
   - Linear Support Vector Classifier (`LinearSVC`)

## Results & Evaluation Summary
- **Evaluation Metrics:** Evaluated using Accuracy, Macro F1-Score, and Weighted F1-Score on a held-out stratified test set (20%).
- **Performance:** Linear classifiers (`LinearSVC` and regularized `Logistic Regression`) achieved superior generalization performance, surpassing the program target with a **Macro F1-Score ≥ 0.85**.
- **Error Analysis:** Confusion matrix inspection revealed minor semantic overlap between adjacent financial categories (e.g., distinguishing between invoice disputes and general payment issues), highlighting areas where secondary clarification prompts can be introduced in production.

## Repository Structure
```text
├── data/                    # Dataset directory
├── notebooks/               # Jupyter notebooks containing EDA and modeling
├── README.md                # Project documentation and results
└── requirements.txt         # Python dependencies (pandas, scikit-learn, matplotlib, seaborn)
