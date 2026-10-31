# Supervised Learning for Intent-Based Call & Chat Routing in Contact Centers (CCaaS)

## Project Overview
In modern Contact Center as a Service (CCaaS) environments, initial call routing is frequently plagued by static touch-tone IVR menus or manual intake, causing misrouted interactions, inflated Average Handle Time (AHT), and customer friction. This project builds a robust supervised machine learning classification framework to automatically map unstructured customer utterances to precise intent categories and high-level departments at the point of entry.

---

## Research Question
Can a supervised machine learning classification framework—leveraging natural language text preprocessing, sub-linear TF-IDF feature vectorization, and multi-class classification algorithms—accurately predict customer intent from raw text transcripts to automate inbound call and chat routing within a CCaaS environment?

---

## Dataset Information & Structure
- **Primary Source:** Bitext Training Dataset for Chatbots and Virtual Assistants (comprising domain-specific customer support utterances mapped across 27 detailed intent classes and 11 high-level functional categories).
- **Input Feature ($X$):** Raw customer message text strings (`utterance`).
- **Target Label ($y$):** Discrete intent categories (e.g., `payment_issue`, `cancel_subscription`, `track_order`, `reset_password`).
- **Metadata/Flags:** Linguistic indicators (slang, syntax complexity, spelling errors) used during exploratory error analysis.

---

## Methodology & Pipeline Architecture
1. **Exploratory Data Analysis (EDA):** Inspected class balance across all 27 intent categories to evaluate skew and analyzed token length distributions via KDE and IQR fences to establish vocabulary boundaries.
2. **Text Preprocessing & Normalization:** Case folding, punctuation removal, URL stripping, and whitespace normalization via regular expressions.
3. **Feature Vectorization:** Extracted numerical feature representations using `TfidfVectorizer` with unigram and bigram ranges (`ngram_range=(1, 2)`), sub-linear term frequency scaling (`sublinear_tf=True`), and frequency cutoffs (`min_df=2`, `max_df=0.95`) to eliminate sparse noise resulting in 1,869 features.
4. **Supervised Modeling & Validation:** Evaluated candidate algorithms using 5-fold stratified cross-validation and hyperparameter grid search:
   - Multinomial Naive Bayes (`MultinomialNB`)
   - Random Forest Classifier (`RandomForestClassifier`)
   - Regularized Logistic Regression (`LogisticRegression`)
   - Linear Support Vector Classifier (`LinearSVC`)

---

## Results & Evaluation Summary
- **Evaluation Metrics:** Evaluated using Accuracy, Macro $F_1$-Score, Weighted $F_1$-Score, Macro Precision, and Macro Recall on a held-out stratified test set (20% / 1,635 samples).
- **Performance Comparison Table:**
  | Model Architecture | 5-Fold CV Macro $F_1$ | Test Accuracy | Macro $F_1$-Score | Weighted $F_1$-Score |
  | :--- | :---: | :---: | :---: | :---: |
  | Multinomial Naive Bayes | 0.9857 | 98.78% | 0.9878 | 0.9877 |
  | Random Forest (100 Trees) | 0.9881 | 98.59% | 0.9859 | 0.9859 |
  | Logistic Regression (Tuned $C=10.0$) | 0.9964 | 99.69% | 0.9970 | 0.9969 |
  | **Linear Support Vector Classifier ($C=1.0$)** | **0.9964** | **99.69%** | **0.9969** | **0.9969** |

- **Key Findings:** The Linear Support Vector Classifier (`LinearSVC`) and tuned Logistic Regression emerged as the top-performing models, achieving a **Macro $F_1$-Score of 0.9969**, significantly outperforming the project benchmark target (Macro $F_1 \ge 0.85$).
- **Error Analysis:** Inspection of the confusion matrix and classification reports demonstrated near-perfect precision and recall across all 27 intent classes. Minor edge-case misclassifications ($5$ total errors out of $1,635$ test records, or a $0.31\%$ error rate) were isolated strictly to closely related account lifecycle tasks (e.g., `switch_account` vs. `create_account`) and shipping updates due to overlapping vocabulary patterns.

---

## Business Value & Strategic Recommendations
1. **Automated Tier-1 Deflection:** Inquiries predicted with confidence $> 0.95$ for routine self-service tasks (`recover_password`, `track_order`, `get_invoice`) should be routed directly to automated self-service bots, deflecting up to 35% of ticket volume from live agents.
2. **Priority Churn-Risk Routing:** High-stakes intents (`delete_account`, `complaint`, `cancel_order`) must bypass standard tier-1 queues and route immediately to Senior Retention Specialists.
3. **Threshold-Based Disambiguation:** For predictions falling below a confidence threshold (e.g., lexical ambiguity between `cancel_order` and `check_cancellation_fee`), present a single-tap confirmation prompt to the user before routing.

---

## Repository Structure
```text
├── data/
│   └── bitext_customer_support_training_dataset.csv   # Primary customer interaction dataset
├── notebooks/
│   └── capstone_eda_baseline_model.ipynb              # EDA baseline model notebook
│   └── capstone_project_final_report.ipynb            # Fully documented technical analysis notebook
└── README.md                                          # Executive nontechnical capstone summary report