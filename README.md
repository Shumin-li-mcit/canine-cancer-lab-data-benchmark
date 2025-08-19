# Assessing the Feasibility of Early Cancer Detection Using Routine Laboratory Data

This repository contains the full source code and analysis for the paper: **"Assessing the Feasibility of Early Cancer Detection Using Routine Laboratory Data: A Critical Evaluation of Machine Learning Approaches on an Imbalanced Dataset."**

This study was designed to establish a rigorous performance benchmark for cancer risk classification in dogs using only routine bloodwork (CBC and biochemistry panels).

---

## 🔬 Abstract

Cancer is a leading cause of mortality in the aging canine population. This study leverages the longitudinal cohort from the Morris Animal Foundation's Golden Retriever Lifetime Study (GRLS) to assess the feasibility of building a cancer risk classification model from routine lab data. Through a comprehensive and transparent comparison of 105 unique machine learning pipelines, we determined the maximal predictive performance achievable with this data modality. Our findings show that while a statistically significant signal exists, it is insufficient for building a clinically reliable classification tool due to the non-specific nature of the biomarkers and the profound challenge of confounding by treatment in observational data. This work serves as a critical benchmark and a methodological guide for future research in veterinary computational oncology.

---

## 📊 Key Findings

The final optimized model (a Logistic Regression with class weighting) demonstrated a clear divergence between its ability to rank patients by risk and its ability to accurately classify them.

* **Good Discriminatory Power:** The model could distinguish between cancer-positive and cancer-negative visits reasonably well, achieving a **ROC-AUC of 0.809**.
* **Poor Classification Performance:** This discriminatory ability did not translate into a clinically useful tool. The model had:
    * A critically low **F1-Score of 0.24**.
    * An extremely low **Positive Predictive Value (PPV) of 0.14**, meaning 86% of positive predictions were false alarms.
* **Conclusion:** Routine laboratory data alone is **not sufficient** for reliable cancer screening in dogs. The future of this research lies in multi-modal data integration.

---

## 🗂️ Data Source

The data used in this study is from the **Morris Animal Foundation's Golden Retriever Lifetime Study (GRLS)**. Due to the sensitive nature of the data, it cannot be hosted in this repository.

Researchers can request access to the data directly from the source:
* **GRLS Data Commons Portal:** [https://www.morrisanimalfoundation.org/golden-retriever-lifetime-study-data-commons-portal](https://www.morrisanimalfoundation.org/golden-retriever-lifetime-study-data-commons-portal)

---

## ⚙️ Installation & Setup

To replicate this analysis, please follow these steps to set up the environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/canine-cancer-lab-data-benchmark.git
    cd canine-cancer-lab-data-benchmark
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Reusable Code

1.  **`data_processing.py`**: This file contains code that perfoms data preprocessing for final dataset, including sex_status encoding, MICE imputation, feature engineering.
2.  **`modeling.py`**: This file contains code that implements a comprehensive comparative framework to identify the best-performing combination of machine learning algorithm, feature set, and data balancing technique, with final model evaluation and SHAP interpretation. 

NOTE: In order to fully replicate the analysis, please refer back to the paper for details on data curation to get the final dataset. 

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
