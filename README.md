# MIA-DP-Implementation
<p align="center">
  <!-- BADGES:START -->
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue"></a>
  <a href="#"><img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-orange"></a>
  <a href="#"><img alt="NumPy" src="https://img.shields.io/badge/numpy-1.x-013243"></a>
  <a href="#"><img alt="Pandas" src="https://img.shields.io/badge/pandas-2.x-150458"></a>
  <a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/matplotlib-3.x-11557c"></a>
  <a href="#"><img alt="Joblib" src="https://img.shields.io/badge/joblib-1.x-6b7f9c"></a>
  <a href="#"><img alt="adjustText (optional)" src="https://img.shields.io/badge/adjustText-optional-999999"></a>
  <!-- BADGES:END -->
</p>

# Membership Inference Attacks for Regression

Machine learning models are increasingly used in sensitive areas like healthcare and insurance, where protecting personal data is crucial. However, even when data is not directly shared, trained models can unintentionally leak information about the samples used during training. Despite mathematical frameworks like differential privacy expanding in usage, one way such information leakage occurs is through Membership Inference Attacks (MIAs) [3], where an adversary attempts to determine whether a specific record was part of a model’s training set.

We investigate data insecurity of machine learning models by implementing a membership inference attack (MIA) on the UCI Wine Quality (white wine) dataset [1]. The dataset contains physicochemical attributes used to predict wine quality, a continuous-valued target, making it suitable for regression tasks.

In this project, we implement a black-box Membership Inference Attack (MIA) on a Random Forest regression model trained on the UCI wine dataset. We evaluate how vulnerable the non-private model is to membership inference and examine the privacy–utility relationship by analyzing the attack’s AUC alongside the model’s test RMSE.

## Dataset

- UCI Machine Learning Repository: *Wine Quality (White Wine) Dataset*  
  Features: 11 physicochemical attributes (e.g., acidity, residual sugar, sulfur dioxide, alcohol)  
  Target: `quality` (integer 0–10, treated as a continuous regression variable)

## What’s included

- **Preprocessing**: deduplication; standardization of all continuous features; deterministic train/val/test/hold-out splits.
- **Target model**: `RandomForestRegressor` trained on the member (training) split.
- **Attack dataset construction**: black-box queries to the trained model to compute per-sample attack features (prediction, absolute error, squared error, prediction variance).
- **Membership inference attacks**:
  - **ART loss-based black-box attack** using `ScikitlearnRegressor` and `MembershipInferenceBlackBox`.
  - **Baseline logistic regression attack** trained on handcrafted error-based features.
- **Evaluation**: AUC and accuracy of the attacks; confusion matrix; ROC curves; member vs. non-member loss distribution analysis.
- **Figures**: loss histograms, ROC curves, and summary tables for target model and attack performance.

## Repo layout

```
data/
    raw/whitequality-white.csv
    processed/
results/
    runs/
src/
    mia_regression/cli/
        run_attack_target.py
        run_prepare_data.py
        run_train_attack.py
        run_train_target_baseline.py
        run_train_attack_art.py
        run_build_attack_dataset_blackbox.py
```

## Quickstart

### 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Prepare data
```bash
### 1) Prepare data
```bash
python -m src.mia_regression.cli.run_prepare_data \
  --csv data/raw/winequality-white.csv \
  --outdir data/processed \
  --runs_dir results/runs \
  --seed 42
```

### 2) Train target model
```bash
python -m src.mia_regression.cli.run_train_target_baseline \
  --n_estimators 300 \
  --seed 42
```

### 3) Build attack dataset (black-box queries)
```bash
python -m src.mia_regression.cli.run_build_attack_dataset_blackbox \
  --splits data/processed/processed_splits.npz \
  --target_model results/runs/<timestamp>/target_model.joblib \
  --outdir data/processed
```

### 4) Attack the target
(a) ART loss-based attack
```bash
python -m src.mia_regression.cli.run_train_attack_art \
  --splits data/processed/processed_splits.npz \
  --target_model results/runs/<timestamp>/target_model.joblib \
  --outdir results/runs \
  --attack_model_type rf \
  --seed 42
```
(b) Logistic regression baseline
```bash
python -m src.mia_regression.cli.run_train_attack \
  --attack_npz data/processed/attack_dataset_regression.npz \
  --outdir results/runs \
  --seed 42
```

### 5) Evaluate attack on the target model
```bash
python -m src.mia_regression.cli.run_attack_target \
  --splits data/processed/processed_splits.npz \
  --target_model results/runs/<timestamp>/target_model.joblib \
  --attack_model results/runs/<timestamp>/attack_model.joblib
```

## Plots

- **Loss distribution histogram**: `graphs/loss_hist.pdf`  
  - Shows the empirical distributions of absolute prediction error \(|y - \hat{y}|\) for training samples (members) and test samples (non-members).  
  - Training points exhibit consistently lower error, providing the statistical signal exploited by membership inference attacks.

- **ROC curve — ART attack**: `graphs/attack_roc_art.pdf`  
  - ROC curve for the ART loss-based membership inference attack.  
  - Includes the random-guessing baseline at AUC = 0.5.  
  - Demonstrates that the ART attack significantly outperforms random guessing.

- **ROC curve — logistic regression baseline**: `graphs/attack_roc.pdf`  
  - ROC curve for the feature-based logistic regression MIA baseline.  
  - Shows the classifier’s ability to separate members from non-members using handcrafted error features.  
  - Also includes the AUC = 0.5 random baseline for comparison.

## Results

- **Target model performance**:  
  - Train RMSE = **0.264**, Test RMSE = **0.694**  
  - The gap indicates moderate overfitting, which creates conditions favorable for membership inference.

- **ART loss-based attack**:  
  - Attack AUC ≈ **0.807**  
  - Accuracy ≈ **0.818**  
  - Successfully distinguishes members from non-members far better than random guessing.

- **Logistic regression baseline attack**:  
  - Attack AUC ≈ **0.853**  
  - Accuracy ≈ **0.749**  
  - Shows that even simple error-based features can reveal membership information.

**Conclusion**:  
Both the ART attack and the baseline logistic regression attack demonstrate that the non-private Random Forest regressor leaks membership information. Training samples consistently incur lower prediction errors, enabling effective membership inference.

## Requirements

- Python 3.9+
- NumPy, Pandas, scikit-learn, Matplotlib, Joblib  
- Optional: `adjustText` (for improved label placement in plots)

Install:
```bash
pip install -r requirements.txt
# optional
pip install adjustText
```

## Notes

- Paths are robust to working directory (scripts resolve the repo root).
- `data/processed/processed_splits.npz` and models under `results/runs/` are generated artifacts.
- For reproducibility, keep the `--seed` fixed across steps.

## Acknowledgments
This work uses the [UCI Wine Quality (White Wine) Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) [1] and draws conceptual guidance from the University of Liverpool’s reading on Membership Inference Attacks and Differential Privacy [3].  
Implementation and experimentation leveraged open-source tools including [scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) [2][4][5].

---

**References**

[1]: Cortez, Paulo, António Cerdeira, Fernando Almeida, Telmo Matos, and José Reis. “Wine Quality Dataset.” *UCI Machine Learning Repository*, 2009. (https://archive.ics.uci.edu/dataset/186/wine+quality)

[2]: Trusted-AI. *Adversarial Robustness Toolbox (ART) – Python Library for Machine Learning Security*. [https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)  

[3]: University of Liverpool. *Membership Inference Attacks and Differential Privacy*. [https://cgi.csc.liv.ac.uk/~acps/documents/readinggroup1.pdf](https://cgi.csc.liv.ac.uk/~acps/documents/readinggroup1.pdf) 
 
[4]: *Adversarial Robustness Toolbox Documentation – Membership Inference Module*. [https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html)  

[5]: Alencaster, Castillo Hernandez, Silva. “MIA-DP-Implementation.” [https://github.com/0x10jalencas/MIA-DP-Implementation](https://github.com/0x10jalencas/MIA-DP-Implementation)

[6]: Alencaster, Jess, Eman Castillo Hernandez, and Adan Silva. *Membership Inference Attacks for Regression*. 
Available at: [https://drive.google.com/file/d/1zuBDftke-K5pN01g3rp-FDi58e6v5v5-/view?usp=sharing](https://drive.google.com/file/d/1zuBDftke-K5pN01g3rp-FDi58e6v5v5-/view?usp=sharing)
