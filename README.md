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

We investigate data insecurity of machine learning models by implementing an MIA on the Health Insurance Charges Dataset from Kaggle [1]. The dataset includes demographic and health-related information used to predict insurance charges, which is a continuous value, making it suitable for regression tasks.

We implement a black-box **Membership Inference Attack (MIA)** on a **regression** model trained on the Kaggle Health Insurance Charges dataset. It quantifies the **privacy–utility trade-off** by sweeping Random Forest regularization and reporting **Attack AUC vs Test RMSE**.

## Dataset

- Kaggle: *Health Insurance Charges Dataset*  
  Columns: `age, sex, bmi, children, smoker, region, charges`  
  Task: predict `charges` (continuous)

## What’s included

- **Preprocessing**: cleaning, one-hot encoding (categoricals), scaling (numerics), deterministic splits.
- **Target model**: RandomForestRegressor (CLI-configurable).
- **Shadow models + attack**: shadow RFs produce attack features; logistic regression attack.
- **Evaluation**: attack on target (AUC/ACC, confusion), ROC, error histograms.
- **Privacy–utility sweep**: AUC vs RMSE across RF configs, optional multi-seed averaging.
- **Publication-quality plots**: scatter-only, AUC=0.5 baseline, CI bars, Pareto frontier overlay with de-overlapped labels.

## Repo layout

```
data/
    raw/insurance.csv
    processed/
notebooks/
    00_eda.ipynb
results/
    runs/
    sweeps/
scripts/
    plot_privacy_utility.py
src/
    mia_regression/cli/
        run_attack_target.py
        run_prepare_data.py
        run_train_attack.py
        run_train_shadows.py
        run_train_target_baseline.py
```

## Quickstart

### 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Put the Kaggle CSV at: data/raw/insurance.csv
```

### 1) Prepare data
```bash
python -m src.mia_regression.cli.run_prepare_data   --csv data/raw/insurance.csv   --outdir data/processed   --runs_dir results/runs   --seed 42 --shadow_pool_frac 0.6
```

### 2) Train target model
```bash
# Can adjust hyperparameters as needed
python -m src.mia_regression.cli.run_train_target_baseline   --n_estimators 300
```

### 3) Train shadows & attack
```bash
python -m src.mia_regression.cli.run_train_shadows   --splits data/processed/processed_splits.npz   --out data/processed/attack_dataset_regression.npz   --n_shadows 7 --shadow_train_frac 0.5 --n_estimators 150 --seed 42

python -m src.mia_regression.cli.run_train_attack
```

### 4) Attack the target
```bash
python -m src.mia_regression.cli.run_attack_target   --target_model results/runs/<timestamp>/target_model.joblib   --attack_model results/runs/attack_model.joblib
```

### 5) Privacy–utility sweep (AUC vs RMSE)
```bash
# Multi-seed averaging for smoother results
python -m src.mia_regression.cli.run_sweep_rf_and_attack --seeds_per_config 3
python scripts/plot_privacy_utility.py
```

## Plots

- **Privacy–utility scatter**: `scripts/plot_privacy_utility.py`
    - This plot shows the trade-off between model utility (RMSE on the x-axis, lower is better) and privacy risk (attack AUC on the y-axis, lower is better). Each point represents a Random Forest configuration, with error bars showing variability across runs.
    - The hollow circles mark Pareto-optimal models, which are those that achieve the best balance between accuracy and privacy without being dominated by any other setting.
  - Random baseline at AUC=0.5

## Results

- Baseline RF (larger/deeper): **Attack AUC ≈ 0.64–0.66**, Test RMSE ≈ **3.9k**  
- Regularized RF (shallower/larger leaf): **Attack AUC ≈ 0.57–0.60**, Test RMSE change ≈ **1–2%**  
**Conclusion**: regularization reduces membership leakage with minor utility cost.

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

This work uses the [Health Insurance Charges Dataset](https://www.kaggle.com/datasets/nalisha/health-insurance-charges-dataset) from Kaggle [1] and draws conceptual guidance from the [University of Liverpool’s reading on Membership Inference Attacks and Differential Privacy](https://cgi.csc.liv.ac.uk/~acps/documents/readinggroup1.pdf) [3].  
Implementation and experimentation leveraged open-source tools including [scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) [2][4][5].

---

**References**

[1]: Nadeem, Aleesha. “Health Insurance Charges Dataset.” *Kaggle*, 22 Oct 2025. [https://www.kaggle.com/datasets/nalisha/health-insurance-charges-dataset](https://www.kaggle.com/datasets/nalisha/health-insurance-charges-dataset)  
[2]: Trusted-AI. *Adversarial Robustness Toolbox (ART) – Python Library for Machine Learning Security*. [https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)  
[3]: University of Liverpool. *Membership Inference Attacks and Differential Privacy*. [https://cgi.csc.liv.ac.uk/~acps/documents/readinggroup1.pdf](https://cgi.csc.liv.ac.uk/~acps/documents/readinggroup1.pdf)  
[4]: *Adversarial Robustness Toolbox Documentation – Membership Inference Module*. [https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html)  
[5]: Alencaster, Castillo Hernandez, Silva. “MIA-DP-Implementation.” [https://github.com/0x10jalencas/MIA-DP-Implementation](https://github.com/0x10jalencas/MIA-DP-Implementation)
