# SAFE Flow

SAFE Flow (Stratified Analysis for Network Flow) is a Computer Engineering web project for screening network sessions as normal or attack. It uses a fitted scikit-learn pipeline trained on a public labeled dataset and exposes manual-entry and CSV analysis through Flask. Results support review and do not monitor live traffic or block attacks.

## Dataset and attribution

The included `data/cybersecurity_intrusion_data.csv` is Dinesh Naveen Kumar Samudrala’s *Cybersecurity Intrusion Detection Dataset* on [Kaggle](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset). Kaggle lists the dataset license as MIT. It contains 9,537 records, nine predictor fields, and the binary target `attack_detected` (0 = normal, 1 = attack). The full dataset has 5,273 normal and 4,264 attack records. Its Kaggle metadata tags the dataset as synthetic, so reported performance should not be generalized to real network traffic.

The predictor fields and data types are: numeric — `network_packet_size`, `login_attempts`, `session_duration`, `ip_reputation_score`, `failed_logins`, and `unusual_time_access`; categorical — `protocol_type`, `encryption_used`, and `browser_type`. The target `attack_detected` is binary numeric; `session_id` is an identifier excluded from training. The literal `None` value in `encryption_used` is preserved as a valid category. The dataset contains no missing predictor/target values and no duplicate rows.

## Training and evaluation

Install the dependencies, then run:

```powershell
python training/train_model.py
```

The script uses a stratified 80/20 train/test split with a fixed random seed. It fits preprocessing and four classifiers (Random Forest, Logistic Regression, Decision Tree, and SVM) on training data and compares them with stratified 3-fold cross-validation. Selection uses attack F2 (beta=2), which weights attack recall more heavily than precision. Candidates within 0.01 mean attack F2 of the cross-validation leader are considered near-tied; the highest mean attack F1, then accuracy, is selected. This keeps Random Forest as the paper's primary model when its recall-weighted score is effectively tied, and surfaces its global feature-importance ranking in the app. Its probability estimates are sigmoid-calibrated using training-only cross-validation; calibration scores come from folds, and one final forest is then refit on the full training split. Numeric values are median-imputed and standardized; categories are mode-imputed and one-hot encoded. All four uncalibrated candidates use the same holdout for like-for-like comparison; the calibrated primary model's deployed metrics are shown separately. The final model is evaluated with accuracy, precision, recall, F1, and a confusion matrix, then saved to `models/safe_flow_model.joblib`, with dataset details and metrics in `models/evaluation.json`.

The current training run selected **Random Forest**, the paper's primary model. Decision Tree led mean cross-validated attack F2 by less than one percentage point; Random Forest had the best attack F1 among candidates in that near-tie group. On the 1,908-row held-out split, the calibrated Random Forest achieved 88.26% accuracy, 97.87% precision, 75.38% attack recall, and 85.17% F1. Confusion matrix (actual rows, predicted columns): `[[1041, 14], [210, 643]]`. The About page shows the like-for-like comparison for all four uncalibrated candidates and the calibrated final model's metrics separately. These results describe this synthetic dataset and split; they are not production validation.

## Run locally

Requires Python 3.10 or newer. On Windows, double-click `Start-SAFE-Flow.bat`. It creates `.venv`, installs `requirements.txt` on first use, opens the browser, and starts the app. Keep the command window open while using SAFE Flow.

To start manually:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Web application features

- Home overview with the problem, intended users, and program fit.
- Manual session form with nine dataset fields and a trained-model prediction with probability and input-sensitivity summary.
- CSV preview, automatic/manual feature mapping, batch predictions, 25-row pagination, and full results export.
- About page with dataset source, class distribution, preprocessing and training methodology, four-model comparison, holdout metrics, and confusion matrix.
- Program relevance page with the problem statement, AI role, Computer Engineering alignment, and target users.

CSV uploads accept comma, semicolon, tab, or pipe delimiters and UTF-8, UTF-16, and common Windows encodings. Uploads are limited to 64 MB and 50,000 data rows. Unmapped rows are exported without a fabricated prediction. The trained pipeline imputes missing mapped values using statistics learned from the training split and ignores unknown categorical values safely.

## Project scope

The dataset contains session-level records. SAFE Flow does not capture packets, monitor live traffic, identify attack subtypes, or replace a security system. IP reputation values follow the Kaggle description: higher values indicate more suspicious activity.
