# Scikit-Learn House Prices Prediction

**Goal:** The goal is simple: build a predictive model that accurately estimates home sale prices in Ames, Iowa by uncovering hidden patterns in the dataset. You'll need to analyze all 79 features—from obvious factors like square footage to subtle influences like neighborhood zoning—and engineer the best approach to beat the competition. The most accurate predictions win.

**Dataset:** Ames Housing dataset (train.csv). Contains 79 explanatory features describing residential homes and the target variable Sale Price.

Predict sale prices of Ames, Iowa homes with scikit-learn pipelines, feature engineering, and RMSLE-optimized models.

This repository contains all notebooks, source modules, and artefacts used for the course and Kaggle competition.

# 📂 Folder Structure

    ├── dataset/ # Raw CSVs only (train, test, sample_submission)
    ├── notebooks/ # EDA, modeling, and report notebooks (run on Colab)
    ├── src/ # Re-usable Python modules (loaders, transformers, models)
    ├── reports/ # Generated figures, tables, and submission files
    └── README.md
![ChatGPT Image May 18, 2025, 11_34_55 AM](https://github.com/user-attachments/assets/3785c7a6-f944-436c-aefd-4ab3ecbddf5c)


# Process & Notes

1. **Data Ingestion & EDA**  
   - Loaded Ames housing CSVs; inspected shape, missing values, and target distribution.  
   - Log-transformed `SalePrice` to stabilize variance.

2. **Preprocessing Pipeline**  
   - **Numerical**: median impute → `log1p` → StandardScaler  
   - **Ordinal**: mapped quality/condition columns (`ExterQual`, `KitchenQual`, etc.) to 0–4  
   - **Categorical**: constant-fill NA (“None”) → OneHotEncoder  
   - Fixed column-name mismatches (removed spaces) and aligned feature lists.

3. **Model Benchmarking**  
   - Evaluated 10 models via 5-fold CV (RMSLE): Ridge, Lasso, ElasticNet, RF, GBM, XGBoost, LightGBM, SVR, KNN, ExtraTrees.  
   - Discovered Ridge (α≈31.6) was best (CV RMSLE ≈0.132), with LightGBM close behind.

4. **Hyperparameter Tuning**  
   - Grid-searched α for Ridge; randomized-searched RF & GBM parameters.  
   - Avoided NaNs in RMSLE by wrapping linear models in `TransformedTargetRegressor` (log-space).

5. **Hold-out Validation**  
   - Verified on a 20% hold-out split: Ridge (0.1301) vs. 70/30 Ridge+LGBM ensemble (0.1273).

6. **Final Submission & Ensemble**  
   - Trained on full data; submitted Ridge baseline (0.11470 public RMSLE).  
   - Submitted 70/30 Ridge+LGBM blend (0.11016 public RMSLE), achieving top leaderboard rank.

**Challenges & Solutions**  
- Column mismatches (“MS SubClass” vs. “MSSubClass”): normalized names.  
- NaNs in categorical encoding: added constant-fill imputer.  
- Double-logging issue: passed raw target into RMSLE scorer.  


# 🙋‍♀️ Contact

Author: Nouhaila ELMALOULI – nouhailaelmalouli@gmail.com

Course: AI Programming Hackathon / IAAC

Instructor: @STASYA00
