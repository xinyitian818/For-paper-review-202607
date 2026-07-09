# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

from scipy.interpolate import UnivariateSpline
import shap
import joblib
import warnings
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from pycirclize import Circos


warnings.filterwarnings('ignore')


INPUT_FILE = "datasets from EP.csv"
OUTPUT_DIR = 'ml results'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)


FEATURES = [
    'latitude', 'delta_shgc', 'ave_shgc',
    't_min', 't_max', 't_avg',
    'ghi', 'avg_rh', 'avg_ws'
]
TARGETS = ['c_total_gj', 'h_total_gj']

FEATURE_LABELS = {
    'latitude': 'Latitude',
    'delta_shgc': r'$\Delta$SHGC',
    'ave_shgc': r'$SHGC_{\mathrm{ave}}$',
    't_min': r'$T_{\mathrm{min}}$',
    't_max': r'$T_{\mathrm{max}}$',
    't_avg': r'$T_{\mathrm{avg}}$',
    'ghi': 'GHI Radiation',
    'avg_rh': 'Avg_RH',
    'avg_ws': 'Avg_WindSpeed'
}


df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()
df = df.dropna(subset=FEATURES + TARGETS + ['city'])

city_lat_map = df.groupby('city')['latitude'].first().reset_index()
city_lat_map['lat_bin'] = pd.qcut(city_lat_map['latitude'], q=5, labels=False)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_avg_r2 = -np.inf
best_fold_idx = -1
best_data = {}

header = f"{'Fold':<5} | {'COOL R^2':<10} | {'HEAT R^2':<10} | {'AVG R^2':<10} | {'COOL MAE (GJ)':<10} | {'HEAT MAE (GJ)':<10}"
print("-" * len(header));
print(header);
print("-" * len(header))

fold_idx = 1
for tr_idx, te_idx in skf.split(city_lat_map['city'], city_lat_map['lat_bin']):
    train_cities, test_cities = city_lat_map.iloc[tr_idx]['city'].values, city_lat_map.iloc[te_idx]['city'].values
    df_tr, df_te = df[df['city'].isin(train_cities)], df[df['city'].isin(test_cities)]
    X_tr, y_tr = df_tr[FEATURES].values, df_tr[TARGETS].values
    X_te, y_te = df_te[FEATURES].values, df_te[TARGETS].values


    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=1500, learning_rate=0.015, max_depth=6,
        min_child_weight=25, gamma=0.4, reg_lambda=25.0, reg_alpha=6.0,
        subsample=0.75, colsample_bytree=0.75, random_state=42, n_jobs=-1
    ))

    model.fit(X_tr, y_tr)
    p_te = model.predict(X_te)

    r2_c, r2_h = r2_score(y_te[:, 0], p_te[:, 0]), r2_score(y_te[:, 1], p_te[:, 1])
    avg_r2 = (r2_c + r2_h) / 2
    print(
        f"#{fold_idx:<4} | {r2_c:<10.5f} | {r2_h:<10.5f} | {avg_r2:<10.5f} | {mean_absolute_error(y_te[:, 0], p_te[:, 0]):<10.2f} | {mean_absolute_error(y_te[:, 1], p_te[:, 1]):<10.2f}")

    if avg_r2 > best_avg_r2:
        best_avg_r2 = avg_r2
        best_fold_idx = fold_idx
        best_data = {
            'model': model,
            'X_te': X_te, 'y_te': y_te, 'p_te': p_te,
            'X_te_df': pd.DataFrame(X_te, columns=FEATURES)
        }
    fold_idx += 1

joblib.dump(best_data['model'], os.path.join(OUTPUT_DIR, 'XGB_ML_model.pkl'))
