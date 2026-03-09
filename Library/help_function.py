import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

class ModelHelper:
    def __init__(self, X, y, n_splits=5):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.features_count = X.shape[1]

    @staticmethod
    def calculate_adjusted_r2(r2, n, p):
        """
        Statistický výpočet Adjusted R2.
        n: počet pozorování, p: počet parametrů (features)
        """
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def objective(self, trial):
        """
        Cílová funkce pro Optunu definovaná uvnitř třídy.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'n_jobs': -1,
            'random_state': 42
        }
        
        # Pro tuning používáme méně splitů kvůli rychlosti
        tscv_tune = TimeSeriesSplit(n_splits=3)
        cv_maes = []
        
        for t_idx, v_idx in tscv_tune.split(self.X):
            m = xgb.XGBRegressor(**params)
            m.fit(self.X.iloc[t_idx], self.y.iloc[t_idx])
            preds = m.predict(self.X.iloc[v_idx])
            cv_maes.append(mean_absolute_error(self.y.iloc[v_idx], preds))
            
        return np.mean(cv_maes)

    def run_tuning(self, n_trials=15):
        """
        Spustí Optuna studii.
        """
        print(f"Zahajuji tuning s {n_trials} pokusy...")
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params