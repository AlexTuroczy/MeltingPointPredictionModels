import xgboost

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_absolute_error

import pandas as pd


def fit_xgboost(x_train, y_train, x_val, y_val, x_test):
    # Feature Selection inside the fold
    selector_model = xgboost.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=2025, verbosity=0)
    selector_model.fit(x_train, y_train)
    selector = SelectFromModel(selector_model, prefit=True, threshold="mean")
    selected_features = x_train.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} features.")

    # Apply feature selection
    x_train_new = x_train[selected_features]
    x_val_new = x_val[selected_features]
    x_test_new = x_test[selected_features]

    # Create XGBoost DMatrix
    dtrain = xgboost.DMatrix(x_train_new, label=y_train)
    dval = xgboost.DMatrix(x_val_new, label=y_val)
    dtest = xgboost.DMatrix(x_test_new)

    best_params = {
        'objective': 'reg:squarederror', 
        'max_depth': 3, 
        'learning_rate': 0.0168, 
        'min_child_weight': 5, 
        'subsample': 0.894, 
        'colsample_bytree': 0.590, 
        'gamma': 0.0113, 
        'lambda': 4.445, 
        'alpha': 0.129,
        'eval_metric': 'mae',
        # 'tree_method': 'gpu_hist'
    }

    # Train main model
    model = xgboost.train(
        params=best_params,
        dtrain=dtrain, 
        num_boost_round=15000, 
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=300,
        verbose_eval=1000
    )

    # Make predictions
    y_val_predict = model.predict(dval)
    y_test_predict = model.predict(dtest)

    fold_mae = mean_absolute_error(y_val, y_val_predict)
    print(f'MAE: {fold_mae}\n')

    return y_test_predict


if __name__ == '__main__':
    from data.utils import load_data, split_data, load_test_data, clean_for_xgboost

    data_df = load_data()

    data_df = clean_for_xgboost(data_df)

    x_train, y_train, x_val, y_val = split_data(data_df, rng=0)

    x_test = load_test_data()


    y_test_predict = fit_xgboost(x_train, y_train, x_val, y_val, x_test)


    y_test_predict = pd.Series(y_test_predict)

    y_test_predict.name = 'Tm'

    y_test_predict.to_csv('MeltingPointPredictionModels/data/submission.csv', index=False, header=True)


    print("Series saved to output.csv")