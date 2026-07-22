import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.model_selection import TimeSeriesSplit


def load_block_data(file_path):

    df = pd.read_excel(file_path, header=None)

    rain = df.iloc[2:40]
    tmax = df.iloc[42:80]
    tmin = df.iloc[82:120]
    rh = df.iloc[122:]

    cols = [
        "YEAR", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
    ]

    blocks = []

    for block, name in zip(
        [rain, tmax, tmin, rh],
        ["RAINFALL", "TMAX", "TMIN", "RH"]
    ):

        block.columns = cols

        block["YEAR"] = pd.to_numeric(
            block["YEAR"],
            errors="coerce"
        )

        block = block.dropna(subset=["YEAR"])

        for c in cols[1:]:
            block[c] = pd.to_numeric(
                block[c],
                errors="coerce"
            )

        block = block.melt(
            id_vars="YEAR",
            var_name="MONTH",
            value_name=name
        )

        blocks.append(block)

    df_final = blocks[0]

    for b in blocks[1:]:
        df_final = df_final.merge(
            b,
            on=["YEAR", "MONTH"]
        )

    return df_final.dropna()


def prepare_annual_features(df_month, yield_df):

    # =========================
    # YEARLY CLIMATE FEATURES
    # =========================

    annual = df_month.groupby("YEAR").agg({
        "TMEAN": "mean",
        "RAINFALL": "sum",
        "VPD": "mean",
        "RH": "mean"
    }).reset_index()

    annual.columns = [
        "YEAR",
        "TMEAN",
        "RAINFALL_SUM",
        "VPD",
        "RH"
    ]

    # =========================
    # GROWING SEASON RAINFALL
    # =========================

    season_rain = (
        df_month[
            df_month["MONTH"].isin(
                ["DEC", "JAN", "FEB", "MAR", "APR", "MAY"]
            )
        ]
        .groupby("YEAR")["RAINFALL"]
        .sum()
        .reset_index()
    )

    season_rain.columns = [
        "YEAR",
        "SEASON_RAIN"
    ]

    # =========================
    # VPD STRESS
    # =========================

    vpd_stress = df_month.groupby("YEAR")["VPD"].apply(
        lambda x: (x > 1.2).sum()
    ).reset_index()

    vpd_stress.columns = [
        "YEAR",
        "VPD_STRESS_MONTHS"
    ]

    # =========================
    # FAW HIGH MONTHS
    # =========================

    faw_high = df_month.groupby("YEAR")["FAW"].apply(
        lambda x: (x == "High").sum()
    ).reset_index()

    faw_high.columns = [
        "YEAR",
        "FAW_HIGH_MONTHS"
    ]

    # =========================
    # SEASONAL RAIN
    # =========================

    seasonal_rain = (
        df_month[
            df_month["MONTH"].isin(
                ["NOV", "DEC", "JAN", "FEB", "MAR", "APR"]
            )
        ]
        .groupby("YEAR")["RAINFALL"]
        .sum()
        .reset_index()
    )

    seasonal_rain.columns = [
        "YEAR",
        "SEASONAL_RAIN"
    ]

    # =========================
    # FAW SCORE
    # =========================

    faw_yearly = df_month.groupby("YEAR")["FAW"].apply(
        lambda x: x.map({
            "Low": 0,
            "Medium": 1,
            "High": 2
        }).max()
    ).reset_index()

    faw_yearly.columns = [
        "YEAR",
        "FAW_SCORE"
    ]

    # =========================
    # MERGE FEATURES
    # =========================

    model_df = pd.merge(
        annual,
        seasonal_rain,
        on="YEAR",
        how="left"
    )

    model_df = pd.merge(
        model_df,
        season_rain,
        on="YEAR",
        how="left"
    )

    model_df = pd.merge(
        model_df,
        faw_yearly,
        on="YEAR",
        how="left"
    )

    model_df = pd.merge(
        model_df,
        yield_df,
        on="YEAR",
        how="inner"
    )

    # =========================
    # CLEAN YIELD COLUMN
    # =========================

    model_df["YIELD"] = (
        model_df["YIELD"]
        .astype(str)
        .str.replace("kg/ha", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    model_df["YIELD"] = pd.to_numeric(
        model_df["YIELD"],
        errors="coerce"
    )

    model_df = model_df.dropna(
        subset=["YIELD"]
    )

    # =========================
    # CLEAN FAW SCORE
    # =========================

    model_df["FAW_SCORE"] = (
        model_df["FAW_SCORE"]
        .fillna(0)
    )

    # =========================
    # RAIN EFFICIENCY
    # =========================

    model_df["RAIN_EFFICIENCY"] = (
        model_df["YIELD"] /
        model_df["RAINFALL_SUM"]
    )

    # =========================
    # RAINFALL ANOMALY
    # =========================

    model_df["RAIN_ANOMALY"] = (
        model_df["RAINFALL_SUM"]
        -
        model_df["RAINFALL_SUM"].mean()
    )

    model_df["SEASONAL_RAIN"] = (
        model_df["SEASONAL_RAIN"]
        .fillna(0)
    )

    # =========================
    # MERGE EXTRA FEATURES
    # =========================

    model_df = pd.merge(
        model_df,
        vpd_stress,
        on="YEAR",
        how="left"
    )

    model_df = pd.merge(
        model_df,
        faw_high,
        on="YEAR",
        how="left"
    )
    # =========================
    # CREATE LAG FEATURES
    # =========================

    # Lag 1
    model_df["YIELD_LAG1"] = model_df["YIELD"].shift(1)
    model_df["RAIN_LAG1"] = model_df["RAINFALL_SUM"].shift(1)
    model_df["VPD_LAG1"] = model_df["VPD"].shift(1)
    model_df["FAW_LAG1"] = model_df["FAW_SCORE"].shift(1)

    # Lag 2
    model_df["YIELD_LAG2"] = model_df["YIELD"].shift(2)
    model_df["RAIN_LAG2"] = model_df["RAINFALL_SUM"].shift(2)
    model_df["VPD_LAG2"] = model_df["VPD"].shift(2)

    model_df["YIELD_LAG3"] = model_df["YIELD"].shift(3)
    model_df["RAIN_LAG3"] = model_df["RAINFALL_SUM"].shift(3)
    model_df["VPD_LAG3"] = model_df["VPD"].shift(3)

    # =========================
    # LAG FEATURES
    # =========================

    model_df = model_df.sort_values(
        "YEAR"
    )


    model_df["YIELD_LAG1"] = (
        model_df["YIELD"]
        .shift(1)
    )


    model_df["RAIN_LAG1"] = (
        model_df["RAINFALL_SUM"]
        .shift(1)
    )


    model_df["VPD_LAG1"] = (
        model_df["VPD"]
        .shift(1)
    )


    # =========================
    # TIME TREND FEATURE
    # =========================

    model_df["YEAR_INDEX"] = (
        model_df["YEAR"]
        - model_df["YEAR"].min()
    )

    return model_df
def get_region_features(region):

    if region == "Mbeya":

        return [
            "TMEAN",
            "RAINFALL_SUM",
            "SEASON_RAIN",
            "SEASONAL_RAIN",
            "RAIN_ANOMALY",
            "VPD",
            "FAW_SCORE",
            "YIELD_LAG1",
            "RAIN_LAG1",
            "VPD_LAG1"
        ]

    elif region == "Kongwa":

        return None

    elif region == "Zanzibar":

        return None

    return None
def get_models():

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    }
def prepare_training_data(model_df, features):

    X = model_df[features]

    y = model_df["YIELD"]

    return X, y
def train_models(models, X, y):

    tscv = TimeSeriesSplit(n_splits=3)

    best_model = None
    best_name = ""
    best_r2 = -999
    best_rmse = None
    best_mae = None

    for model_name, model in models.items():

        r2_scores = []
        rmse_scores = []
        mae_scores = []

        for train_index, test_index in tscv.split(X):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(
                np.sqrt(
                    mean_squared_error(y_test, y_pred)
                )
            )
            mae_scores.append(
                mean_absolute_error(y_test, y_pred)
            )

        avg_r2 = np.mean(r2_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_rmse = avg_rmse
            best_mae = avg_mae
            best_name = model_name
            best_model = model

    return (
        best_model,
        best_name,
        best_r2,
        best_rmse,
        best_mae
    )
def predict_yield(
    best_model,
    X,
    y,
    latest_data
):

    best_model.fit(X, y)

    prediction = best_model.predict(latest_data)[0]

    return prediction