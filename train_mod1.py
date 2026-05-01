import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# get the directory where this script lives — all csvs and outputs go here
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csvs():
    # read zone-level daily impression data with dates parsed
    zone_daily = pd.read_csv(os.path.join(DATA_DIR, "zone_daily.csv"), parse_dates=["event_date"])
    # read fleet-level daily data (how many aircraft were active each day)
    fleet_daily = pd.read_csv(os.path.join(DATA_DIR, "fleet_daily.csv"), parse_dates=["event_date"])
    # read campaign data (names, dates, planned impressions, zones, etc.)
    campaigns = pd.read_csv(os.path.join(DATA_DIR, "campaigns_cleaned.csv"),
                            parse_dates=["start_date", "end_date"])
    # read daily delivery data per campaign (how many impressions each campaign served each day)
    campaign_delivery = pd.read_csv(os.path.join(DATA_DIR, "campaign_delivery.csv"),
                                     parse_dates=["event_date"])
    return zone_daily, fleet_daily, campaigns, campaign_delivery


def build_model1_features(zone_daily, fleet_daily, campaigns, campaign_delivery):
    # grab just the date and aircraft count columns from fleet data
    fleet_flights = fleet_daily[["event_date", "active_tails"]].copy()
    # join zone daily data with fleet data so each row has the number of active aircraft that day
    df = zone_daily.merge(fleet_flights, on="event_date", how="left")

    # drop rows where we don't know how many aircraft were flying
    df = df.dropna(subset=["active_tails"])
    # drop rows where zero aircraft
    df = df[df["active_tails"] > 0].copy()

    # normalize impressions by dividing by number of flights — our target variable
    df["imp_per_flight"] = df["total_impression_cnt"] / df["active_tails"]

    # count how many unique days each zone has data for
    zone_day_counts = df.groupby("zone_name")["event_date"].nunique()
    # only keep zones with at least 30 days — need enough history to have rolling features
    valid_zones = zone_day_counts[zone_day_counts >= 30].index
    before = df["zone_name"].nunique()
    # filter the dataframe to only valid zones
    df = df[df["zone_name"].isin(valid_zones)].copy()
    print(f"  Filtered zones: {before} → {df['zone_name'].nunique()} (kept zones with >= 30 days)")

    # some zones have multiple rows for the same date — aggregate them
    num_cols = df.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in df.columns if c not in num_cols and c not in ["zone_name", "event_date"]]
    agg_dict = {c: "sum" for c in num_cols}
    agg_dict.update({c: "first" for c in other_cols})
    df = df.groupby(["zone_name", "event_date"], as_index=False).agg(agg_dict)

    # recalculate impressions per flight after summing
    df["imp_per_flight"] = df["total_impression_cnt"] / df["active_tails"]

    # sort by zone then date so shift/rolling operations work correctly within each zone
    df = df.sort_values(["zone_name", "event_date"]).reset_index(drop=True)

    # how many calendar days since this zone's previous data point
    df["days_since_last"] = df.groupby("zone_name")["event_date"].diff().dt.days.fillna(1)

    # impression features — group by zone so shift/rolling only looks within the same zone
    grouped = df.groupby("zone_name")["imp_per_flight"]
    # lag_1d: this zone's impressions per flight from the previous observation
    df["lag_1d"] = grouped.shift(1)
    # lag_7d: this zone's impressions per flight from 7 observations ago
    df["lag_7d"] = grouped.shift(7)
    # rolling_mean_7d: average of the last 7 observations
    df["rolling_mean_7d"] = grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    # rolling_mean_30d: average of the last 30 observations
    df["rolling_mean_30d"] = grouped.transform(lambda x: x.shift(1).rolling(30, min_periods=3).mean())
    # rolling_std_7d: standard deviation of last 7 observations — how volatile the zone is
    df["rolling_std_7d"] = grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    df["rolling_std_7d"] = df["rolling_std_7d"].fillna(0)

    # zone identity: categorize zone by its name
    df["zone_category"] = (
        df["zone_name"]
        .str.lower()
        .str.extract(r"(home|tv|movies|movie|fb|boarding|seatback|map|music|games|shopping|welcome)", expand=False)
        .fillna("other")
    )
    df["zone_category"] = df["zone_category"].replace("movie", "movies")

    # campaign load features per zone per day
    camp_zones = campaigns[["campaign_id", "zone_name", "start_date", "end_date",
                            "planned_impression", "campaign_duration_days",
                            "revenue_type"]].drop_duplicates(subset=["campaign_id", "zone_name"])

    cum_served = (
        campaign_delivery
        .groupby(["campaign_id", "event_date"])["served_impressions"]
        .sum().reset_index().sort_values(["campaign_id", "event_date"])
    )
    cum_served["cumulative_served"] = cum_served.groupby("campaign_id")["served_impressions"].cumsum()

    unique_dates = df["event_date"].unique()
    demand_records = []

    print(f"  Computing campaign load features for {len(unique_dates)} dates...")
    for i, date in enumerate(unique_dates):
        date_ts = pd.Timestamp(date)

        active = camp_zones[
            (camp_zones["start_date"] <= date_ts) &
            (camp_zones["end_date"] >= date_ts)
        ].copy()

        if active.empty:
            continue

        active["remaining_days"] = (active["end_date"] - date_ts).dt.days.clip(lower=1)

        served_by_date = cum_served[cum_served["event_date"] <= date_ts]
        latest_served = served_by_date.groupby("campaign_id")["cumulative_served"].last().reset_index()

        active = active.merge(latest_served, on="campaign_id", how="left")
        active["cumulative_served"] = active["cumulative_served"].fillna(0)

        active["daily_demand"] = (
            (active["planned_impression"] - active["cumulative_served"]) / active["remaining_days"]
        ).clip(lower=0)

        zone_demand = active.groupby("zone_name").agg(
            total_planned_daily_demand=("daily_demand", "sum"),
            paid_count=("revenue_type", lambda x: (x == "paid").sum()),
            total_count=("revenue_type", "count")
        ).reset_index()

        zone_demand["paid_ratio"] = zone_demand["paid_count"] / zone_demand["total_count"]
        zone_demand["event_date"] = date_ts
        demand_records.append(zone_demand[["zone_name", "event_date",
                                           "total_planned_daily_demand", "paid_ratio"]])

    if demand_records:
        demand_df = pd.concat(demand_records, ignore_index=True)
        df = df.merge(demand_df, on=["zone_name", "event_date"], how="left")

    df["total_planned_daily_demand"] = df.get("total_planned_daily_demand", pd.Series(0)).fillna(0)
    df["paid_ratio"] = df.get("paid_ratio", pd.Series(0)).fillna(0)

    # time features
    df["day_of_week"] = df["event_date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["event_date"].dt.month

    # drop rows where lag features aren't available yet
    df = df.dropna(subset=["lag_1d", "lag_7d", "rolling_mean_7d", "rolling_mean_30d"]).copy()

    # one-hot encode zone_category into separate binary columns
    df = pd.get_dummies(df, columns=["zone_category"], prefix="cat")

    # set the target variable
    df["target"] = df["imp_per_flight"]

    return df


def train_model1(df):
    # select feature columns: everything except identifiers, raw values, and the target
    feature_cols = [c for c in df.columns if c not in [
        "zone_id", "zone_name", "icao_airline", "event_date",
        "total_impression_cnt", "active_tails", "imp_per_flight", "target"
    ]]

    X = df[feature_cols]
    y = df["target"]

    # time-based split: train on the first 80% of dates, test on the last 20%
    dates_sorted = df["event_date"].sort_values().unique()
    split_idx = int(len(dates_sorted) * 0.8)
    split_date = dates_sorted[split_idx]

    train_mask = df["event_date"] < split_date
    test_mask = df["event_date"] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train: {len(X_train)} rows (before {split_date.date()})")
    print(f"  Test:  {len(X_test)} rows (from {split_date.date()})")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    # --- metrics ---

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # mape only on zones with actual > 1 imp/flight (zones near zero inflate mape to thousands)
    meaningful = y_test > 1.0
    if meaningful.sum() > 0:
        mape_filtered = np.mean(np.abs(
            (y_test[meaningful] - y_pred[meaningful]) / y_test[meaningful]
        )) * 100
    else:
        mape_filtered = float("nan")

    # median absolute percentage error (not pulled by outliers like mape is)
    if meaningful.sum() > 0:
        mdape = np.median(np.abs(
            (y_test[meaningful] - y_pred[meaningful]) / y_test[meaningful]
        )) * 100
    else:
        mdape = float("nan")

    print(f"\n  Model 1 Results:")
    print(f"    MAE:   {mae:.2f} imp/flight")
    print(f"    RMSE:  {rmse:.2f}")
    print(f"    R²:    {r2:.4f}")
    print(f"    MAPE:  {mape_filtered:.1f}%  (only zones > 1 imp/flight)")
    print(f"    MdAPE: {mdape:.1f}%  (median, only zones > 1 imp/flight)")

    # feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Feature Importances:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat}: {imp:.4f}")

    # error breakdown by zone size
    test_df = df[test_mask].copy()
    test_df["pred"] = y_pred

    zone_medians = test_df.groupby("zone_name")["target"].median()
    small_zones = zone_medians[zone_medians < 1].index
    medium_zones = zone_medians[(zone_medians >= 1) & (zone_medians < 50)].index
    large_zones = zone_medians[zone_medians >= 50].index

    print(f"\n  Error by Zone Size:")
    for label, zset in [("Small (<1 imp/flight)", small_zones),
                         ("Medium (1-50)", medium_zones),
                         ("Large (>50)", large_zones)]:
        subset = test_df[test_df["zone_name"].isin(zset)]
        if len(subset) > 0:
            m = mean_absolute_error(subset["target"], subset["pred"])
            print(f"    {label}: MAE={m:.2f} ({len(zset)} zones, {len(subset)} rows)")

    # save test predictions
    test_results = df[test_mask][["zone_name", "event_date", "total_impression_cnt"]].copy()
    test_results["forecasted_imp_per_flight"] = y_pred
    test_results["actual_imp_per_flight"] = y_test.values

    return model, feature_cols, test_results


if __name__ == "__main__":
    zone_daily, fleet_daily, campaigns, campaign_delivery = load_csvs()

    print("MODEL 1: ZONE CAPACITY FORECASTING\n")
    m1_df = build_model1_features(zone_daily, fleet_daily, campaigns, campaign_delivery)
    model1, m1_features, m1_test_results = train_model1(m1_df)

    # save the trained model to a pkl file (pickle — serializes the python object to disk
    # so it can be loaded later without retraining)
    joblib.dump(model1, os.path.join(DATA_DIR, "model1.pkl"))
    # save the feature list so pipeline.py knows what columns to prepare
    json.dump(m1_features, open(os.path.join(DATA_DIR, "model1_features.json"), "w"))
    # save test predictions — model 2 uses these to compute zpr
    m1_test_results.to_csv(os.path.join(DATA_DIR, "model1_test_results.csv"), index=False)

    print("\nSaved: model1.pkl, model1_features.json, model1_test_results.csv")
