import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csvs():
    # read completed campaigns with their final delivery percentage
    completed = pd.read_csv(os.path.join(DATA_DIR, "completed_labeled.csv"),
                            parse_dates=["end_date"])
    # read campaign metadata (names, dates, planned impressions, zones, etc.)
    campaigns = pd.read_csv(os.path.join(DATA_DIR, "campaigns_cleaned.csv"),
                            parse_dates=["start_date", "end_date"])
    # read daily delivery data per campaign (how many impressions each campaign served each day)
    campaign_delivery = pd.read_csv(os.path.join(DATA_DIR, "campaign_delivery.csv"),
                                     parse_dates=["event_date"])
    # read fleet-level daily data (how many aircraft were active each day)
    fleet_daily = pd.read_csv(os.path.join(DATA_DIR, "fleet_daily.csv"), parse_dates=["event_date"])
    # read model 1's test predictions (needed to compute zpr)
    model1_test_results = pd.read_csv(os.path.join(DATA_DIR, "model1_test_results.csv"),
                                       parse_dates=["event_date"])
    return completed, campaigns, campaign_delivery, fleet_daily, model1_test_results


def build_model2_features(completed, campaigns, campaign_delivery, fleet_daily, model1_test_results):
    # get one row per campaign with its metadata
    camp_dates = (
        campaigns[["campaign_id", "start_date", "end_date", "planned_impression",
                    "campaign_duration_days", "zones_per_campaign",
                    "planned_daily_impressions", "revenue_type", "goal_type"]]
        .drop_duplicates(subset=["campaign_id"])
    )

    # join completed campaigns with their metadata
    comp = completed[["campaign_id", "delivery_pct"]].merge(camp_dates, on="campaign_id", how="inner")
    # only keep campaigns that had a planned impression target
    comp = comp[comp["planned_impression"] > 0].copy()

    # cap delivery at 300%, anything above is an extreme outlier that would distort training
    comp["target_raw"] = (comp["delivery_pct"] / 100.0).clip(upper=3.0)
    # log-transform the target to handle skewness (values range from 0.07% to 3344%)
    # log1p(x) = log(1+x), keeps 0 at 0 and compresses large values
    comp["target"] = np.log1p(comp["target_raw"])

    # calculate cumulative impressions served per campaign per day
    daily_delivery = (
        campaign_delivery
        .groupby(["campaign_id", "event_date"])["served_impressions"]
        .sum().reset_index().sort_values(["campaign_id", "event_date"])
    )
    # running total of impressions served
    daily_delivery["cumulative_served"] = daily_delivery.groupby("campaign_id")["served_impressions"].cumsum()

    # fleet data for computing impressions per flight
    fleet_flights = fleet_daily[["event_date", "active_tails"]].copy()

    # zpr (zone performance ratio) from model 1 test results
    # zpr = actual impressions / forecasted impressions, tells us if zones are over/underperforming
    zpr_by_zone_date = None
    if model1_test_results is not None and len(model1_test_results) > 0:
        zpr_df = model1_test_results.copy()
        # if forecast was 100 and actual was 80, zpr = 0.8 (zone underperformed)
        zpr_df["zpr"] = np.where(
            zpr_df["forecasted_imp_per_flight"] > 0,
            zpr_df["actual_imp_per_flight"] / zpr_df["forecasted_imp_per_flight"],
            0
        )
        zpr_by_zone_date = zpr_df[["zone_name", "event_date", "zpr"]]

    # take 9 snapshots per campaign at 10%, 20%, ..., 90% of its lifecycle
    # this turns 73 campaigns into ~655 training rows
    snapshot_fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []

    for _, camp in comp.iterrows():
        cid = camp["campaign_id"]
        start = camp["start_date"]
        end = camp["end_date"]
        duration = camp["campaign_duration_days"]
        planned = camp["planned_impression"]
        # the target (final delivery %) stays the same for all snapshots of the same campaign
        # because we know the outcome, we are training the model to predict it from partial data
        target = camp["target"]

        # get this campaign's daily delivery data
        camp_daily = daily_delivery[daily_delivery["campaign_id"] == cid]
        if camp_daily.empty:
            continue

        # create a snapshot at each fraction of the campaign's lifecycle
        for frac in snapshot_fractions:
            # how many days into the campaign this snapshot is
            days_elapsed = int(duration * frac)
            if days_elapsed < 1:
                continue
            # the calendar date of this snapshot
            snapshot_date = start + pd.Timedelta(days=days_elapsed)
            # days remaining in the campaign from this snapshot
            remaining_days = max(duration - days_elapsed, 1)

            # how many impressions had been delivered by this snapshot date
            delivered_data = camp_daily[camp_daily["event_date"] <= snapshot_date]
            actual_delivered = delivered_data["cumulative_served"].iloc[-1] if len(delivered_data) > 0 else 0

            # how many impressions should have been delivered by now (assuming linear pacing)
            expected_by_now = planned * frac
            if expected_by_now <= 0:
                continue

            # pacing_ratio: actual/expected, below 1 means behind schedule, above 1 means ahead
            pacing_ratio = actual_delivered / expected_by_now
            # fraction_elapsed: what % of the campaign's duration has passed (0.1 to 0.9)
            fraction_elapsed = frac

            # how many zones this campaign runs in
            zones = camp["zones_per_campaign"]
            # planned daily impressions (total planned / duration)
            pdi = camp["planned_daily_impressions"]

            # impressions per flight for this campaign up to the snapshot date
            # normalize by fleet size to remove bias from varying flight counts
            camp_daily_to_snap = camp_daily[camp_daily["event_date"] <= snapshot_date].merge(
                fleet_flights, on="event_date", how="left"
            )
            camp_daily_to_snap = camp_daily_to_snap.dropna(subset=["active_tails"])
            if len(camp_daily_to_snap) > 0 and camp_daily_to_snap["active_tails"].sum() > 0:
                imp_per_flight = (
                    camp_daily_to_snap["served_impressions"].sum() /
                    camp_daily_to_snap["active_tails"].sum()
                )
            else:
                imp_per_flight = 0

            # needed_vs_runway: how much faster the campaign needs to go to finish on time
            # if needed_vs_runway = 2, the campaign needs to deliver 2x its current daily rate
            remaining_impressions = max(planned - actual_delivered, 0)
            needed_daily = remaining_impressions / remaining_days
            current_daily = actual_delivered / max(days_elapsed, 1)
            needed_vs_runway = needed_daily / max(current_daily, 1.0)

            # avg_zone_zpr: average zone performance ratio across this campaign's zones
            avg_zpr = np.nan
            if zpr_by_zone_date is not None:
                camp_zones = campaign_delivery[
                    campaign_delivery["campaign_id"] == cid
                ]["zone_name"].unique()

                zone_zpr = zpr_by_zone_date[
                    (zpr_by_zone_date["zone_name"].isin(camp_zones)) &
                    (zpr_by_zone_date["event_date"] <= snapshot_date)
                ]
                if len(zone_zpr) > 0:
                    avg_zpr = zone_zpr["zpr"].mean()

            # goal_type: 1 if "consideration" (higher priority), 0 if "awareness"
            goal_encoded = 1 if camp["goal_type"] == "consideration" else 0
            # revenue_type: paid=2 (highest priority), partner=1, house=0
            rev_map = {"paid": 2, "partner": 1, "house": 0}
            rev_encoded = rev_map.get(camp["revenue_type"], 0)

            rows.append({
                "campaign_id": cid,
                "snapshot_frac": frac,
                "pacing_ratio": pacing_ratio,
                "fraction_elapsed": fraction_elapsed,
                "zones_per_campaign": zones,
                "planned_daily_impressions": pdi,
                "impressions_per_flight": imp_per_flight,
                "needed_vs_runway": needed_vs_runway,
                "avg_zone_zpr": avg_zpr,
                "goal_type": goal_encoded,
                "revenue_type": rev_encoded,
                "target": target,
                "target_raw": (camp["delivery_pct"] / 100.0),
            })

    df = pd.DataFrame(rows)
    return df


def train_model2(df):
    # the 9 features model 2 uses to predict final delivery percentage
    feature_cols = [
        "pacing_ratio", "fraction_elapsed", "zones_per_campaign",
        "planned_daily_impressions", "impressions_per_flight",
        "needed_vs_runway", "avg_zone_zpr", "goal_type", "revenue_type"
    ]

    # fill missing zpr values with the median, some campaigns don't have zpr data
    df["avg_zone_zpr"] = df["avg_zone_zpr"].fillna(df["avg_zone_zpr"].median())
    # if still nan (all values were nan), default to 1.0 (neutral performance)
    df["avg_zone_zpr"] = df["avg_zone_zpr"].fillna(1.0)

    # cap extreme values to reduce outlier influence
    df["pacing_ratio"] = df["pacing_ratio"].clip(upper=10.0)
    df["needed_vs_runway"] = df["needed_vs_runway"].clip(upper=20.0)

    X = df[feature_cols]
    y = df["target"]

    # groupkfold cross-validation: all snapshots from the same campaign stay in the same fold
    # prevents data leakage (if campaign 123's 10% snapshot is in train,
    # its 50% snapshot can't be in test because they share the same label)
    unique_campaigns = df["campaign_id"].unique()
    n_folds = min(5, len(unique_campaigns))

    # randomly assign each campaign to a fold
    np.random.seed(42)
    fold_assignments = np.random.permutation(len(unique_campaigns)) % n_folds
    camp_to_fold = dict(zip(unique_campaigns, fold_assignments))
    df["fold"] = df["campaign_id"].map(camp_to_fold)

    # array to store out-of-fold predictions for every row
    all_preds = np.full(len(df), np.nan)
    fold_metrics = []

    # train one model per fold, predict on the held-out fold
    for fold in range(n_folds):
        train_mask = df["fold"] != fold
        test_mask = df["fold"] == fold

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.5,
            reg_lambda=2.0,
            objective="reg:absoluteerror",
            early_stopping_rounds=20,
            random_state=42
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        all_preds[test_mask.values] = preds

        # mae for this fold in real space (not log space)
        y_actual_real = np.expm1(y_test)
        y_pred_real = np.expm1(preds)
        fold_mae = mean_absolute_error(y_actual_real, y_pred_real)
        fold_metrics.append(fold_mae)

    # overall metrics, convert from log space back to real percentages
    y_actual_all = np.expm1(y.values)
    y_pred_all = np.expm1(all_preds)
    y_pred_all = np.clip(y_pred_all, 0, 5.0)

    mae = mean_absolute_error(y_actual_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_actual_all, y_pred_all))
    r2 = r2_score(y_actual_all, y_pred_all)

    print(f"\n  Model 2 Results ({n_folds}-fold GroupKFold CV):")
    print(f"    MAE:   {mae:.4f} (in delivery ratio, 0.10 = 10% off)")
    print(f"    RMSE:  {rmse:.4f}")
    print(f"    R²:    {r2:.4f}")
    print(f"    Per-fold MAE: {[f'{m:.4f}' for m in fold_metrics]}")

    # train final model on ALL data (cv already told us how well it generalizes)
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="reg:absoluteerror",
        random_state=42
    )
    final_model.fit(X, y)

    # feature importance
    importance = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Feature Importances:")
    for feat, imp in importance.items():
        print(f"    {feat}: {imp:.4f}")

    # under vs over delivering analysis
    df["predicted_raw"] = y_pred_all
    df["actual_raw"] = y_actual_all

    under = df[df["target_raw"] < 1.0]
    over = df[df["target_raw"] >= 1.0]

    if len(under) > 0:
        under_mae = mean_absolute_error(under["actual_raw"].clip(upper=3.0),
                                         under["predicted_raw"].clip(upper=3.0))
        correct_under = (under["predicted_raw"] < 1.0).sum()
        print(f"\n  Underdelivering campaigns ({len(under)} snapshots):")
        print(f"    MAE: {under_mae:.4f}")
        print(f"    Correctly predicted <100%: {correct_under}/{len(under)} "
              f"({correct_under/len(under)*100:.0f}%)")

    if len(over) > 0:
        over_mae = mean_absolute_error(over["actual_raw"].clip(upper=3.0),
                                        over["predicted_raw"].clip(upper=3.0))
        print(f"\n  Overdelivering campaigns ({len(over)} snapshots):")
        print(f"    MAE: {over_mae:.4f}")

    # sample predictions
    print(f"\n  Sample UNDER-delivering predictions:")
    under_sample = under.drop_duplicates("campaign_id").head(10)
    for _, row in under_sample.iterrows():
        snapshots = df[df["campaign_id"] == row["campaign_id"]]
        for _, s in snapshots.iterrows():
            print(f"    Campaign {int(s['campaign_id'])} @ {int(s['snapshot_frac']*100)}%: "
                  f"actual={s['actual_raw']*100:.0f}% predicted={s['predicted_raw']*100:.0f}%")

    print(f"\n  Sample OVER-delivering predictions:")
    over_sample = over.drop_duplicates("campaign_id").head(5)
    for _, row in over_sample.iterrows():
        snap = df[(df["campaign_id"] == row["campaign_id"]) & (df["snapshot_frac"] == 0.5)]
        for _, s in snap.iterrows():
            print(f"    Campaign {int(s['campaign_id'])} @ 50%: "
                  f"actual={min(s['actual_raw'],3.0)*100:.0f}% predicted={s['predicted_raw']*100:.0f}%")

    df.drop(columns=["predicted_raw", "actual_raw", "fold"], inplace=True, errors="ignore")

    return final_model, feature_cols


if __name__ == "__main__":
    completed, campaigns, campaign_delivery, fleet_daily, model1_test_results = load_csvs()

    print("MODEL 2: CAMPAIGN COMPLETION / RISK\n")
    m2_df = build_model2_features(completed, campaigns, campaign_delivery, fleet_daily, model1_test_results)
    model2, m2_features = train_model2(m2_df)

    # save the trained model to a pkl file (pickle, serializes the python object to disk
    # so it can be loaded later without retraining)
    joblib.dump(model2, os.path.join(DATA_DIR, "model2.pkl"))
    # save the feature list so pipeline.py knows what columns to prepare
    json.dump(m2_features, open(os.path.join(DATA_DIR, "model2_features.json"), "w"))

    print("\nSaved: model2.pkl, model2_features.json")
