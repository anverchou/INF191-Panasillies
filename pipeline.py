import os
import json
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import linprog

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
# LOAD DATA + SAVED MODELS
# ──────────────────────────────────────────────────────────────

def load_data_and_models():
    campaigns = pd.read_csv(os.path.join(DATA_DIR, "campaigns_cleaned.csv"),
                            parse_dates=["start_date", "end_date"])
    zone_daily = pd.read_csv(os.path.join(DATA_DIR, "zone_daily.csv"), parse_dates=["event_date"])
    fleet_daily = pd.read_csv(os.path.join(DATA_DIR, "fleet_daily.csv"), parse_dates=["event_date"])
    campaign_delivery = pd.read_csv(os.path.join(DATA_DIR, "campaign_delivery.csv"),
                                     parse_dates=["event_date"])
    fleet_metrics = pd.read_csv(os.path.join(DATA_DIR, "fleet_metrics.csv")).iloc[0].to_dict()

    model1 = joblib.load(os.path.join(DATA_DIR, "model1.pkl"))
    model1_features = json.load(open(os.path.join(DATA_DIR, "model1_features.json")))
    model2 = joblib.load(os.path.join(DATA_DIR, "model2.pkl"))

    return campaigns, zone_daily, fleet_daily, campaign_delivery, fleet_metrics, model1, model1_features, model2


# ══════════════════════════════════════════════════════════════
# STEP 3: ZONE CAPACITY FORECASTING (runs before Step 2)
# Uses saved Model 1 to predict tomorrow's zone capacities
# ══════════════════════════════════════════════════════════════

def step3_forecast_zone_capacity(model1, model1_features, zone_daily, fleet_daily,
                                  campaigns, campaign_delivery, target_date=None):
    if target_date is None:
        target_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

    fleet_flights = fleet_daily[["event_date", "active_tails"]].copy()

    # merge zone_daily with fleet to get imp_per_flight
    df = zone_daily.merge(fleet_flights, on="event_date", how="left")
    df = df.dropna(subset=["active_tails"])
    df = df[df["active_tails"] > 0].copy()
    df["imp_per_flight"] = df["total_impression_cnt"] / df["active_tails"]
    df = df.sort_values(["zone_name", "event_date"]).reset_index(drop=True)

    # build features for each zone on the target date using historical data
    zones = df["zone_name"].unique()
    prediction_rows = []

    # precompute campaign load features for target date
    camp_zones = campaigns[["campaign_id", "zone_name", "start_date", "end_date",
                            "planned_impression", "campaign_duration_days",
                            "revenue_type"]].drop_duplicates(subset=["campaign_id", "zone_name"])

    active_camps = camp_zones[
        (camp_zones["start_date"] <= target_date) &
        (camp_zones["end_date"] >= target_date)
    ].copy()

    # cumulative served per campaign
    cum_served = (
        campaign_delivery.groupby(["campaign_id", "event_date"])["served_impressions"]
        .sum().reset_index().sort_values(["campaign_id", "event_date"])
    )
    cum_served["cumulative_served"] = cum_served.groupby("campaign_id")["served_impressions"].cumsum()
    latest_served = cum_served.groupby("campaign_id")["cumulative_served"].last().reset_index()

    if not active_camps.empty:
        active_camps["remaining_days"] = (active_camps["end_date"] - target_date).dt.days.clip(lower=1)
        active_camps = active_camps.merge(latest_served, on="campaign_id", how="left")
        active_camps["cumulative_served"] = active_camps["cumulative_served"].fillna(0)
        active_camps["daily_demand"] = (
            (active_camps["planned_impression"] - active_camps["cumulative_served"]) /
            active_camps["remaining_days"]
        ).clip(lower=0)

        zone_demand = active_camps.groupby("zone_name").agg(
            total_planned_daily_demand=("daily_demand", "sum"),
            paid_count=("revenue_type", lambda x: (x == "paid").sum()),
            total_count=("revenue_type", "count"),
            active_campaigns_in_zone=("campaign_id", "nunique")
        ).reset_index()
        zone_demand["paid_ratio"] = zone_demand["paid_count"] / zone_demand["total_count"]
    else:
        zone_demand = pd.DataFrame(columns=["zone_name", "total_planned_daily_demand",
                                             "paid_ratio", "active_campaigns_in_zone"])

    for zone in zones:
        zone_data = df[df["zone_name"] == zone].sort_values("event_date")
        if len(zone_data) < 7:
            continue

        ipf = zone_data["imp_per_flight"]
        last_row = zone_data.iloc[-1]

        row = {}
        row["lag_1d"] = ipf.iloc[-1]
        row["lag_7d"] = ipf.iloc[-7] if len(ipf) >= 7 else ipf.iloc[0]
        row["rolling_mean_7d"] = ipf.tail(7).mean()
        row["rolling_mean_30d"] = ipf.tail(30).mean()
        row["rolling_std_7d"] = ipf.tail(7).std()
        row["active_campaigns_in_zone"] = last_row.get("active_campaigns_in_zone", 0)
        # days since last observation for this zone
        last_date = zone_data["event_date"].iloc[-1]
        row["days_since_last"] = (target_date - last_date).days

        # zone category
        cat = "other"
        for c in ["home", "tv", "movies", "movie", "fb", "boarding", "seatback",
                   "map", "music", "games", "shopping", "welcome"]:
            if c in zone.lower():
                cat = "movies" if c == "movie" else c
                break

        # campaign load from precomputed
        zd = zone_demand[zone_demand["zone_name"] == zone]
        if not zd.empty:
            row["total_planned_daily_demand"] = zd.iloc[0]["total_planned_daily_demand"]
            row["paid_ratio"] = zd.iloc[0]["paid_ratio"]
            row["active_campaigns_in_zone"] = zd.iloc[0]["active_campaigns_in_zone"]
        else:
            row["total_planned_daily_demand"] = 0
            row["paid_ratio"] = 0

        # time features
        row["day_of_week"] = target_date.dayofweek
        row["is_weekend"] = 1 if target_date.dayofweek >= 5 else 0
        row["month"] = target_date.month

        # one-hot encode zone_category
        for feat in model1_features:
            if feat.startswith("cat_"):
                expected_cat = feat.replace("cat_", "")
                row[feat] = 1 if cat == expected_cat else 0

        prediction_rows.append({"zone_name": zone, **row})

    pred_df = pd.DataFrame(prediction_rows)

    # ensure all feature columns exist
    for feat in model1_features:
        if feat not in pred_df.columns:
            pred_df[feat] = 0

    X = pred_df[model1_features]
    # clip to >= 0 (negative predictions would break the LP with infeasible constraints)
    pred_df["predicted_imp_per_flight"] = np.clip(model1.predict(X), 0, None)

    # convert to total impressions using recent average flight count
    # sort by date and filter to dates before target_date to avoid future junk data
    fleet_sorted = fleet_daily[fleet_daily["event_date"] < target_date].sort_values("event_date")
    recent_flights = fleet_sorted.tail(7)["active_tails"].mean()
    pred_df["daily_capacity"] = pred_df["predicted_imp_per_flight"] * recent_flights

    zone_capacities = dict(zip(pred_df["zone_name"], pred_df["daily_capacity"]))

    print(f"  Forecasted {len(zone_capacities)} zones for {target_date.date()}")
    print(f"  Using avg {recent_flights:.0f} flights (last 7 days)")
    print(f"  Total fleet capacity: {sum(zone_capacities.values()):,.0f} impressions")

    return zone_capacities, pred_df


# ══════════════════════════════════════════════════════════════
# STEP 2: PRIORITY SCORE
# Scores all active campaigns: Pacing Gap + Time Urgency +
# Zone Fit + Contract Weight + Goal Modifier = 0-100
# ══════════════════════════════════════════════════════════════

def step2_priority_score(campaigns, zone_capacities, pacing_data=None):
    today = pd.Timestamp.now().normalize()

    # active campaigns: started and not ended
    active = campaigns[
        (campaigns["start_date"] <= today) &
        (campaigns["end_date"] >= today)
    ].drop_duplicates(subset=["campaign_id"]).copy()

    if active.empty:
        print("  No active campaigns found.")
        return pd.DataFrame()

    scores = []

    # precompute max urgency for normalization
    # guard against 0 or NaN duration (would cause inf/NaN urgency and break priority scores)
    active["campaign_duration_days"] = active["campaign_duration_days"].clip(lower=1)
    active["urgency_raw"] = 1.0 / np.sqrt(active["campaign_duration_days"])
    max_urgency = active["urgency_raw"].max()

    for _, camp in active.iterrows():
        cid = camp["campaign_id"]

        # 1. Pacing Gap (0-20)
        if pacing_data is not None and cid in pacing_data:
            pacing_ratio = pacing_data[cid]
            gap = max(0, 1.0 - pacing_ratio)
            pacing_gap = gap * 20
        else:
            pacing_gap = 0.0

        # 2. Time Urgency (0-20)
        time_urgency = (camp["urgency_raw"] / max_urgency) * 20 if max_urgency > 0 else 0

        # 3. Zone Fit (0-20)
        # sum capacity of campaign's assigned zones
        camp_zones = campaigns[campaigns["campaign_id"] == cid]["zone_name"].unique()
        total_capacity = sum(zone_capacities.get(z, 0) for z in camp_zones)
        pdi = camp["planned_daily_impressions"]
        if pdi > 0:
            ratio = total_capacity / pdi
            zone_fit = min(ratio * 10, 20.0)
        else:
            zone_fit = 20.0

        # 4. Contract Weight (0-20)
        contract_weights = {"paid": 3.0, "partner": 2.0, "house": 1.0}
        cw = contract_weights.get(camp["revenue_type"], 1.0)
        contract_weight = (cw / 3.0) * 20

        # 5. Goal Modifier (0-20)
        goal_modifiers = {"consideration": 1.2, "awareness": 1.0}
        gm = goal_modifiers.get(camp["goal_type"], 1.0)
        goal_modifier = (gm / 1.2) * 20

        priority_score = pacing_gap + time_urgency + zone_fit + contract_weight + goal_modifier

        scores.append({
            "campaign_id": cid,
            "campaign_name": camp["campaign_name"],
            "revenue_type": camp["revenue_type"],
            "goal_type": camp["goal_type"],
            "planned_daily_impressions": pdi,
            "pacing_gap": round(pacing_gap, 2),
            "time_urgency": round(time_urgency, 2),
            "zone_fit": round(zone_fit, 2),
            "contract_weight": round(contract_weight, 2),
            "goal_modifier": round(goal_modifier, 2),
            "priority_score": round(priority_score, 2),
        })

    priority_df = pd.DataFrame(scores).sort_values("priority_score", ascending=False).reset_index(drop=True)
    priority_df["queue_position"] = range(1, len(priority_df) + 1)

    print(f"  Scored {len(priority_df)} active campaigns")
    print(f"  Top 5:")
    for _, row in priority_df.head(5).iterrows():
        print(f"    #{int(row['queue_position'])} {row['campaign_name'][:50]} "
              f"→ {row['priority_score']:.1f}")

    return priority_df


# ══════════════════════════════════════════════════════════════
# STEP 4: RECOMMENDATION & OPTIMIZATION (LP)
# Allocates campaigns to zones using linear programming
# ══════════════════════════════════════════════════════════════

def step4_optimize(campaigns, zone_capacities, priority_df, campaign_delivery):
    today = pd.Timestamp.now().normalize()

    # get active campaign-zone pairs
    active_ids = set(priority_df["campaign_id"])
    camp_zone_pairs = campaigns[
        campaigns["campaign_id"].isin(active_ids)
    ][["campaign_id", "zone_name"]].drop_duplicates()

    if camp_zone_pairs.empty:
        print("  No campaign-zone pairs to optimize.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # priority lookup
    priority_map = dict(zip(priority_df["campaign_id"], priority_df["priority_score"]))

    # compute ZPR per zone from recent delivery vs capacity
    # ZPR = actual recent impressions / forecasted capacity
    recent_delivery = campaign_delivery[
        campaign_delivery["event_date"] >= today - pd.Timedelta(days=7)
    ]
    zone_actual = recent_delivery.groupby("zone_name")["served_impressions"].sum().reset_index()
    zone_actual["daily_actual"] = zone_actual["served_impressions"] / 7

    zpr_map = {}
    for _, row in zone_actual.iterrows():
        zn = row["zone_name"]
        cap = zone_capacities.get(zn, 0)
        zpr_map[zn] = row["daily_actual"] / cap if cap > 0 else 0

    # build LP
    pairs = camp_zone_pairs.values.tolist()
    n_vars = len(pairs)

    if n_vars == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # objective: maximize sum(priority * zpr * x) → minimize negative
    c = []
    for cid, zn in pairs:
        pri = priority_map.get(cid, 0)
        zpr = zpr_map.get(zn, 1.0)
        c.append(-(pri * max(zpr, 0.01)))

    # constraint 1: per zone, sum of allocations ≤ capacity
    zones_in_lp = list(set(zn for _, zn in pairs))
    A_ub = []
    b_ub = []

    for zone in zones_in_lp:
        row = [0] * n_vars
        for i, (cid, zn) in enumerate(pairs):
            if zn == zone:
                row[i] = 1
        A_ub.append(row)
        b_ub.append(zone_capacities.get(zone, 0))

    # constraint 2: per campaign, sum of allocations ≤ planned_daily
    campaigns_in_lp = list(set(cid for cid, _ in pairs))
    # fillna(0) prevents NaN from entering LP constraints (NaN in b_ub breaks linprog)
    camp_dedup = campaigns.drop_duplicates("campaign_id")
    pdi_map = dict(zip(camp_dedup["campaign_id"], camp_dedup["planned_daily_impressions"].fillna(0)))

    for camp_id in campaigns_in_lp:
        row = [0] * n_vars
        for i, (cid, zn) in enumerate(pairs):
            if cid == camp_id:
                row[i] = 1
        A_ub.append(row)
        b_ub.append(pdi_map.get(camp_id, 0))

    # bounds: each variable ≥ 0
    bounds = [(0, None)] * n_vars

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        allocations = result.x
        lp_status = "solved"
    else:
        # fallback: equal split per zone
        print("  LP infeasible, using equal split fallback.")
        lp_status = "fallback"
        allocations = []
        for cid, zn in pairs:
            camps_in_zone = sum(1 for c, z in pairs if z == zn)
            cap = zone_capacities.get(zn, 0)
            allocations.append(cap / camps_in_zone if camps_in_zone > 0 else 0)
        allocations = np.array(allocations)

    # build allocation DataFrame
    alloc_rows = []
    for i, (cid, zn) in enumerate(pairs):
        alloc_rows.append({
            "campaign_id": cid,
            "zone_name": zn,
            "allocated_impressions": round(allocations[i], 2),
            "zone_zpr": round(zpr_map.get(zn, 1.0), 4),
            "zone_capacity": round(zone_capacities.get(zn, 0), 2),
        })
    alloc_df = pd.DataFrame(alloc_rows)

    # allocation weight per campaign
    camp_totals = alloc_df.groupby("campaign_id")["allocated_impressions"].sum().reset_index()
    camp_totals.columns = ["campaign_id", "total_allocated"]
    alloc_df = alloc_df.merge(camp_totals, on="campaign_id")
    alloc_df["allocation_weight"] = np.where(
        alloc_df["total_allocated"] > 0,
        alloc_df["allocated_impressions"] / alloc_df["total_allocated"],
        0
    )

    # feasibility per campaign
    feasibility_rows = []
    for cid in campaigns_in_lp:
        camp_alloc = alloc_df[alloc_df["campaign_id"] == cid]
        total_alloc = camp_alloc["allocated_impressions"].sum()
        planned = pdi_map.get(cid, 0)
        fulfillment = (total_alloc / planned * 100) if planned > 0 else 100

        if fulfillment >= 95:
            confidence = "High"
        elif fulfillment >= 70:
            confidence = "Medium"
        else:
            confidence = "Low"

        name = priority_df[priority_df["campaign_id"] == cid]["campaign_name"].values
        name = name[0] if len(name) > 0 else "Unknown"

        feasibility_rows.append({
            "campaign_id": cid,
            "campaign_name": name,
            "planned_daily": round(planned, 2),
            "allocated_daily": round(total_alloc, 2),
            "fulfillment_pct": round(fulfillment, 2),
            "confidence": confidence,
        })
    feasibility_df = pd.DataFrame(feasibility_rows).sort_values("fulfillment_pct")

    # zone utilization
    zone_util_rows = []
    for zone in zones_in_lp:
        zone_alloc = alloc_df[alloc_df["zone_name"] == zone]["allocated_impressions"].sum()
        cap = zone_capacities.get(zone, 0)
        util = (zone_alloc / cap * 100) if cap > 0 else 0
        zone_util_rows.append({
            "zone_name": zone,
            "total_allocated": round(zone_alloc, 2),
            "capacity": round(cap, 2),
            "utilization_pct": round(util, 2),
            "zpr": round(zpr_map.get(zone, 1.0), 4),
        })
    zone_util_df = pd.DataFrame(zone_util_rows).sort_values("utilization_pct", ascending=False)

    conflicts = zone_util_df[zone_util_df["utilization_pct"] > 90]

    print(f"  LP {lp_status}: {len(alloc_df)} campaign-zone allocations")
    print(f"  Campaigns: {len(feasibility_df)}")
    high = (feasibility_df["confidence"] == "High").sum()
    med = (feasibility_df["confidence"] == "Medium").sum()
    low = (feasibility_df["confidence"] == "Low").sum()
    print(f"    High confidence: {high}")
    print(f"    Medium confidence: {med}")
    print(f"    Low confidence: {low}")
    print(f"  Zone conflicts (>90% utilization): {len(conflicts)}")

    return alloc_df, feasibility_df, zone_util_df


# ══════════════════════════════════════════════════════════════
# STEP 5: MONITORING
# Pacing, risk predictions, alerts, feedback to Step 2
# ══════════════════════════════════════════════════════════════

def step5_monitor(campaigns, campaign_delivery, fleet_daily, fleet_metrics,
                  model2, feasibility_df, zone_capacities):
    today = pd.Timestamp.now().normalize()

    # active campaigns with planned targets
    active = campaigns[
        (campaigns["start_date"] <= today) &
        (campaigns["end_date"] >= today) &
        (campaigns["planned_impression"] > 0)
    ].drop_duplicates(subset=["campaign_id"]).copy()

    if active.empty:
        print("  No active campaigns to monitor.")
        return pd.DataFrame(), {}, []

    # cumulative served per campaign
    cum = (
        campaign_delivery.groupby(["campaign_id", "event_date"])["served_impressions"]
        .sum().reset_index().sort_values(["campaign_id", "event_date"])
    )
    cum["cumulative_served"] = cum.groupby("campaign_id")["served_impressions"].cumsum()
    latest_cum = cum.groupby("campaign_id")["cumulative_served"].last().reset_index()

    # filter out future junk dates from fleet data
    fleet_flights = fleet_daily[fleet_daily["event_date"] <= today][["event_date", "active_tails"]].copy()

    # compute zone-level zpr (zone performance ratio) from recent delivery vs forecasted capacity
    # zpr = actual daily impressions / forecasted daily capacity per zone
    recent_delivery = campaign_delivery[
        campaign_delivery["event_date"] >= today - pd.Timedelta(days=7)
    ]
    zone_actual = recent_delivery.groupby("zone_name")["served_impressions"].sum().reset_index()
    zone_actual["daily_actual"] = zone_actual["served_impressions"] / 7
    zpr_map = {}
    for _, row in zone_actual.iterrows():
        zn = row["zone_name"]
        cap = zone_capacities.get(zn, 0)
        # if zone has no forecasted capacity, default to 1.0 (neutral)
        zpr_map[zn] = row["daily_actual"] / cap if cap > 0 else 1.0

    pacing_rows = []
    pacing_feedback = {}
    alerts = []

    for _, camp in active.iterrows():
        cid = camp["campaign_id"]
        start = camp["start_date"]
        end = camp["end_date"]
        planned = camp["planned_impression"]
        duration = camp["campaign_duration_days"]
        zones = camp["zones_per_campaign"]

        days_elapsed = (today - start).days
        if days_elapsed < 1:
            continue

        remaining_days = max((end - today).days, 1)

        # actual delivered
        served_row = latest_cum[latest_cum["campaign_id"] == cid]
        actual_delivered = served_row["cumulative_served"].values[0] if len(served_row) > 0 else 0

        # expected by now (linear pacing)
        expected_by_now = planned * (days_elapsed / duration)
        pacing_ratio = actual_delivered / expected_by_now if expected_by_now > 0 else 1.0

        # pacing status
        if pacing_ratio > 1.1:
            status = "Over-Pacing"
        elif pacing_ratio < 0.9:
            status = "Under-Pacing"
        else:
            status = "On-Track"

        # feasibility: can it still catch up?
        remaining_impressions = max(planned - actual_delivered, 0)
        needed_daily = remaining_impressions / remaining_days

        # available daily = sum of forecasted capacity for this campaign's zones
        camp_zones = campaigns[campaigns["campaign_id"] == cid]["zone_name"].unique()
        available_daily = sum(zone_capacities.get(z, 0) for z in camp_zones)

        feasibility = "OK" if needed_daily <= available_daily else "At Risk"

        # avg zone zpr: average performance ratio across this campaign's zones
        # zpr = actual/forecasted, tells model 2 if zones are over/underperforming
        zone_zprs = [zpr_map.get(z, 1.0) for z in camp_zones]
        avg_zone_zpr = np.mean(zone_zprs) if zone_zprs else 1.0

        # performance index
        pi = pacing_ratio * avg_zone_zpr

        # Model 2 risk prediction
        fraction_elapsed = days_elapsed / duration
        pdi = camp["planned_daily_impressions"]

        # impressions per flight for this campaign
        camp_delivery = campaign_delivery[campaign_delivery["campaign_id"] == cid]
        camp_with_flights = camp_delivery.merge(fleet_flights, on="event_date", how="left")
        camp_with_flights = camp_with_flights.dropna(subset=["active_tails"])
        if len(camp_with_flights) > 0 and camp_with_flights["active_tails"].sum() > 0:
            imp_per_flight = (
                camp_with_flights["served_impressions"].sum() /
                camp_with_flights["active_tails"].sum()
            )
        else:
            imp_per_flight = 0

        goal_encoded = 1 if camp["goal_type"] == "consideration" else 0
        rev_map = {"paid": 2, "partner": 1, "house": 0}
        rev_encoded = rev_map.get(camp["revenue_type"], 0)

        # needed_vs_runway: how much the campaign needs to speed up
        current_daily = actual_delivered / days_elapsed if days_elapsed > 0 else 0
        needed_vs_runway = needed_daily / max(current_daily, 1.0)
        needed_vs_runway = min(needed_vs_runway, 20.0)  # clip like training

        m2_features = np.array([[
            pacing_ratio, fraction_elapsed, zones, pdi,
            imp_per_flight, needed_vs_runway, avg_zone_zpr, goal_encoded, rev_encoded
        ]])
        predicted_log = model2.predict(m2_features)[0]
        predicted_delivery = np.expm1(predicted_log)

        # alert based on PI
        if pi < 0.5:
            severity = "CRITICAL"
        elif pi < 0.8:
            severity = "WARNING"
        elif pi < 1.0:
            severity = "WATCH"
        else:
            severity = "OK"

        if severity in ("CRITICAL", "WARNING"):
            alerts.append({
                "campaign_id": cid,
                "campaign_name": camp["campaign_name"],
                "severity": severity,
                "pi": round(pi, 4),
                "pacing_ratio": round(pacing_ratio, 4),
                "predicted_delivery_pct": round(predicted_delivery * 100, 1),
                "root_cause": f"Pacing at {pacing_ratio:.2f}, zones at {avg_zone_zpr:.2f}",
            })

        pacing_feedback[cid] = pacing_ratio

        pacing_rows.append({
            "campaign_id": cid,
            "campaign_name": camp["campaign_name"],
            "revenue_type": camp["revenue_type"],
            "planned": planned,
            "delivered": round(actual_delivered),
            "delivery_pct": round(actual_delivered / planned * 100, 1),
            "pacing_ratio": round(pacing_ratio, 4),
            "status": status,
            "needed_daily": round(needed_daily),
            "available_daily": round(available_daily),
            "feasibility": feasibility,
            "remaining_days": remaining_days,
            "pi": round(pi, 4),
            "predicted_final_delivery_pct": round(predicted_delivery * 100, 1),
            "severity": severity,
        })

    pacing_df = pd.DataFrame(pacing_rows).sort_values("pi")

    # fleet trend alert
    fleet_alerts = []
    # filter out future junk dates and sort before grabbing recent data
    fleet_sorted = fleet_daily[fleet_daily["event_date"] <= today].sort_values("event_date")
    recent_avg = fleet_sorted.tail(7)["total_impression_cnt"].mean()
    fleet_avg = fleet_metrics["daily_avg_impressions"]
    trend = recent_avg / fleet_avg if fleet_avg > 0 else 1.0
    if trend < 0.85:
        fleet_alerts.append(f"FLEET ALERT: Impressions dropped {(1-trend)*100:.0f}% vs average")
    elif trend < 0.95:
        fleet_alerts.append(f"FLEET WARNING: Impressions down {(1-trend)*100:.0f}% vs average")

    print(f"  Monitored {len(pacing_df)} campaigns")
    on_track = (pacing_df["status"] == "On-Track").sum()
    under = (pacing_df["status"] == "Under-Pacing").sum()
    over = (pacing_df["status"] == "Over-Pacing").sum()
    print(f"    On-Track: {on_track}")
    print(f"    Under-Pacing: {under}")
    print(f"    Over-Pacing: {over}")
    print(f"  Alerts: {len(alerts)} campaigns flagged")
    for a in alerts[:5]:
        print(f"    [{a['severity']}] {a['campaign_name'][:40]} "
              f"PI={a['pi']}, predicted={a['predicted_delivery_pct']:.0f}%")
    for fa in fleet_alerts:
        print(f"  {fa}")

    return pacing_df, pacing_feedback, alerts


# ══════════════════════════════════════════════════════════════
# RUN FULL PIPELINE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ONEMEDIA CAMPAIGN OPTIMIZATION PIPELINE")
    print("=" * 60)

    print("\nLoading data and models...")
    (campaigns, zone_daily, fleet_daily, campaign_delivery,
     fleet_metrics, model1, model1_features, model2) = load_data_and_models()

    # ── STEP 3: Zone Capacity Forecasting ──
    print("\n" + "-" * 60)
    print("STEP 3: ZONE CAPACITY FORECASTING")
    print("-" * 60)
    zone_capacities, zone_forecast_df = step3_forecast_zone_capacity(
        model1, model1_features, zone_daily, fleet_daily, campaigns, campaign_delivery
    )

    # ── STEP 2: Priority Score (first pass, no pacing data yet) ──
    print("\n" + "-" * 60)
    print("STEP 2: PRIORITY SCORING (initial, no pacing data)")
    print("-" * 60)
    priority_df = step2_priority_score(campaigns, zone_capacities)

    if priority_df.empty:
        print("No active campaigns. Exiting.")
        exit()

    # ── STEP 4: LP Optimization ──
    print("\n" + "-" * 60)
    print("STEP 4: RECOMMENDATION & OPTIMIZATION")
    print("-" * 60)
    alloc_df, feasibility_df, zone_util_df = step4_optimize(
        campaigns, zone_capacities, priority_df, campaign_delivery
    )

    # ── STEP 5: Monitoring ──
    print("\n" + "-" * 60)
    print("STEP 5: MONITORING & RISK PREDICTION")
    print("-" * 60)
    pacing_df, pacing_feedback, alerts = step5_monitor(
        campaigns, campaign_delivery, fleet_daily, fleet_metrics,
        model2, feasibility_df, zone_capacities
    )

    # ── FEEDBACK LOOP: Step 5 → Step 2 (re-score with pacing data) ──
    print("\n" + "-" * 60)
    print("FEEDBACK LOOP: RE-SCORING WITH PACING DATA")
    print("-" * 60)
    priority_df_v2 = step2_priority_score(campaigns, zone_capacities, pacing_data=pacing_feedback)

    # ── SAVE OUTPUTS ──
    print("\n" + "-" * 60)
    print("SAVING RESULTS")
    print("-" * 60)

    zone_forecast_df.to_csv(os.path.join(DATA_DIR, "output_zone_forecasts.csv"), index=False)
    priority_df_v2.to_csv(os.path.join(DATA_DIR, "output_priority_scores.csv"), index=False)
    alloc_df.to_csv(os.path.join(DATA_DIR, "output_allocations.csv"), index=False)
    feasibility_df.to_csv(os.path.join(DATA_DIR, "output_feasibility.csv"), index=False)
    zone_util_df.to_csv(os.path.join(DATA_DIR, "output_zone_utilization.csv"), index=False)
    pacing_df.to_csv(os.path.join(DATA_DIR, "output_pacing.csv"), index=False)

    if alerts:
        pd.DataFrame(alerts).to_csv(os.path.join(DATA_DIR, "output_alerts.csv"), index=False)

    print("  Saved all output CSVs.")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Zone forecasts:    {len(zone_capacities)} zones")
    print(f"  Priority scores:   {len(priority_df_v2)} campaigns")
    print(f"  LP allocations:    {len(alloc_df)} campaign-zone pairs")
    print(f"  Monitoring:        {len(pacing_df)} campaigns tracked")
    print(f"  Alerts:            {len(alerts)} campaigns flagged")
