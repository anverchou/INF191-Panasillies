import os
import pandas as pd
import numpy as np
from datetime import datetime
from pyathena import connect

AWS_REGION = "us-west-2"
ATHENA_S3_OUTPUT = "s3://aws-athena-query-results-296959725007-us-west-2/"

TODAY_DT = datetime.now().strftime("%Y%m%d")

conn = connect(
    s3_staging_dir=ATHENA_S3_OUTPUT,
    region_name=AWS_REGION
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


QUERY_CAMPAIGNS = """
SELECT
    c.id AS campaign_id,
    c.created AS campaign_creation,
    c.name AS campaign_name,
    c.start_date,
    c.end_date,
    c.airline_id,
    c.revenue_type,
    c.goal_type,
    c.creative_type,
    c.type AS campaign_type,
    cs.planned_impression,
    cs.flight_impression_capping,
    z.id AS zone_id,
    z.name AS zone_name,
    z.path AS zone_path,
    z.type AS zone_type,
    CAST(z.created AS DATE) AS zone_created_date,
    z.status AS zone_status,
    a.id AS advertiser_id,
    a.name AS advertiser_name

FROM dl_pac_dp_onemedia3_sql_raw_prod_v1.campaigns c
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.advertisers a ON c.advertiser_id = a.id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.adgroups ag ON ag.campaign_id = c.id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.adgroup_include_zones aiz ON aiz.adgroup_id = ag.id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.zones z ON z.id = aiz.zone_id
LEFT JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.campaign_settings cs ON cs.campaign_id = c.id

WHERE c.workflow_status = 'approved'
    AND a.is_test = 0
    AND c.type IN ('banner', 'interstitial')
    AND a.dt >= '20250101'

UNION ALL

SELECT
    c.id AS campaign_id,
    c.created AS campaign_creation,
    c.name AS campaign_name,
    c.start_date,
    c.end_date,
    c.airline_id,
    c.revenue_type,
    c.goal_type,
    c.creative_type,
    c.type AS campaign_type,
    cs.planned_impression,
    cs.flight_impression_capping,
    z.id AS zone_id,
    z.name AS zone_name,
    z.path AS zone_path,
    z.type AS zone_type,
    CAST(z.created AS DATE) AS zone_created_date,
    z.status AS zone_status,
    a.id AS advertiser_id,
    a.name AS advertiser_name

FROM dl_pac_dp_onemedia3_sql_raw_prod_v1.campaigns c
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.advertisers a ON c.advertiser_id = a.id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.adgroups ag ON ag.campaign_id = c.id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.video_group_contents vgc ON vgc.adgroup_id = ag.id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.video_groups vg ON vg.id = vgc.video_group_id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.video_group_include_zones vgiz ON vgiz.video_group_id = vg.id
INNER JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.zones z ON z.id = vgiz.zone_id
LEFT JOIN dl_pac_dp_onemedia3_sql_raw_prod_v1.campaign_settings cs ON cs.campaign_id = c.id

WHERE c.workflow_status = 'approved'
    AND a.is_test = 0
    AND c.type = 'video'
    AND a.dt >= '20250101'
"""


QUERY_ZONE_DAILY = """
SELECT
    zone_id,
    zone_name,
    icao_airline,
    CONCAT(
        SUBSTRING(event_timestamp, 1, 4), '-',
        SUBSTRING(event_timestamp, 5, 2), '-',
        SUBSTRING(event_timestamp, 7, 2)
    ) AS event_date,
    COUNT(DISTINCT campaign_id) AS active_campaigns_in_zone,
    COUNT(CASE WHEN event_type = 'i' THEN 1 END) AS total_impression_cnt

FROM dl_pac_ife_om3_curated_v1_prod.onemedia_v1

WHERE dt BETWEEN '20250101' AND '{today}'
    AND advertiser_is_test = 0
    AND icao_airline = 'UAL'

GROUP BY
    zone_id,
    zone_name,
    icao_airline,
    CONCAT(
        SUBSTRING(event_timestamp, 1, 4), '-',
        SUBSTRING(event_timestamp, 5, 2), '-',
        SUBSTRING(event_timestamp, 7, 2)
    )
"""

QUERY_CAMPAIGN_DELIVERY = """
SELECT
    campaign_id,
    campaign_name,
    campaign_revenue_type,
    campaign_goal_type,
    campaign_type,
    zone_id,
    zone_name,
    icao_airline,
    CONCAT(
        SUBSTRING(event_timestamp, 1, 4), '-',
        SUBSTRING(event_timestamp, 5, 2), '-',
        SUBSTRING(event_timestamp, 7, 2)
    ) AS event_date,
    COUNT(CASE WHEN event_type = 'i' THEN 1 END) AS served_impressions,
    COUNT(CASE WHEN event_type = 'i' AND LOWER(TRIM(COALESCE(campaign_type, 'unk'))) <> 'video'
              AND campaign_goal_type = 'consideration' THEN 1 END) AS clickable_impressions,
    COUNT(CASE WHEN event_type = 'c' THEN 1 END) AS clicks,
    COUNT(CASE WHEN event_type = 'i' AND LOWER(TRIM(COALESCE(campaign_type, 'unk'))) = 'video' THEN 1 END) AS video_impressions,
    COUNT(CASE WHEN watch_event_type = 'SKIPPED' THEN 1 END) AS skip_cnt,
    COUNT(CASE WHEN watch_event_type = 'COMPLETED' THEN 1 END) AS complete_cnt

FROM dl_pac_ife_om3_curated_v1_prod.onemedia_v1

WHERE dt BETWEEN '20250101' AND '{today}'
    AND advertiser_is_test = 0
    AND icao_airline = 'UAL'

GROUP BY
    campaign_id,
    campaign_name,
    campaign_revenue_type,
    campaign_goal_type,
    campaign_type,
    zone_id,
    zone_name,
    icao_airline,
    CONCAT(
        SUBSTRING(event_timestamp, 1, 4), '-',
        SUBSTRING(event_timestamp, 5, 2), '-',
        SUBSTRING(event_timestamp, 7, 2)
    )
"""

QUERY_FLEET_DAILY = """
SELECT
    icao_airline,
    CONCAT(
        SUBSTRING(event_timestamp, 1, 4), '-',
        SUBSTRING(event_timestamp, 5, 2), '-',
        SUBSTRING(event_timestamp, 7, 2)
    ) AS event_date,
    COUNT(CASE WHEN event_type = 'i' THEN 1 END) AS total_impression_cnt,
    COUNT(CASE WHEN event_type = 'i' AND LOWER(TRIM(COALESCE(campaign_type, 'unk'))) <> 'video' THEN 1 END) AS non_video_impression_cnt,
    COUNT(CASE WHEN event_type = 'i' AND LOWER(TRIM(COALESCE(campaign_type, 'unk'))) <> 'video'
              AND campaign_goal_type = 'consideration' THEN 1 END) AS clickable_impression_cnt,
    COUNT(CASE WHEN event_type = 'c' THEN 1 END) AS total_click_cnt,
    COUNT(CASE WHEN event_type = 'i' AND LOWER(TRIM(COALESCE(campaign_type, 'unk'))) = 'video' THEN 1 END) AS video_impression_cnt,
    COUNT(CASE WHEN watch_event_type = 'SKIPPED' THEN 1 END) AS total_skip_cnt,
    COUNT(CASE WHEN watch_event_type = 'COMPLETED' THEN 1 END) AS total_complete_cnt,
    COUNT(DISTINCT tail_number) AS active_tails

FROM dl_pac_ife_om3_curated_v1_prod.onemedia_v1

WHERE dt BETWEEN '20250101' AND '{today}'
    AND advertiser_is_test = 0
    AND icao_airline = 'UAL'

GROUP BY
    icao_airline,
    CONCAT(
        SUBSTRING(event_timestamp, 1, 4), '-',
        SUBSTRING(event_timestamp, 5, 2), '-',
        SUBSTRING(event_timestamp, 7, 2)
    )
"""

QUERY_COMPLETED_CAMPAIGNS = """
SELECT
    campaign_id,
    campaign_name,
    campaign_revenue_type,
    campaign_goal_type,
    icao_airline,
    COUNT(DISTINCT zone_id) AS zones_per_campaign,
    COUNT(CASE WHEN event_type = 'i' THEN 1 END) AS total_served_impressions

FROM dl_pac_ife_om3_curated_v1_prod.onemedia_v1

WHERE dt BETWEEN '20250101' AND '{today}'
    AND advertiser_is_test = 0
    AND icao_airline = 'UAL'

GROUP BY
    campaign_id,
    campaign_name,
    campaign_revenue_type,
    campaign_goal_type,
    icao_airline
"""




def load_all_data():
    campaigns = pd.read_sql(QUERY_CAMPAIGNS, conn)

    zone_daily = pd.read_sql(QUERY_ZONE_DAILY.format(today=TODAY_DT), conn)

    campaign_delivery = pd.read_sql(QUERY_CAMPAIGN_DELIVERY.format(today=TODAY_DT), conn)

    fleet_daily = pd.read_sql(QUERY_FLEET_DAILY.format(today=TODAY_DT), conn)

    completed = pd.read_sql(QUERY_COMPLETED_CAMPAIGNS.format(today=TODAY_DT), conn)

    return campaigns, zone_daily, campaign_delivery, fleet_daily, completed




def process_campaigns(campaigns):


    # parse dates
    for col in ["start_date", "end_date", "campaign_creation", "zone_created_date"]:
        if col in campaigns.columns:
            campaigns[col] = pd.to_datetime(campaigns[col], errors="coerce")

    # fill missing values with defaults
    campaigns["planned_impression"] = campaigns["planned_impression"].fillna(0)
    campaigns["flight_impression_capping"] = campaigns["flight_impression_capping"].fillna(0)
    campaigns["creative_type"] = campaigns["creative_type"].fillna("image")

    # normalize revenue_type to lowercase ('Paid' → 'paid')
    campaigns["revenue_type"] = campaigns["revenue_type"].str.lower()

    # normalize goal_type to lowercase
    campaigns["goal_type"] = campaigns["goal_type"].str.lower()

    # exclude conversion goal type — not in scope per implementation doc
    before = campaigns["campaign_id"].nunique()
    campaigns = campaigns[campaigns["goal_type"] != "conversion"].copy()
    dropped = before - campaigns["campaign_id"].nunique()
    if dropped > 0:
        print(f"  Dropped {dropped} conversion campaigns")

    # extract zone_category from zone_name via regex
    campaigns["zone_category"] = (
        campaigns["zone_name"]
        .str.lower()
        .str.extract(
            r"(home|tv|movies|movie|fb|boarding|seatback|map|music|games|shopping|welcome)",
            expand=False
        )
        .fillna("other")
    )
    campaigns["zone_category"] = campaigns["zone_category"].replace("movie", "movies")

    # drop campaigns that ended before 2024
    before = campaigns["campaign_id"].nunique()
    campaigns = campaigns[campaigns["end_date"] >= pd.Timestamp("2024-01-01")].copy()
    dropped = before - campaigns["campaign_id"].nunique()
    if dropped > 0:
        print(f"  Dropped {dropped} campaigns with end_date before 2024")

    # exclude "forever" campaigns (end_date beyond 2027) — house ads, placeholders, no real target
    before = campaigns["campaign_id"].nunique()
    campaigns = campaigns[campaigns["end_date"] <= pd.Timestamp("2027-01-01")].copy()
    dropped = before - campaigns["campaign_id"].nunique()
    if dropped > 0:
        print(f"  Dropped {dropped} 'forever' campaigns (end_date > 2027)")

    # campaign_duration_days = end_date - start_date (minimum 1)
    campaigns["campaign_duration_days"] = (
        (campaigns["end_date"] - campaigns["start_date"]).dt.days
    ).clip(lower=1)

    # planned_daily_impressions = planned_impression / campaign_duration_days
    campaigns["planned_daily_impressions"] = (
        campaigns["planned_impression"] / campaigns["campaign_duration_days"]
    )

    # zones_per_campaign = count unique zones per campaign
    zones_per = campaigns.groupby("campaign_id")["zone_name"].nunique().reset_index()
    zones_per.columns = ["campaign_id", "zones_per_campaign"]
    campaigns = campaigns.merge(zones_per, on="campaign_id", how="left")


    return campaigns






def process_zone_daily(zone_daily):
    zone_daily["event_date"] = pd.to_datetime(zone_daily["event_date"], errors="coerce")
    zone_daily = zone_daily.dropna(subset=["event_date"])

    # filter to valid date range (2025-2026 only
    before = len(zone_daily)
    zone_daily = zone_daily[
        (zone_daily["event_date"] >= "2025-01-01") &
        (zone_daily["event_date"] <= "2026-12-31")
    ].copy()


    zone_daily = zone_daily.sort_values(["zone_name", "event_date"])


    return zone_daily



def process_campaign_delivery(campaign_delivery):
    campaign_delivery["event_date"] = pd.to_datetime(
        campaign_delivery["event_date"], errors="coerce"
    )
    campaign_delivery = campaign_delivery.dropna(subset=["event_date"])

    before = len(campaign_delivery)
    campaign_delivery = campaign_delivery[
        (campaign_delivery["event_date"] >= "2025-01-01") &
        (campaign_delivery["event_date"] <= "2026-12-31")
    ].copy()

    if "campaign_revenue_type" in campaign_delivery.columns:
        campaign_delivery["campaign_revenue_type"] = (
            campaign_delivery["campaign_revenue_type"].str.lower()
        )


    return campaign_delivery


def compute_fleet_metrics(fleet_daily):

    fleet_daily["event_date"] = pd.to_datetime(fleet_daily["event_date"], errors="coerce")
    fleet_daily = fleet_daily.dropna(subset=["event_date"])

    before = len(fleet_daily)
    fleet_daily = fleet_daily[
        (fleet_daily["event_date"] >= "2025-01-01") &
        (fleet_daily["event_date"] <= "2026-12-31")
    ].copy()

    fleet_daily = fleet_daily.sort_values("event_date")

    total_impressions = fleet_daily["total_impression_cnt"].sum()
    total_clicks = fleet_daily["total_click_cnt"].sum()
    total_clickable = fleet_daily["clickable_impression_cnt"].sum()
    total_video = fleet_daily["video_impression_cnt"].sum()
    total_completes = fleet_daily["total_complete_cnt"].sum()
    total_skips = fleet_daily["total_skip_cnt"].sum()

    # fleet_daily_avg = mean daily impressions across all dates
    daily_avg = fleet_daily["total_impression_cnt"].mean()
    # fleet_daily_std = standard deviation
    daily_std = fleet_daily["total_impression_cnt"].std()

    # fleet_ctr = total clicks / total clickable impressions
    ctr = total_clicks / total_clickable if total_clickable > 0 else 0
    # video_completion_rate
    video_completion_rate = (
        total_completes / (total_completes + total_skips)
        if (total_completes + total_skips) > 0 else 0
    )

    # 7-day trend: compare last 7 days avg to overall avg
    last_7 = fleet_daily.tail(7)["total_impression_cnt"].mean()
    trend_ratio = last_7 / daily_avg if daily_avg > 0 else 1.0
    if trend_ratio > 1.05:
        trend = "UP"
    elif trend_ratio < 0.95:
        trend = "DOWN"
    else:
        trend = "STABLE"

    fleet_metrics = {
        "daily_avg_impressions": daily_avg,
        "daily_std_impressions": daily_std,
        "ctr": ctr,
        "video_completion_rate": video_completion_rate,
        "trend_ratio": trend_ratio,
        "trend": trend,
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_days": len(fleet_daily),
    }



    return fleet_metrics, fleet_daily



def label_completed_campaigns(completed, campaigns):

    # aggregate Query 5 by campaign_id — same campaign can appear with different names
    # sum impressions, take max zones, keep first name/revenue_type/goal_type
    completed_agg = completed.groupby("campaign_id").agg(
        campaign_name=("campaign_name", "first"),
        campaign_revenue_type=("campaign_revenue_type", "first"),
        campaign_goal_type=("campaign_goal_type", "first"),
        zones_per_campaign=("zones_per_campaign", "max"),
        total_served_impressions=("total_served_impressions", "sum"),
    ).reset_index()

    # get planned_impression per campaign (deduplicate — campaigns has one row per zone)
    planned = (
        campaigns[["campaign_id", "planned_impression", "end_date",
                    "revenue_type", "goal_type"]]
        .drop_duplicates(subset=["campaign_id"])
    )

    labeled = completed_agg.merge(planned, on="campaign_id", how="inner")

    # drop duplicate columns from Query 5 (keep Query 1's revenue_type/goal_type)
    labeled = labeled.drop(columns=["campaign_revenue_type", "campaign_goal_type"])

    # only completed campaigns (end_date in the past)
    today = pd.Timestamp.now()
    labeled = labeled[labeled["end_date"] < today].copy()

    # only campaigns that had a planned target (> 0)
    labeled = labeled[labeled["planned_impression"] > 0].copy()

    # label: failed = 1, met target = 0
    labeled["label"] = (
        labeled["total_served_impressions"] < labeled["planned_impression"]
    ).astype(int)

    # delivery percentage
    labeled["delivery_pct"] = (
        labeled["total_served_impressions"] / labeled["planned_impression"] * 100
    ).round(2)



    return labeled



def save_all_csvs(campaigns, zone_daily, campaign_delivery, fleet_daily, fleet_metrics, completed_labeled):

    campaigns.to_csv(os.path.join(OUTPUT_DIR, "campaigns_cleaned.csv"), index=False)

    zone_daily.to_csv(os.path.join(OUTPUT_DIR, "zone_daily.csv"), index=False)

    campaign_delivery.to_csv(os.path.join(OUTPUT_DIR, "campaign_delivery.csv"), index=False)

    fleet_daily.to_csv(os.path.join(OUTPUT_DIR, "fleet_daily.csv"), index=False)

    # fleet metrics as single-row CSV
    metrics_df = pd.DataFrame([fleet_metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "fleet_metrics.csv"), index=False)

    completed_labeled.to_csv(os.path.join(OUTPUT_DIR, "completed_labeled.csv"), index=False)




if __name__ == "__main__":


    # load raw data from Athena
    raw_campaigns, raw_zone_daily, raw_delivery, raw_fleet, raw_completed = load_all_data()



    # parse dates, fill defaults, compute derived columns
    campaigns = process_campaigns(raw_campaigns)


    # parse dates on zone daily history
    zone_daily = process_zone_daily(raw_zone_daily)


    # parse dates on campaign delivery
    campaign_delivery = process_campaign_delivery(raw_delivery)

    # compute fleet-level baseline metrics
    fleet_metrics, fleet_daily = compute_fleet_metrics(raw_fleet)

    # label completed campaigns for classifier training
    completed_labeled = label_completed_campaigns(raw_completed, campaigns)

    # save all to CSVs

    save_all_csvs(campaigns, zone_daily, campaign_delivery, fleet_daily,
                  fleet_metrics, completed_labeled)
