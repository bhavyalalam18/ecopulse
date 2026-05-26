-- ============================================================
-- EcoPulse SQL Queries
-- Demonstrates: Window Functions, CTEs, Views, Aggregations
-- ============================================================

-- 1. CREATE VIEW: Feature table used by ML pipeline
CREATE VIEW IF NOT EXISTS vw_hourly_features AS
SELECT
    datetime,
    PJME_MW,
    year, month, day, hour, dayofweek,
    is_weekend, quarter,
    hour_sin, hour_cos,
    month_sin, month_cos,
    dow_sin, dow_cos,
    lag_1h, lag_24h, lag_168h,
    rolling_mean_24h, rolling_std_24h, rolling_mean_168h
FROM energy_hourly;


-- 2. WINDOW FUNCTION: Rank hours within each month by average MW
SELECT
    month,
    hour,
    ROUND(AVG(PJME_MW), 2) AS avg_mw,
    RANK() OVER (
        PARTITION BY month
        ORDER BY AVG(PJME_MW) DESC
    ) AS rank_in_month
FROM energy_hourly
GROUP BY month, hour
ORDER BY month, rank_in_month;


-- 3. WINDOW FUNCTION: 7-day rolling average (168 hours)
SELECT
    datetime,
    PJME_MW,
    ROUND(
        AVG(PJME_MW) OVER (
            ORDER BY datetime
            ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_7d_avg,
    ROUND(
        AVG(PJME_MW) OVER (
            ORDER BY datetime
            ROWS BETWEEN 719 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_30d_avg
FROM energy_hourly
ORDER BY datetime;


-- 4. WINDOW FUNCTION: ROW_NUMBER to find peak hour each year
WITH yearly_peaks AS (
    SELECT
        year,
        datetime,
        PJME_MW,
        ROW_NUMBER() OVER (
            PARTITION BY year
            ORDER BY PJME_MW DESC
        ) AS rn
    FROM energy_hourly
)
SELECT year, datetime AS peak_datetime, ROUND(PJME_MW, 2) AS peak_mw
FROM yearly_peaks
WHERE rn = 1
ORDER BY year;


-- 5. CTE: Year-over-year change in average monthly demand
WITH monthly_avg AS (
    SELECT
        year,
        month,
        ROUND(AVG(PJME_MW), 2) AS avg_mw
    FROM energy_hourly
    GROUP BY year, month
),
yoy AS (
    SELECT
        a.year,
        a.month,
        a.avg_mw AS current_avg,
        b.avg_mw AS prev_avg,
        ROUND((a.avg_mw - b.avg_mw) / b.avg_mw * 100, 2) AS yoy_pct_change
    FROM monthly_avg a
    LEFT JOIN monthly_avg b
        ON a.year = b.year + 1 AND a.month = b.month
)
SELECT * FROM yoy WHERE prev_avg IS NOT NULL ORDER BY year, month;


-- 6. Peak demand hours: top 10% threshold
WITH percentile_calc AS (
    SELECT
        PJME_MW,
        NTILE(10) OVER (ORDER BY PJME_MW) AS decile
    FROM energy_hourly
)
SELECT
    MIN(PJME_MW) AS p90_threshold,
    MAX(PJME_MW) AS max_mw
FROM percentile_calc
WHERE decile = 10;


-- 7. Weekend vs Weekday comparison per year
SELECT
    year,
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    ROUND(AVG(PJME_MW), 2)  AS avg_mw,
    ROUND(MAX(PJME_MW), 2)  AS peak_mw,
    ROUND(MIN(PJME_MW), 2)  AS min_mw,
    COUNT(*)                 AS hours
FROM energy_hourly
GROUP BY year, is_weekend
ORDER BY year, is_weekend;


-- 8. LAG self-join: same hour, consecutive years
SELECT
    a.year        AS year,
    a.month,
    a.hour,
    ROUND(AVG(a.PJME_MW), 2) AS avg_mw_current,
    ROUND(AVG(b.PJME_MW), 2) AS avg_mw_prev_year,
    ROUND(AVG(a.PJME_MW - b.PJME_MW), 2) AS delta_mw
FROM energy_hourly a
JOIN energy_hourly b
    ON a.month = b.month
    AND a.hour  = b.hour
    AND a.year  = b.year + 1
GROUP BY a.year, a.month, a.hour
ORDER BY a.year, a.month, a.hour;
