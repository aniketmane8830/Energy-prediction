from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

# ============================================================
# PJMW HOURLY ENERGY DEMAND FORECAST APP
# Uses the trained XGBoost model and bundled PJMW historical data.
# Supports forecasts from 1 day up to 31 days.
# ============================================================

st.set_page_config(
    page_title="PJMW Energy Demand Forecast",
    page_icon=None,
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_model.json"
DATA_PATH = BASE_DIR / "PJMW_MW_Hourly.xlsx"

FEATURES = [
    "Hour",
    "Day_of_Week",
    "Month",
    "Quarter",
    "Is_Weekend",
    "Lag_1",
    "Lag_24",
    "Lag_168",
    "Rolling_Mean_24",
    "Rolling_Std_24",
    "Rolling_Mean_168",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "xgboost_model.json was not found next to app.py."
        )

    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    # Confirm this is a fitted XGBoost model
    model.get_booster()

    return model


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_history():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "PJMW_MW_Hourly.xlsx was not found next to app.py."
        )

    data = pd.read_excel(DATA_PATH)

    data.columns = [str(c).strip() for c in data.columns]

    required = {"Datetime", "PJMW_MW"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    data["Datetime"] = pd.to_datetime(
        data["Datetime"],
        errors="coerce"
    )

    data["PJMW_MW"] = pd.to_numeric(
        data["PJMW_MW"],
        errors="coerce"
    )

    data = (
        data.dropna(subset=["Datetime", "PJMW_MW"])
        .sort_values("Datetime")
        .drop_duplicates("Datetime")
        .reset_index(drop=True)
    )

    return data[["Datetime", "PJMW_MW"]]


# ============================================================
# BUILD FEATURES
# ============================================================

def build_feature_row(timestamp, history):
    """
    Create exactly the 11 features used during XGBoost training.
    """

    values = history.to_numpy(dtype=float)

    if len(values) < 168:
        raise ValueError(
            "At least 168 historical hourly values are required."
        )

    row = {
        "Hour": timestamp.hour,
        "Day_of_Week": timestamp.dayofweek,
        "Month": timestamp.month,
        "Quarter": timestamp.quarter,
        "Is_Weekend": int(timestamp.dayofweek >= 5),

        "Lag_1": values[-1],
        "Lag_24": values[-24],
        "Lag_168": values[-168],

        "Rolling_Mean_24": values[-24:].mean(),

        "Rolling_Std_24": (
            values[-24:].std(ddof=1)
            if len(values[-24:]) > 1
            else 0
        ),

        "Rolling_Mean_168": values[-168:].mean(),
    }

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


# ============================================================
# FORECAST FUNCTION
# ============================================================

def forecast_next_hours(
    model,
    history,
    last_timestamp,
    hours
):
    """
    Recursive forecast.

    Each predicted value is added to the history and becomes
    available for the next prediction's lag and rolling features.
    """

    values = history.copy().reset_index(drop=True)

    results = []

    for step in range(1, hours + 1):

        timestamp = (
            last_timestamp
            + pd.Timedelta(hours=step)
        )

        X = build_feature_row(
            timestamp,
            values
        )

        prediction = float(
            model.predict(X)[0]
        )

        results.append(
            {
                "Datetime": timestamp,
                "Predicted_MW": prediction,
                "Predicted_GW": prediction / 1000,
            }
        )

        # Add prediction to history
        values.loc[len(values)] = prediction

    return pd.DataFrame(results)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("PJMW Hourly Energy Demand Forecast")

st.caption(
    "PJM West electricity demand forecasting using "
    "the trained XGBoost model."
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:
    model = load_model()
    history_df = load_history()

except Exception as error:
    st.error(
        f"Could not start the application: {error}"
    )
    st.stop()


if len(history_df) < 168:
    st.error(
        "The historical dataset must contain at least "
        "168 hourly observations."
    )
    st.stop()


history = history_df["PJMW_MW"].astype(float)

last_timestamp = history_df["Datetime"].iloc[-1]


st.success(
    "Trained XGBoost model loaded successfully. "
    "You can forecast up to 31 days ahead."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Forecast Settings")

    forecast_days = st.slider(
        "Forecast horizon (days)",
        min_value=1,
        max_value=31,
        value=7,
        step=1,
    )

    horizon = forecast_days * 24

    st.info(
        f"The model will forecast the next "
        f"{forecast_days} day(s) "
        f"({horizon} hours)."
    )

    st.markdown("---")

    st.write("**Model:** XGBoost Regressor")
    st.write("**Features:** 11")

    st.write(
        "**History available:** "
        f"{len(history_df):,} rows"
    )

    st.write(
        "**Forecast horizon:** "
        f"{horizon:,} hours"
    )


# ============================================================
# CURRENT DATA SUMMARY
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Historical Rows",
    f"{len(history_df):,}"
)

c2.metric(
    "Last Timestamp",
    last_timestamp.strftime(
        "%Y-%m-%d %H:%M"
    )
)

c3.metric(
    "Last Demand",
    f"{history.iloc[-1]:,.0f} MW"
)


st.markdown("---")


# ============================================================
# FORECAST BUTTON
# ============================================================

if st.button(
    "Generate Forecast",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        f"Generating {forecast_days}-day forecast..."
    ):

        try:

            forecast_df = forecast_next_hours(
                model=model,
                history=history,
                last_timestamp=last_timestamp,
                hours=horizon,
            )

            # ==================================================
            # FORECAST SUMMARY
            # ==================================================

            st.subheader(
                f"Next {forecast_days} Day(s) Forecast"
            )

            best_row = forecast_df.loc[
                forecast_df["Predicted_MW"].idxmax()
            ]

            low_row = forecast_df.loc[
                forecast_df["Predicted_MW"].idxmin()
            ]

            average_demand = (
                forecast_df["Predicted_MW"].mean()
            )

            total_energy_gwh = (
                forecast_df["Predicted_MW"].sum()
                / 1000
            )

            # ==================================================
            # SUMMARY METRICS
            # ==================================================

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "First Forecast",
                f"{forecast_df.iloc[0]['Predicted_MW']:,.0f} MW"
            )

            m2.metric(
                "Maximum Forecast",
                f"{best_row['Predicted_MW']:,.0f} MW"
            )

            m3.metric(
                "Minimum Forecast",
                f"{low_row['Predicted_MW']:,.0f} MW"
            )

            m4.metric(
                "Average Demand",
                f"{average_demand:,.0f} MW"
            )

            # ==================================================
            # PEAK / LOW INFORMATION
            # ==================================================

            st.markdown("### Forecast Insights")

            i1, i2, i3 = st.columns(3)

            i1.info(
                "Peak Demand\n\n"
                f"{best_row['Predicted_MW']:,.0f} MW\n\n"
                f"{best_row['Datetime'].strftime('%Y-%m-%d %H:%M')}"
            )

            i2.info(
                "**Lowest Demand**\n\n"
                f"{low_row['Predicted_MW']:,.0f} MW\n\n"
                f"{low_row['Datetime'].strftime('%Y-%m-%d %H:%M')}"
            )

            i3.info(
                "Estimated Energy\n\n"
                f"{total_energy_gwh:,.2f} GWh\n\n"
                "Over the selected forecast period"
            )

            # ==================================================
            # HOURLY FORECAST CHART
            # ==================================================

            st.markdown("### Hourly Demand Forecast")

            chart_data = (
                forecast_df
                .set_index("Datetime")
                [["Predicted_MW"]]
            )

            st.line_chart(
                chart_data,
                use_container_width=True
            )

            # ==================================================
            # DAILY SUMMARY
            # ==================================================

            st.markdown("### Daily Forecast Summary")

            daily_df = forecast_df.copy()

            daily_df["Date"] = (
                daily_df["Datetime"]
                .dt.date
            )

            daily_summary = (
                daily_df
                .groupby("Date")
                .agg(
                    Average_MW=(
                        "Predicted_MW",
                        "mean"
                    ),
                    Maximum_MW=(
                        "Predicted_MW",
                        "max"
                    ),
                    Minimum_MW=(
                        "Predicted_MW",
                        "min"
                    ),
                    Total_GWh=(
                        "Predicted_MW",
                        lambda x: x.sum() / 1000
                    ),
                )
                .reset_index()
            )

            daily_summary["Average_MW"] = (
                daily_summary["Average_MW"]
                .round(2)
            )

            daily_summary["Maximum_MW"] = (
                daily_summary["Maximum_MW"]
                .round(2)
            )

            daily_summary["Minimum_MW"] = (
                daily_summary["Minimum_MW"]
                .round(2)
            )

            daily_summary["Total_GWh"] = (
                daily_summary["Total_GWh"]
                .round(3)
            )

            st.dataframe(
                daily_summary,
                use_container_width=True,
                hide_index=True,
            )

            # ==================================================
            # DAILY AVERAGE CHART
            # ==================================================

            st.markdown("### Daily Average Demand")

            daily_chart = (
                daily_summary
                .set_index("Date")
                [["Average_MW"]]
            )

            st.line_chart(
                daily_chart,
                use_container_width=True
            )

            # ==================================================
            # HOURLY FORECAST TABLE
            # ==================================================

            st.markdown("### Hourly Forecast Data")

            display_df = forecast_df.copy()

            display_df["Datetime"] = (
                display_df["Datetime"]
                .dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
            )

            display_df["Predicted_MW"] = (
                display_df["Predicted_MW"]
                .round(2)
            )

            display_df["Predicted_GW"] = (
                display_df["Predicted_GW"]
                .round(3)
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            # ==================================================
            # DOWNLOAD HOURLY FORECAST
            # ==================================================

            csv = forecast_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Hourly Forecast CSV",
                data=csv,
                file_name=(
                    f"pjmw_{forecast_days}_day_forecast.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

            # ==================================================
            # DOWNLOAD DAILY SUMMARY
            # ==================================================

            daily_csv = daily_summary.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Daily Summary CSV",
                data=daily_csv,
                file_name=(
                    f"pjmw_{forecast_days}_day_daily_summary.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                f"Forecast failed: {error}"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        "**Model:** Tuned XGBoost Regressor"
    )

    st.write(
        "**Training rounds:** 400"
    )

    st.write(
        "**Prediction target:** PJMW_MW (MW)"
    )

    st.write(
        "**Input upload required:** No"
    )

    st.write(
        "**Historical data:** "
        "Bundled PJMW_MW_Hourly.xlsx"
    )

    st.write(
        "**Forecast method:** "
        "Recursive hourly forecasting"
    )

    st.markdown(
        "### Features used by the model"
    )

    feature_table = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Description": [
                "Hour of day",
                "Day of week",
                "Month",
                "Quarter",
                "Weekend indicator",
                "Demand 1 hour earlier",
                "Demand 24 hours earlier",
                "Demand 168 hours earlier",
                "24-hour rolling average",
                "24-hour rolling standard deviation",
                "168-hour rolling average",
            ],
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "PJMW Energy Demand Forecasting - XGBoost - "
    "Hourly Recursive Forecast"
)