# PJMW Hourly Energy Demand Forecasting

A Streamlit-based machine learning application for forecasting PJM West
(PJMW) hourly electricity demand using a tuned XGBoost regression model.

## Project Overview

This project uses historical PJMW electricity demand data to train an
XGBoost regression model and generate recursive hourly forecasts.

The Streamlit application includes:

-   Historical PJMW demand data bundled with the project
-   An XGBoost regression model saved in XGBoost's native JSON format
-   Forecasting from 1 day up to 31 days ahead
-   Recursive hourly forecasting
-   Forecast summary metrics
-   Hourly demand visualization
-   Daily forecast summary
-   Daily average demand visualization
-   Hourly and daily CSV downloads
-   Model and feature information

## Project Structure

``` text
PROJECT DEPLOYMENT FINAL/
│
├── app.py
├── xgboost_model.json
├── PJMW_MW_Hourly.xlsx
├── PJMW_model.ipynb
├── requirements.txt
└── README.md
```

### Files

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit application used for
                                      forecasting

  `xgboost_model.json`                Trained XGBoost model in native
                                      JSON format

  `PJMW_MW_Hourly.xlsx`               Historical PJMW electricity demand
                                      data

  `PJMW_model.ipynb`                  Notebook containing data
                                      preparation, model training,
                                      evaluation, and model saving

  `requirements.txt`                  Python dependencies required to run
                                      the project

  `README.md`                         Project documentation
  -----------------------------------------------------------------------

## Machine Learning Model

The project uses a tuned `XGBRegressor`.

### Model Parameters

``` text
n_estimators = 400
max_depth = 5
learning_rate = 0.1
subsample = 0.7
colsample_bytree = 1.0
reg_alpha = 1
reg_lambda = 4
min_child_weight = 15
```

### Model Evaluation

The trained model achieved the following evaluation results in the
project notebook:

  Metric       Result
  ---------- --------
  MAE           62.10
  RMSE          81.02
  MAPE          1.10%
  R² Score      0.992

These values are based on the evaluation shown in the project notebook.

## Features

The model uses 11 features:

  Feature              Description
  -------------------- ------------------------------------
  `Hour`               Hour of the day
  `Day_of_Week`        Day of the week
  `Month`              Month
  `Quarter`            Quarter of the year
  `Is_Weekend`         Weekend indicator
  `Lag_1`              Demand one hour earlier
  `Lag_24`             Demand 24 hours earlier
  `Lag_168`            Demand 168 hours earlier
  `Rolling_Mean_24`    24-hour rolling average
  `Rolling_Std_24`     24-hour rolling standard deviation
  `Rolling_Mean_168`   168-hour rolling average

## Forecasting Method

The application uses recursive hourly forecasting.

For the first forecast hour, the model uses historical demand values to
create the required lag and rolling features.

After the first prediction is generated, that predicted value is added
to the demand history. The updated history is then used to generate the
features for the next hour.

The process continues until the selected forecast horizon is reached.

``` text
Historical Demand
       |
       v
Feature Engineering
       |
       v
XGBoost Model
       |
       v
Next Hour Prediction
       |
       v
Add Prediction to History
       |
       v
Create Features for Next Hour
       |
       v
Repeat Until Forecast Horizon
```

## Forecast Horizon

The application allows users to select:

-   Minimum: 1 day
-   Maximum: 31 days
-   Default: 7 days

Each day represents 24 forecast hours.

Therefore:

``` text
1 day  = 24 hours
7 days = 168 hours
31 days = 744 hours
```

## Installation

### 1. Clone or download the project

Place all project files in the same directory.

### 2. Create a virtual environment

It is recommended to use a dedicated Python environment for the project.

For example:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

## Running the Application

From the project directory, run:

``` bash
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

``` text
http://localhost:8501
```

Open that address in a browser to use the application.

## Using the Application

1.  Start the Streamlit application.
2.  Select the forecast horizon from the sidebar.
3.  Choose between 1 and 31 days.
4.  Click `Generate Forecast`.
5.  Review the forecast metrics.
6.  View the hourly forecast chart.
7.  Review the daily forecast summary.
8.  Download the hourly forecast CSV if required.
9.  Download the daily summary CSV if required.

## Model Storage

The application uses:

``` text
xgboost_model.json
```

instead of a Python pickle file.

The model is loaded using XGBoost's native model-loading functionality:

``` python
model = XGBRegressor()
model.load_model("xgboost_model.json")
```

This approach is specifically suited to storing an XGBoost model and
avoids relying on Python pickle deserialization for the trained model.

## Data Requirements

The application expects the bundled Excel file:

``` text
PJMW_MW_Hourly.xlsx
```

The dataset must contain:

``` text
Datetime
PJMW_MW
```

The application converts `Datetime` to a datetime type and `PJMW_MW` to
numeric values before forecasting.

At least 168 historical hourly observations are required because the
model uses a 168-hour lag and rolling features.

## Important Deployment Notes

Keep the following files in the same directory as `app.py`:

``` text
app.py
xgboost_model.json
PJMW_MW_Hourly.xlsx
```

If the model file is missing, the application will display an error.

If the historical Excel file is missing, the application will also
display an error.

The application does not require users to upload the historical Excel
file every time. The data is bundled with the project.

## Technologies Used

-   Python
-   Pandas
-   XGBoost
-   Scikit-learn
-   Streamlit
-   OpenPyXL
-   Jupyter Notebook

## Future Improvements

Possible future improvements include:

-   Adding confidence intervals to forecasts
-   Comparing XGBoost with other forecasting models
-   Adding weather and calendar variables
-   Adding automated model retraining
-   Deploying the application to a cloud platform
-   Adding historical-versus-forecast comparison charts
-   Adding model monitoring and performance tracking

## Author

PJMW Energy Demand Forecasting Project

## License

Add the appropriate project license here if the project is published
publicly.
