# Taxi Fare Prediction Using Random Forest Regressor

A supervised machine-learning regression project that predicts taxi fare
amounts from pickup and drop-off locations, trip distance, passenger
count, and pickup-time features. The project covers data inspection,
cleaning, exploratory data analysis (EDA), feature engineering, model
training, hyperparameter tuning, evaluation, and export of predictions.

------------------------------------------------------------------------

## 1. Project Overview

**Goal:** Estimate the taxi fare (`amount`) for a trip using information
available at pickup time and the trip's pickup/drop-off coordinates.

**Problem type:** Supervised learning --- Regression\
**Main algorithm:** `RandomForestRegressor`\
**Libraries:** Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn,
joblib\
**Dataset:** `TaxiFare.csv` --- 50,000 original records and 8 columns

The final cleaned dataset contains **48,731 records**. The model uses
**12 numerical or binary features** and predicts a continuous fare
value.

## 2. Repository Contents

A suggested project folder layout is:

``` text
Taxi_Fare_Prediction/
│
├── TaxiFare.csv
├── taxi_fare_prediction.ipynb
├── Taxi_Fare_Prediction_Report.pdf
├── taxi_fare_predictions.csv
├── taxi_fare_random_forest.pkl
├── requirements.txt
└── README.md
```

-   `TaxiFare.csv` --- original dataset used for the analysis.
-   `taxi_fare_prediction.ipynb` --- notebook containing the full
    workflow, code, outputs, and visualizations.
-   `Taxi_Fare_Prediction_Report.pdf` --- concise project report with
    methodology, results, charts, and limitations.
-   `taxi_fare_predictions.csv` --- test-set actual fares, model
    predictions, and absolute errors.
-   `taxi_fare_random_forest.pkl` --- serialized fitted
    preprocessing/model pipeline.
-   `requirements.txt` --- Python dependencies.
-   `README.md` --- project documentation.

If the model file or dataset is not included in a particular submission,
remove it from the folder listing rather than leaving a misleading
placeholder.

## 3. Dataset Description

The original dataset has **50,000 rows and 8 columns**:

  Column                   Description
  ------------------------ ------------------------------
  `unique_id`              Record identifier
  `date_time_of_pickup`    Date and time of pickup
  `longitude_of_pickup`    Pickup longitude
  `latitude_of_pickup`     Pickup latitude
  `longitude_of_dropoff`   Drop-off longitude
  `latitude_of_dropoff`    Drop-off latitude
  `no_of_passenger`        Number of passengers
  `amount`                 Taxi fare; prediction target

The original inspection found no missing values and no duplicate rows.
The pickup timestamps were converted to datetime values. The observed
date range was January 2009 through June 2015.

**Target variable:** `amount`

## 4. Data Cleaning and Preprocessing

The notebook applies the following checks and cleaning steps:

1.  Inspected dataset shape, columns, data types, missing values,
    duplicates, and descriptive statistics.
2.  Converted `date_time_of_pickup` to datetime.
3.  Excluded `unique_id` from the model because it is an identifier, not
    a meaningful trip predictor.
4.  Removed records with invalid or geographically implausible
    coordinates.
5.  Removed non-positive fares and records with zero passengers.
6.  Investigated unusually high fares using fare amount, trip distance,
    passenger count, and coordinate plausibility. Selected suspicious
    records were removed; plausible high fares were retained.
7.  Applied approximate New York City coordinate bounds:
    -   Longitude: `-74.3` to `-73.6`
    -   Latitude: `40.4` to `41.1`
8.  Kept plausible zero-distance records because zero distance alone
    does not prove that a trip is invalid.
9.  Calculated `fare_per_km` temporarily for diagnostics, then removed
    it before modeling because it is undefined for zero-distance trips.

**Final cleaned data:** 48,731 rows. The final modeling data has no
missing values in the selected features or target.

### Cleaning caveat

The geographic bounds are approximate. They may exclude legitimate trips
outside the selected New York City area. This is a limitation of the
chosen cleaning rule and should be considered when interpreting the
results.

## 5. Exploratory Data Analysis

The notebook explores fare and distance distributions, correlations, and
average fare by pickup hour.

Key observations from the cleaned data:

  Measure                                                  Result
  --------------------------------------- -----------------------
  Average trip distance                                   3.33 km
  Median trip distance                                    2.16 km
  Maximum trip distance                                  43.65 km
  Average fare                              Approximately \$11.30
  Maximum retained fare                                  \$104.67
  Correlation: trip distance and fare         Approximately 0.732
  Correlation: passenger count and fare       Approximately 0.016
  Average weekday fare                      Approximately \$11.34
  Average weekend fare                      Approximately \$11.29

Trip distance has a strong positive relationship with fare in this
dataset. Passenger count has a very weak linear correlation with fare.

The trip-distance distribution is right-skewed: most trips are short,
with fewer long-distance trips forming the right tail.

### Visualizations included

-   Taxi fare distribution
-   Trip distance distribution
-   Correlation heatmap
-   Average fare by pickup hour
-   Actual versus predicted fares
-   Residual analysis
-   Random Forest feature importance

## 6. Feature Engineering

### 6.1 Trip distance using the Haversine formula

The straight-line distance between pickup and drop-off coordinates is
calculated using the Haversine formula, with Earth's radius set to
**6,371 km**.

The resulting feature is:

-   `trip_distance`

This is a geographic straight-line estimate, not the actual road
distance travelled.

### 6.2 Pickup-time features

The pickup timestamp is used to derive:

  Feature            Meaning
  ------------------ ---------------------------------------------------------
  `pickup_hour`      Hour of pickup, 0--23
  `pickup_day`       Day of month
  `pickup_month`     Month
  `pickup_weekday`   Day of week
  `is_weekend`       1 for Saturday/Sunday, otherwise 0
  `is_night`         1 for pickup from 10:00 PM through 4:59 AM, otherwise 0

The original timestamp is not passed directly to the estimator.

## 7. Model Inputs and Target

The model uses these 12 features:

``` python
features = [
    "longitude_of_pickup",
    "latitude_of_pickup",
    "longitude_of_dropoff",
    "latitude_of_dropoff",
    "no_of_passenger",
    "trip_distance",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_weekday",
    "is_weekend",
    "is_night"
]

X = df_model[features]
y = df_model["amount"]
```

All selected inputs are numerical or binary indicator features, so
categorical encoding was not needed for this feature set.

## 8. Train/Test Split and Pipeline

The data is split into training and test sets:

-   Training set: 80% --- 38,984 rows
-   Test set: 20% --- 9,747 rows
-   `random_state=42`

A scikit-learn `Pipeline` combines `StandardScaler` and
`RandomForestRegressor`.

Random Forest does not require feature scaling for its split-based
learning. The scaler was retained as part of the requested workflow and
is fitted inside the pipeline, helping prevent preprocessing leakage
across validation folds.

### Baseline model

``` python
Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
```

## 9. Baseline Model Results

The baseline model was evaluated on the held-out test set.

  Metric       Baseline
  ---------- ----------
  MAE            2.1489
  MSE           16.3723
  RMSE           4.0463
  R² score       0.8196

**Metric interpretation:**

-   **MAE:** Average absolute difference between actual and predicted
    fares, in fare units.
-   **MSE:** Average squared prediction error; larger errors receive
    greater penalty.
-   **RMSE:** Square root of MSE, expressed in fare units.
-   **R²:** Proportion of target variance explained by the model on the
    test set.

## 10. Hyperparameter Tuning

`GridSearchCV` was used to search for a better Random Forest
configuration.

  Parameter             Values tested
  --------------------- ---------------
  `n_estimators`        100, 200
  `max_depth`           10, 15
  `min_samples_split`   2, 5

Configuration:

-   Cross-validation folds: 3
-   Scoring: `neg_mean_absolute_error`
-   Search parallelism: `n_jobs=1`

**Best parameters:**

``` python
{
    "regressor__max_depth": 15,
    "regressor__min_samples_split": 5,
    "regressor__n_estimators": 200
}
```

Best GridSearchCV cross-validation MAE: **approximately 2.1106**.

## 11. Tuned Model Results

The tuned model was evaluated on the same held-out test set.

  Metric       Baseline         Tuned
  ---------- ---------- -------------
  MAE            2.1489    **2.1354**
  MSE           16.3723   **16.2539**
  RMSE           4.0463    **4.0316**
  R² score       0.8196    **0.8209**

The tuned model shows a modest improvement over the baseline on all four
reported test metrics.

A separate five-fold cross-validation evaluation on the training data
produced:

-   Mean CV MAE: **2.0984**
-   Standard deviation: **0.0339**

The five-fold result is a training-data validation estimate; the
held-out test metrics remain the final test-set evaluation.

## 12. Feature Importance

The tuned Random Forest's impurity-based feature importance indicates:

  Feature                    Importance
  ------------------------ ------------
  `trip_distance`                85.41%
  `longitude_of_dropoff`          4.61%
  `longitude_of_pickup`           2.62%
  `latitude_of_dropoff`           2.36%
  `latitude_of_pickup`            1.49%
  `pickup_hour`                   1.09%
  `pickup_day`                    0.84%
  `pickup_month`                  0.68%
  `pickup_weekday`                0.48%
  `no_of_passenger`               0.23%
  `is_night`                      0.13%
  `is_weekend`                    0.06%

Trip distance is the dominant feature in this fitted model. These
importance scores describe how the model used features; they do not
establish causation. Correlated features can also affect impurity-based
importance.

## 13. Prediction and Model Export

The tuned pipeline was saved with joblib:

``` python
import joblib

joblib.dump(best_model, "taxi_fare_random_forest.pkl")
```

The test-set prediction results were exported to CSV:

``` python
final_results = pd.DataFrame({
    "Actual_Fare": y_test.values,
    "Predicted_Fare": y_pred_tuned
})

final_results["Absolute_Error"] = abs(
    final_results["Actual_Fare"] -
    final_results["Predicted_Fare"]
)

final_results.to_csv(
    "taxi_fare_predictions.csv",
    index=False
)
```

The prediction file contains:

-   `Actual_Fare`
-   `Predicted_Fare`
-   `Absolute_Error`

The notebook also reloads the saved model and checks that its
predictions match the predictions from the in-memory fitted model.

## 14. How to Run the Project

### Requirements

Use Python 3.9 or a compatible later version with package versions that
support the notebook.

Install the dependencies:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

Or install from `requirements.txt` if it is included:

``` bash
pip install -r requirements.txt
```

### Run the notebook

1.  Clone or download the project folder.
2.  Place `TaxiFare.csv` in the same directory as the notebook.
3.  Install the dependencies.
4.  Start Jupyter:

``` bash
jupyter notebook
```

5.  Open `taxi_fare_prediction.ipynb`.
6.  Run all cells from top to bottom.

The notebook saves the fitted model and prediction CSV in its current
working directory. Ensure that directory is writable.

## 15. Limitations and Future Improvements

### Current limitations

-   The dataset does not contain a drop-off timestamp, so actual trip
    duration cannot be calculated.
-   The Haversine distance is straight-line distance, not road-network
    distance.
-   Approximate geographic boundaries may remove valid trips outside the
    selected area.
-   The data is historical (2009--2015) and may not represent current
    fare rules, surcharges, or traffic conditions.
-   The model's feature importance is not a causal explanation.
-   Scaling is not required by Random Forest, although it is included in
    the pipeline for this workflow.

### Possible future improvements

-   Validate geographic cleaning rules against a documented service-area
    definition.
-   Add road distance, trip duration, traffic, weather, and fare-rule
    information if reliable data becomes available.
-   Evaluate performance across fare ranges and trip-distance groups.
-   Compare Random Forest with other regression models using the same
    train/test split and evaluation metrics.
-   Monitor performance on newer data before using the model for
    real-world fare estimates.

## 16. Final Summary

This project builds a taxi-fare regression workflow using a Random
Forest Regressor. Starting from 50,000 original records, the data was
cleaned to 48,731 records, geographic trip distance and pickup-time
features were engineered, and a 12-feature model was trained.

After GridSearchCV tuning, the model achieved:

-   **Test MAE:** 2.1354
-   **Test RMSE:** 4.0316
-   **Test R²:** 0.8209

Trip distance was the most influential feature in the fitted model. The
final pipeline and test-set predictions were exported for reuse and
inspection.

------------------------------------------------------------------------

**Project:** Taxi Fare Prediction Using Random Forest Regressor\
**Primary implementation:** `taxi_fare_prediction.ipynb`\
**Main outputs:** `taxi_fare_random_forest.pkl` and
`taxi_fare_predictions.csv`
