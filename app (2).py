
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import urllib.request
import os

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="NASA C-MAPSS Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("🔧 NASA C-MAPSS Predictive Maintenance")
st.markdown(
    "### 🚀 AI-Based Engine Health Monitoring Dashboard"
)

st.divider()

# ------------------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "PunVas/nasa-c-mapss/main/train_FD001.txt"
)

DATA_FILE = "train_FD001.txt"

if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

# ------------------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------------------

columns = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3"
]

for i in range(1, 22):
    columns.append(f"sensor_{i}")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    sep=r"\s+",
    header=None
)

df = df.iloc[:, :26]
df.columns = columns

# ------------------------------------------------------------
# CREATE RUL
# ------------------------------------------------------------

max_cycle = (
    df.groupby("engine_id")["cycle"]
    .transform("max")
)

df["RUL"] = max_cycle - df["cycle"]

# ------------------------------------------------------------
# SELECT USEFUL SENSORS
# ------------------------------------------------------------

features = []

for i in range(1, 22):

    sensor = f"sensor_{i}"

    if df[sensor].nunique() > 1:
        features.append(sensor)

# ------------------------------------------------------------
# PREPARE MODEL
# ------------------------------------------------------------

X = df[features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(X.median())

y = df["RUL"]

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

@st.cache_resource
def train_model():

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model

rul_model = train_model()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("⚙️ Engine Selection")

engine_id = st.sidebar.selectbox(
    "Select Engine",
    sorted(df["engine_id"].unique())
)

# ------------------------------------------------------------
# ENGINE DATA
# ------------------------------------------------------------

engine = df[
    df["engine_id"] == engine_id
].copy()

engine = engine.sort_values("cycle")

latest = engine.iloc[-1]

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

latest_values = []

for sensor in features:

    value = latest[sensor]

    if pd.isna(value):
        value = df[sensor].median()

    latest_values.append(float(value))

X_latest = np.array(
    latest_values
).reshape(1, -1)

predicted_rul = float(
    rul_model.predict(X_latest)[0]
)

predicted_rul = max(
    0,
    predicted_rul
)

# ------------------------------------------------------------
# STATUS
# ------------------------------------------------------------

if predicted_rul > 60:

    status = "🟢 HEALTHY"
    risk = "LOW"
    recommendation = "Continue normal monitoring."

elif predicted_rul > 30:

    status = "🟡 WARNING"
    risk = "MEDIUM"
    recommendation = "Schedule preventive inspection."

else:

    status = "🔴 CRITICAL"
    risk = "HIGH"
    recommendation = "Maintenance required soon."

# ------------------------------------------------------------
# TOP METRICS
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Engine ID",
        engine_id
    )

with col2:
    st.metric(
        "Current Cycle",
        int(latest["cycle"])
    )

with col3:
    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.1f}"
    )

with col4:
    st.metric(
        "Failure Risk",
        risk
    )

st.divider()

# ------------------------------------------------------------
# MACHINE STATUS
# ------------------------------------------------------------

st.subheader("🏭 Engine Health Overview")

st.success(
    f"Machine Status: {status}"
)

st.info(
    f"Maintenance Recommendation: {recommendation}"
)

# ------------------------------------------------------------
# RUL GAUGE
# ------------------------------------------------------------

st.subheader("⏳ Remaining Useful Life")

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=min(predicted_rul, 125),
        title={
            "text": "Predicted RUL (Cycles)"
        },
        gauge={
            "axis": {
                "range": [0, 125]
            }
        }
    )
)

gauge.update_layout(
    height=350
)

st.plotly_chart(
    gauge,
    use_container_width=True
)

# ------------------------------------------------------------
# SENSOR GRAPH
# ------------------------------------------------------------

st.subheader("📈 Sensor Monitoring")

sensor_fig = go.Figure()

for sensor in features[:5]:

    sensor_fig.add_trace(
        go.Scatter(
            x=engine["cycle"],
            y=engine[sensor],
            mode="lines",
            name=sensor
        )
    )

sensor_fig.update_layout(
    xaxis_title="Cycle",
    yaxis_title="Sensor Value",
    height=450
)

st.plotly_chart(
    sensor_fig,
    use_container_width=True
)

# ------------------------------------------------------------
# RUL CURVE
# ------------------------------------------------------------

st.subheader("📉 RUL Degradation Curve")

rul_fig = go.Figure()

rul_fig.add_trace(
    go.Scatter(
        x=engine["cycle"],
        y=engine["RUL"],
        mode="lines",
        name="Actual RUL"
    )
)

rul_fig.update_layout(
    xaxis_title="Cycle",
    yaxis_title="RUL",
    height=450
)

st.plotly_chart(
    rul_fig,
    use_container_width=True
)

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

st.subheader("⭐ Feature Importance")

importance_df = pd.DataFrame({
    "Sensor": features,
    "Importance": rul_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(10)
)

importance_fig = go.Figure(
    go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Sensor"],
        orientation="h"
    )
)

importance_fig.update_layout(
    yaxis=dict(
        categoryorder="total ascending"
    ),
    height=450
)

st.plotly_chart(
    importance_fig,
    use_container_width=True
)

# ------------------------------------------------------------
# ENGINE DATA
# ------------------------------------------------------------

st.subheader("📋 Latest Engine Data")

display_columns = [
    "engine_id",
    "cycle"
] + features[:10]

st.dataframe(
    engine[display_columns].tail(10),
    use_container_width=True
)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "NASA C-MAPSS FD001 | Predictive Maintenance AI Project"
)

print("✅ app.py created successfully!")
