from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import shap
import lime
import lime.lime_tabular

from ev_fuzzy_system import calculate_charging_comfort
from rl_time_agent import recommend_charging_action
from clustering_core import (
    train_fcm_model,
    build_cluster_summary,
    predict_memberships,
    get_clustering_features
)

app = Flask(__name__)

# =====================================================

def pad_numeric_features(num_data, scaler):
    expected = scaler.n_features_in_
    if len(num_data) < expected:
        num_data += [0.0] * (expected - len(num_data))
    return num_data


def pad_categorical_features(cat_data, encoder):
    expected = encoder.n_features_in_
    current = len(cat_data)
    if current < expected:
        cat_data += [
            encoder.categories_[i][0]
            for i in range(current, expected)
        ]
    return cat_data


# =====================================================
# LOAD XAI EXPLAINERS
# =====================================================
shap_explainer = pickle.load(open("explainers/shap_explainer.pkl", "rb"))
lime_explainer = pickle.load(open("explainers/lime_explainer.pkl", "rb"))

# =====================================================
# LOAD MODELS
# =====================================================
cost_model = pickle.load(open("saved_models/cost_model_rf.pkl", "rb"))
time_model = pickle.load(open("saved_models/time_model_lr.pkl", "rb"))
class_model = pickle.load(open("saved_models/long_session_model_gb.pkl", "rb"))

X_cost_train = pd.read_pickle("data/X_train_cost.pkl")
X_time_train = pd.read_pickle("data/X_train_time.pkl")
X_class_train = pd.read_pickle("data/X_train_classification.pkl")

cost_features = list(X_cost_train.columns)
time_features = list(X_time_train.columns)
class_features = list(X_class_train.columns)

ALL_FEATURES = sorted(set(cost_features + time_features + class_features))

# =====================================================
# CATEGORICAL MAPS
# =====================================================
CAT_MAPS = {
    "Time of Day": {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3},
    "Day of Week": {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    },
    "Charger Type": {"Level 1": 0, "Level 2": 1, "DC Fast Charger": 2},
    "Vehicle Model": {
        "BMW i3": 0, "Hyundai Kona": 1, "Chevy Bolt": 2,
        "Nissan Leaf": 3, "Tesla Model 3": 4
    },
    "Charging Station Location": {
        "Chicago": 0, "Houston": 1, "Los Angeles": 2,
        "New York": 3, "San Francisco": 4
    },
    "User Type": {"Casual Driver": 0, "Commuter": 1, "Long-Distance Traveler": 2},
}

# =====================================================
# ANOMALY DETECTION (LAB-CORRECT)
# =====================================================
anomaly_model = pickle.load(open("saved_models/best_anomaly_model.pkl", "rb"))
scaler = pickle.load(open("data/scaler.pkl", "rb"))
ohe = pickle.load(open("data/onehot_encoder.pkl", "rb"))

num_cols_raw = pickle.load(open("data/num_cols.pkl", "rb"))
cat_cols_raw = pickle.load(open("data/cat_cols.pkl", "rb"))

FORBIDDEN_COLS = {
    "Long Session",
    "Charging Cost (USD)",
    "Charging Time Difference (minutes)"
}

num_cols = [c for c in num_cols_raw if c not in FORBIDDEN_COLS]
cat_cols = [c for c in cat_cols_raw if c not in FORBIDDEN_COLS]

ANOMALY_CAT_OPTIONS = {
    col: list(CAT_MAPS[col].keys())
    for col in cat_cols if col in CAT_MAPS
}

# =====================================================
# CLUSTERING (FCM)
# =====================================================
CLUSTERING_ERROR = None
CLUSTER_MODEL = None
CLUSTER_SUMMARY = None
CLUSTER_FEATURES = get_clustering_features()

try:
    CLUSTER_MODEL = train_fcm_model(data_dir="data", n_clusters=3)
    CLUSTER_SUMMARY = build_cluster_summary(CLUSTER_MODEL)
except Exception as e:
    CLUSTERING_ERROR = str(e)

# =====================================================
# ROUTES – MENUS
# =====================================================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/user")
def user_menu():
    return render_template("user_menu.html")


@app.route("/admin")
def admin_menu():
    return render_template("admin_menu.html")


# =====================================================
# USER PREDICTOR (COST + SHAP + LIME)
# =====================================================
@app.route("/predictor", methods=["GET", "POST"])
def predictor():

    prediction = None
    shap_explanation = []
    lime_explanation = []

    # 🔑 detect correct classification key (LAB SAFE)
    CLASS_KEY = [k for k in shap_explainer.keys() if k not in ["cost", "time"]][0]

    if request.method == "POST":
        action = request.form.get("action")

        # ===============================
        # BUILD USER INPUT
        # ===============================
        user_input = {}
        for f in ALL_FEATURES:
            if f in CAT_MAPS:
                user_input[f] = CAT_MAPS[f][request.form[f]]
            else:
                user_input[f] = float(request.form[f])

        X_all = pd.DataFrame([user_input])

        # ===============================
        # COST PREDICTION + XAI
        # ===============================
        if action == "cost":
            X_used = X_all[cost_features]
            pred = cost_model.predict(X_used)[0]
            prediction = f"Predicted Charging Cost: {round(pred, 2)} USD"

            shap_vals = shap_explainer["cost"].shap_values(X_used)[0]
            shap_explanation = [
                f"{f} {'increased' if v > 0 else 'decreased'} cost"
                for f, v in zip(cost_features, shap_vals)
            ]

            lime_exp = lime_explainer["cost"].explain_instance(
                X_used.iloc[0].values,
                cost_model.predict,
                num_features=5
            )
            lime_explanation = [rule for rule, _ in lime_exp.as_list()]

        # ===============================
        # TIME PREDICTION + XAI
        # ===============================
        elif action == "time":
            X_used = X_all[time_features]
            pred = time_model.predict(X_used)[0]
            prediction = f"Predicted Charging Time: {round(pred, 2)} minutes"

            shap_vals = shap_explainer["time"].shap_values(X_used)[0]
            shap_explanation = [
                f"{f} {'increased' if v > 0 else 'decreased'} time"
                for f, v in zip(time_features, shap_vals)
            ]

            lime_exp = lime_explainer["time"].explain_instance(
                X_used.iloc[0].values,
                time_model.predict,
                num_features=5
            )
            lime_explanation = [rule for rule, _ in lime_exp.as_list()]

        # ===============================
        # CLASSIFICATION + XAI
        # ===============================
        elif action == "class":
            X_used = X_all[class_features]
            prob = class_model.predict_proba(X_used)[0][1]
            prediction = "Long Session" if prob >= 0.5 else "Normal Session"

            shap_raw = shap_explainer[CLASS_KEY].shap_values(X_used)
            shap_vals = shap_raw[1][0] if isinstance(shap_raw, list) else shap_raw[0]

            shap_explanation = [
                f"{f} {'increased' if v > 0 else 'decreased'} long-session likelihood"
                for f, v in zip(class_features, shap_vals)
            ]

            lime_exp = lime_explainer[CLASS_KEY].explain_instance(
                X_used.iloc[0].values,
                class_model.predict_proba,
                num_features=5
            )
            lime_explanation = [rule for rule, _ in lime_exp.as_list()]

    return render_template(
        "predictor.html",
        ALL_FEATURES=ALL_FEATURES,
        CAT_MAPS=CAT_MAPS,
        prediction=prediction,
        shap_explanation=shap_explanation,
        lime_explanation=lime_explanation
    )



@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():

    if request.method == "POST":

        user_input = {}
        for f in ALL_FEATURES:
            if f in CAT_MAPS:
                user_input[f] = CAT_MAPS[f][request.form[f]]
            else:
                user_input[f] = float(request.form[f])

        X_user = pd.DataFrame([user_input])

        predicted_cost = cost_model.predict(X_user[cost_features])[0]
        predicted_time = time_model.predict(X_user[time_features])[0]

        urgency = float(request.form["urgency"])
        budget = float(request.form["budget"])

        comfort, strategy, advice = calculate_charging_comfort(
            predicted_cost,
            predicted_time,
            urgency,
            budget
        )

        return render_template(
            "recommendations.html",
            ALL_FEATURES=ALL_FEATURES,
            CAT_MAPS=CAT_MAPS,
            predicted_cost=round(predicted_cost, 2),
            predicted_time=round(predicted_time, 2),
            comfort_score=round(comfort, 2),
            strategy=strategy,
            advice=advice
        )

    return render_template(
        "recommendations.html",
        ALL_FEATURES=ALL_FEATURES,
        CAT_MAPS=CAT_MAPS
    )


@app.route("/admin/anomaly", methods=["GET", "POST"])
def admin_anomaly():

    result = None

    if request.method == "POST":
        try:
            # Numeric
            num_data = [float(request.form[c]) for c in num_cols]
            num_data = pad_numeric_features(num_data, scaler)
            X_num = scaler.transform([num_data])

            # Categorical
            cat_data = [request.form[c] for c in cat_cols]
            cat_data = pad_categorical_features(cat_data, ohe)
            X_cat = ohe.transform([cat_data])

            # Final vector
            X = np.hstack([X_num, X_cat])

            score = -anomaly_model.decision_function(X)[0]
            label = anomaly_model.predict(X)[0]

            result = {
                "status": "Anomalous Session" if label == -1 else "Normal Session",
                "score": round(float(score), 4)
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template(
        "admin_anomaly.html",
        num_cols=num_cols,
        cat_options=ANOMALY_CAT_OPTIONS,
        result=result
    )



@app.route("/admin/clustering", methods=["GET", "POST"])
def admin_clustering():

    if CLUSTERING_ERROR:
        return render_template(
            "clustering.html",
            error=CLUSTERING_ERROR,
            features=CLUSTER_FEATURES,
            summary_table=[],
            result=None
        )

    result = None

    if request.method == "POST":
        user_values = {f: float(request.form[f]) for f in CLUSTER_FEATURES}
        assigned_cluster, memberships = predict_memberships(CLUSTER_MODEL, user_values)
        result = {
            "assigned_cluster": assigned_cluster,
            "memberships": memberships
        }

    return render_template(
        "clustering.html",
        error=None,
        features=CLUSTER_FEATURES,
        summary_table=CLUSTER_SUMMARY,
        result=result
    )

@app.route("/user/rl-time", methods=["GET", "POST"])
def rl_time_recommendation():

    recommendation = None

    if request.method == "POST":
        try:
            recommendation = recommend_charging_action(
                soc_start=float(request.form["soc_start"]),
                energy_consumed=float(request.form["energy_consumed"]),
                battery_capacity=float(request.form["battery_capacity"]),
                temperature=float(request.form["temperature"]),
                charger_type=request.form["charger_type"],
                user_type=request.form["user_type"]
            )
        except Exception as e:
            recommendation = f"Error: {str(e)}"

    return render_template(
        "rl_time.html",
        recommendation=recommendation
    )




@app.route("/admin/classification", methods=["GET", "POST"])
def admin_classification():

    prediction = None
    probability = None

    if request.method == "POST":
        user_input = {}
        for f in class_features:
            if f in CAT_MAPS:
                user_input[f] = CAT_MAPS[f][request.form[f]]
            else:
                user_input[f] = float(request.form[f])

        X_admin = pd.DataFrame([user_input])
        prob = class_model.predict_proba(X_admin)[0][1]
        prediction = "Long Session" if prob >= 0.5 else "Normal Session"
        probability = round(prob, 3)

    return render_template(
        "admin_classification.html",
        class_features=class_features,
        CAT_MAPS=CAT_MAPS,
        prediction=prediction,
        probability=probability
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
