import os
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler


import os

def find_dataset_path(data_dir="data"):
    return os.path.join(data_dir, "ev_charging_dataset_Final(in).csv")




def get_clustering_features():
    return [
        "Energy Consumed (kWh)",
        "Charging Rate (kW)",
        "Charging Time Difference (minutes)",
        "Charging Cost (USD)"
    ]



def train_fcm_model(
    data_dir="data",
    n_clusters=3,
    m=2.0,
    error=0.005,
    maxiter=1000,
    seed=42
):
    path = find_dataset_path(data_dir)
    df = pd.read_csv(path)


    features = get_clustering_features()

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing clustering columns in dataset: {missing}")

    df_clean = df[features].dropna().copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_clean.values)  # shape: (N, F)

    data = X_scaled.T

    np.random.seed(seed)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None
    )

    labels = np.argmax(u, axis=0)  
    df_clean["cluster"] = labels

    model = {
        "features": features,
        "scaler": scaler,
        "centers": cntr,   
        "u": u,            
        "fpc": fpc,
        "df_clean": df_clean  
    }
    return model



def build_cluster_summary(model):
    df_clean = model["df_clean"]
    features = model["features"]

    summary = (
        df_clean.groupby("cluster")[features]
        .mean()
        .reset_index()
        .sort_values("cluster")
    )
    return summary.to_dict(orient="records")



def predict_memberships(model, user_values):
    """
    user_values: dict {feature_name: float}
    returns:
      assigned_cluster (int)
      memberships_list [(cluster_id, degree), ...]
    """
    features = model["features"]
    scaler = model["scaler"]
    centers = model["centers"]

    x = np.array([[float(user_values[f]) for f in features]])  # (1, F)
    x_scaled = scaler.transform(x)  # (1, F)

    x_data = x_scaled.T  # (F, 1)

    u_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        test_data=x_data,
        cntr_trained=centers,
        m=2.0,
        error=0.005,
        maxiter=1000
    )

    u_vec = u_pred[:, 0]
    assigned = int(np.argmax(u_vec))

    memberships = [(i + 1, float(u_vec[i])) for i in range(len(u_vec))]  # C1..Ck
    return assigned + 1, memberships
