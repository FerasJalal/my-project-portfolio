import numpy as np
import pandas as pd
import itertools
import random

df = pd.read_csv("data/ev_charging_dataset_Final(in).csv")

CHARGER_MAP = {
    "Level 1": 0,
    "Level 2": 1,
    "DC Fast Charger": 2,
}

USER_MAP = {
    "Casual Driver": 0,
    "Commuter": 1,
    "Long-Distance Traveler": 2,
}

df["Charger_Level"] = df["Charger Type"].map(CHARGER_MAP)
df["User_Level"] = df["User Type"].map(USER_MAP)

df["SOC_Level"], soc_bins = pd.qcut(
    df["State of Charge (Start %)"],
    3,
    labels=[0, 1, 2],
    retbins=True,
    duplicates="drop",
)
df["SOC_Level"] = df["SOC_Level"].astype(int)

df["Energy_Level"], energy_bins = pd.qcut(
    df["Energy Consumed (kWh)"],
    3,
    labels=[0, 1, 2],
    retbins=True,
    duplicates="drop",
)
df["Energy_Level"] = df["Energy_Level"].astype(int)

df["Battery_Level"], battery_bins = pd.qcut(
    df["Battery Capacity (kWh)"],
    3,
    labels=[0, 1, 2],
    retbins=True,
    duplicates="drop",
)
df["Battery_Level"] = df["Battery_Level"].astype(int)

df["Temp_Level"], temp_bins = pd.qcut(
    df["Temperature (°C)"],
    3,
    labels=[0, 1, 2],
    retbins=True,
    duplicates="drop",
)
df["Temp_Level"] = df["Temp_Level"].astype(int)

df["State"] = list(
    zip(
        df["SOC_Level"],
        df["Energy_Level"],
        df["Battery_Level"],
        df["Charger_Level"],
        df["Temp_Level"],
        df["User_Level"],
    )
)

actions = {
    0: "Charge Now",
    1: "Switch to Faster Charger / Another Station",
    2: "Delay Charging",
}

df["Reward"] = (
    (2 - df["SOC_Level"]) * 2
    + (2 - df["Energy_Level"]) * 2
    + (2 - df["Charger_Level"])
)

all_states = list(
    itertools.product(
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    )
)

state_to_index = {s: i for i, s in enumerate(all_states)}
Q = np.zeros((len(all_states), 1))

episodes = 20000
alpha = 0.05
epsilon = 1.0
epsilon_decay = 0.97
epsilon_min = 0.05

for _ in range(episodes):
    s = random.choice(all_states)
    s_idx = state_to_index[s]

    subset = df[df["State"] == s]
    r = subset.sample(1)["Reward"].values[0] if len(subset) > 0 else 0

    Q[s_idx, 0] += alpha * (r - Q[s_idx, 0])
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

def safe_bin(value, bins, labels):
    value = np.clip(value, bins[0], bins[-1])
    level = pd.cut([value], bins=bins, labels=labels, include_lowest=True)[0]
    return int(level)

def recommend_charging_action(
    soc_start,
    energy_consumed,
    battery_capacity,
    temperature,
    charger_type,
    user_type,
):
    soc = safe_bin(soc_start, soc_bins, [0, 1, 2])
    energy = safe_bin(energy_consumed, energy_bins, [0, 1, 2])
    battery = safe_bin(battery_capacity, battery_bins, [0, 1, 2])
    temp = safe_bin(temperature, temp_bins, [0, 1, 2])

    state = (
        soc,
        energy,
        battery,
        CHARGER_MAP[charger_type],
        temp,
        USER_MAP[user_type],
    )

    urgency = Q[state_to_index[state], 0]

    if urgency >= 4:
        return actions[0]  # Charge Now
    elif urgency >= 2:
        return actions[1]  # Switch
    else:
        return actions[2]  # Delay
