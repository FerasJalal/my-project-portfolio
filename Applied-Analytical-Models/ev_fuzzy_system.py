import numpy as np
import skfuzzy as fuzz

# =========================
# Universes
# =========================
time_universe = np.arange(0, 301, 1)
cost_universe = np.arange(0, 51, 1)
urgency_universe = np.arange(0, 11, 1)
budget_universe = np.arange(0, 11, 1)
comfort_universe = np.arange(0, 101, 1)

# =========================
# Membership Functions
# =========================

time_short = fuzz.trimf(time_universe, [0, 0, 60])
time_medium = fuzz.trimf(time_universe, [40, 120, 200])
time_long = fuzz.trimf(time_universe, [150, 300, 300])

cost_low = fuzz.trimf(cost_universe, [0, 0, 15])
cost_medium = fuzz.trimf(cost_universe, [10, 25, 40])
cost_high = fuzz.trimf(cost_universe, [30, 50, 50])

urg_low = fuzz.trimf(urgency_universe, [0, 0, 4])
urg_med = fuzz.trimf(urgency_universe, [3, 5, 7])
urg_high = fuzz.trimf(urgency_universe, [6, 10, 10])

bud_low = fuzz.trimf(budget_universe, [0, 0, 4])
bud_med = fuzz.trimf(budget_universe, [3, 5, 7])
bud_high = fuzz.trimf(budget_universe, [6, 10, 10])

comfort_low = fuzz.trimf(comfort_universe, [0, 0, 40])
comfort_med = fuzz.trimf(comfort_universe, [30, 50, 70])
comfort_high = fuzz.trimf(comfort_universe, [60, 100, 100])


def _m(v, u, mf):
    return fuzz.interp_membership(u, mf, v)


def calculate_charging_comfort(predicted_time, predicted_cost, urgency, budget):

    predicted_time = float(np.clip(predicted_time, 0, 300))
    predicted_cost = float(np.clip(predicted_cost, 0, 50))
    urgency = float(np.clip(urgency, 0, 10))
    budget = float(np.clip(budget, 0, 10))

    t_s = _m(predicted_time, time_universe, time_short)
    t_m = _m(predicted_time, time_universe, time_medium)
    t_l = _m(predicted_time, time_universe, time_long)

    c_l = _m(predicted_cost, cost_universe, cost_low)
    c_m = _m(predicted_cost, cost_universe, cost_medium)
    c_h = _m(predicted_cost, cost_universe, cost_high)

    u_l = _m(urgency, urgency_universe, urg_low)
    u_m = _m(urgency, urgency_universe, urg_med)
    u_h = _m(urgency, urgency_universe, urg_high)

    b_l = _m(budget, budget_universe, bud_low)
    b_m = _m(budget, budget_universe, bud_med)
    b_h = _m(budget, budget_universe, bud_high)

    low = np.fmax(np.fmin(u_h, t_l), np.fmin(b_h, c_h))
    med = np.fmax(np.fmin(t_m, c_m), np.fmin(u_m, b_m))
    high = np.fmax(np.fmin(t_s, c_l), np.fmin(u_l, b_l))

    aggregated = np.fmax(
        np.fmin(low, comfort_low),
        np.fmax(np.fmin(med, comfort_med), np.fmin(high, comfort_high))
    )

    comfort_score = fuzz.defuzz(comfort_universe, aggregated, "centroid")

    if comfort_score < 40:
        strategy = "Fast"
        advice = "Fast charging is recommended due to urgency or inefficiency."
    elif comfort_score < 70:
        strategy = "Balance"
        advice = "A balanced charging strategy suits your needs."
    else:
        strategy = "Slow"
        advice = "Slow charging maximizes comfort and reduces cost."

    return round(comfort_score, 2), strategy, advice
