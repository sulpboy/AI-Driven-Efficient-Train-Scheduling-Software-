import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ortools.sat.python import cp_model


class HeterogeneousIngestionEngine:
    """
    Validates, normalizes, and transforms multi-source unstructured/structured
    transit metrics into a strict typed DataFrame for downstream scheduling.
    """

    @staticmethod
    def parse_telemetry(raw_payloads: list[dict]) -> pd.DataFrame:
        clean_records = []
        now = datetime.now()

        for idx, payload in enumerate(raw_payloads):
            try:
                train_id = str(payload.get("id_string", f"TS-{idx:03d}")).replace("-", "").strip()

                # Normalize varying incoming timestamp formats (ISO, legacy string, etc.)
                sig_cert = pd.to_datetime(payload.get("sig_cert", now + timedelta(days=30)))
                brk_cert = pd.to_datetime(payload.get("brk_log", now + timedelta(days=30)))
                sf_cert = pd.to_datetime(payload.get("safety_meta", now + timedelta(days=30)))

                # Text mining layer: classify safety state from unstructured operator logs
                maint_log = str(payload.get("unstructured_maintenance_text", "Status Clear")).upper()
                has_maint = False
                maint_prio = "NONE"

                if any(x in maint_log for x in ["CRITICAL", "BREAKDOWN", "SEVERE"]):
                    has_maint = True
                    maint_prio = "CRITICAL"
                elif any(x in maint_log for x in ["WARNING", "REPAIR", "DEGRADED"]):
                    has_maint = True
                    maint_prio = "HIGH"
                elif any(x in maint_log for x in ["SCHEDULED", "ROUTINE", "CHECK"]):
                    has_maint = True
                    maint_prio = "LOW"

                clean_records.append({
                    'Train_ID': train_id,
                    'Signal_Cert_Expiry': sig_cert,
                    'Brake_Cert_Expiry': brk_cert,
                    'Safety_Cert_Expiry': sf_cert,
                    'Has_Maintenance': has_maint,
                    'Maintenance_Priority': maint_prio,
                    'Total_Mileage': int(payload.get("odometer_reading", 80000)),
                    'Has_Branding': bool(payload.get("ad_contract_active", False)),
                    'Brand_Name': payload.get("contract_client", None),
                    'Brand_Hours_Required': int(payload.get("hours_owed", 0)),
                    'Last_Cleaning': pd.to_datetime(payload.get("cleaned_at", now - timedelta(days=5))),
                    'Stabling_Bay': int(payload.get("bay_location", 1))
                })
            except Exception as e:
                # Fault isolation: skip corrupt records instead of crashing the pipeline
                print(f"[INGESTION] Skipped payload index {idx}: {str(e)}")
                continue

        return pd.DataFrame(clean_records)


class MultiObjectiveScheduler:
    """
    Multi-objective train scheduling via Google OR-Tools CP-SAT, balancing
    mileage variance, stabling-bay friction, and certificate expiry margins.
    """

    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def build_and_solve(self, df_trains: pd.DataFrame, df_diagrams: pd.DataFrame, df_bays: pd.DataFrame) -> dict:
        T = list(df_trains.index)
        D = list(df_diagrams.index)
        B = list(df_bays.index)

        model = cp_model.CpModel()
        now = datetime.now()

        # Decision variables
        x = {(t, d): model.NewBoolVar(f"x_t{t}_d{d}") for t in T for d in D}
        standby = {t: model.NewBoolVar(f"standby_t{t}") for t in T}
        ibl = {t: model.NewBoolVar(f"ibl_t{t}") for t in T}
        y = {(t, b): model.NewBoolVar(f"y_t{t}_b{b}") for t in T for b in B}

        # Linearization variables for absolute mileage deviation
        dplus = {t: model.NewIntVar(0, 10000, f"dplus_t{t}") for t in T}
        dminus = {t: model.NewIntVar(0, 10000, f"dminus_t{t}") for t in T}

        # Operational constraints
        for d in D:
            model.Add(sum(x[(t, d)] for t in T) == 1)  # every diagram gets exactly one train

        for t in T:
            model.Add(sum(x[(t, d)] for d in D) + standby[t] + ibl[t] == 1)  # mutually exclusive states
            model.Add(sum(y[(t, b)] for b in B) == 1)  # every train needs exactly one bay

            # Hard safety gate: critical-maintenance trains are pulled from service
            if df_trains.loc[t, "Maintenance_Priority"] == "CRITICAL":
                model.Add(sum(x[(t, d)] for d in D) == 0)
                model.Add(ibl[t] == 1)

        for b in B:
            model.Add(sum(y[(t, b)] for t in T) <= 1)  # no bay collisions

        # Mileage deviation from fleet target
        mu_target = int(df_diagrams["Planned_Km"].mean()) if len(df_diagrams) > 0 else 0
        mileage_terms = []
        for t in T:
            model.Add(sum(int(df_diagrams.loc[d, "Planned_Km"]) * x[(t, d)] for d in D) - mu_target == dplus[t] - dminus[t])
            mileage_terms.append(dplus[t] + dminus[t])

        # Stabling friction: linearize (train in service) AND (assigned to bay b)
        W = {k: int(v * 100) for k, v in self.weights.items()}
        stabling_terms = []
        for t in T:
            is_serving = sum(x[(t, d)] for d in D)
            for b in B:
                y_served = model.NewBoolVar(f"y_served_{t}_{b}")
                model.Add(y_served <= y[(t, b)])
                model.Add(y_served <= is_serving)
                model.Add(y_served >= y[(t, b)] + is_serving - 1)
                stabling_terms.append(int(df_bays.loc[b, "ExitCost"]) * y_served)

        # Certificate expiry buffer (negative cost acts as a maximization reward)
        cert_rewards = []
        for t in T:
            margin_days = min(30, max(0, (df_trains.loc[t, "Signal_Cert_Expiry"] - now).days))
            cert_rewards.append(margin_days * sum(x[(t, d)] for d in D))

        mileage_cost = W.get("mileage", 100) * sum(mileage_terms)
        stabling_cost = W.get("stabling", 50) * sum(stabling_terms)
        cert_cost = -W.get("certificates", 120) * sum(cert_rewards)

        model.Minimize(mileage_cost + stabling_cost + cert_cost)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # bounded, predictable solve time
        status = solver.Solve(model)

        return self._package_results(status, solver, x, y, df_trains, df_diagrams, df_bays, mu_target)

    def _package_results(self, status, solver, x, y, df_trains, df_diagrams, df_bays, mu_target) -> dict:
        output = {"status": status, "service": [], "xai_attributions": {}}
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return output

        T, D, B = list(df_trains.index), list(df_diagrams.index), list(df_bays.index)
        now = datetime.now()

        for t in T:
            t_id = df_trains.loc[t, "Train_ID"]
            assigned_d = [d for d in D if solver.Value(x[(t, d)]) == 1]
            assigned_b = [b for b in B if solver.Value(y[(t, b)]) == 1]

            # XAI attribution: per-train breakdown of each cost term's contribution
            mil_dev = abs(sum(df_diagrams.loc[d, "Planned_Km"] for d in assigned_d) - mu_target) if assigned_d else 0
            cert_margin = min(30, max(0, (df_trains.loc[t, "Signal_Cert_Expiry"] - now).days))
            stabling_friction = df_bays.loc[assigned_b[0], "ExitCost"] if assigned_b else 0

            output["xai_attributions"][t_id] = {
                "Mileage Target Variance Penalty": float(mil_dev * self.weights["mileage"]),
                "Certificate Expiration Cost Reward": float(-cert_margin * self.weights["certificates"]),
                "Stabling Bay Access Bottleneck Cost": float(stabling_friction * self.weights["stabling"])
            }

            if assigned_d:
                d_idx = assigned_d[0]
                output["service"].append({
                    "Diagram_ID": df_diagrams.loc[d_idx, "Diagram_ID"],
                    "Depart_Time": df_diagrams.loc[d_idx, "Depart_Time"].strftime("%H:%M"),
                    "Train_ID": t_id,
                    "Bay": int(df_bays.loc[assigned_b[0], "Bay"]) if assigned_b else 0,
                    "Planned_Km": int(df_diagrams.loc[d_idx, "Planned_Km"])
                })
        return output


class OnlineFeedbackLearningLoop:
    """
    Inverse-optimization update step: adjusts multi-objective weights based on
    supervisor rejection feedback via online gradient descent.
    """

    def __init__(self, learning_rate: float = 0.25):
        self.eta = learning_rate

    def compute_gradient_update(self, current_weights: dict[str, float], rejection_profile: str) -> dict[str, float]:
        new_weights = current_weights.copy()

        if rejection_profile == "HIGH_MILEAGE_VARIANCE":
            new_weights["mileage"] += self.eta * 2.0
            new_weights["stabling"] -= self.eta * 0.5
        elif rejection_profile == "SAFETY_MARGIN_VIOLATION":
            new_weights["certificates"] += self.eta * 2.5
            new_weights["mileage"] -= self.eta * 0.2
        elif rejection_profile == "STABLING_BOTTLENECK":
            new_weights["stabling"] += self.eta * 1.8

        # Clip weights to valid positive bounds
        for k in new_weights:
            new_weights[k] = round(float(np.clip(new_weights[k], 0.1, 5.0)), 3)

        return new_weights
