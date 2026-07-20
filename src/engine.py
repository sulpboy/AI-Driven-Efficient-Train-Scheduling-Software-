# engine.py
"""
Core data generation, heuristic scoring, and CP-SAT optimization logic
for the AI train induction & scheduling platform.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

@st.cache_data
def generate_sample_data():
    """Generate realistic sample data for 25 trains."""
    np.random.seed(42)
    trains = []
    now = datetime.now()

    for i in range(1, 26):
        train_id = f"KM{i:03d}"

        # Fitness certificates
        signal_cert = now + timedelta(days=np.random.randint(-10, 60))
        brake_cert  = now + timedelta(days=np.random.randint(-5, 45))
        safety_cert = now + timedelta(days=np.random.randint(-3, 30))

        # Maintenance status
        has_maintenance = np.random.random() < 0.2
        maintenance_priority = np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']) if has_maintenance else None

        # Mileage
        total_mileage   = np.random.randint(40_000, 120_000)
        monthly_mileage = np.random.randint(2_000, 8_000)

        # Branding
        has_branding = np.random.random() < 0.3
        brand_name   = np.random.choice(['TechCorp', 'GreenEnergy', 'SmartCity', 'HealthPlus']) if has_branding else None
        brand_hours_required = np.random.randint(4, 8) if has_branding else 0

        # Cleaning
        last_cleaning = now - timedelta(days=np.random.randint(1, 15))
        next_cleaning = last_cleaning + timedelta(days=14)

        # Stabling position (1-8 easy, 9-16 medium, 17-25 hard)
        stabling_bay = np.random.randint(1, 26)

        trains.append({
            'Train_ID': train_id,
            'Signal_Cert_Expiry': signal_cert,
            'Brake_Cert_Expiry': brake_cert,
            'Safety_Cert_Expiry': safety_cert,
            'Has_Maintenance': has_maintenance,
            'Maintenance_Priority': maintenance_priority,
            'Total_Mileage': total_mileage,
            'Monthly_Mileage': monthly_mileage,
            'Has_Branding': has_branding,
            'Brand_Name': brand_name,
            'Brand_Hours_Required': brand_hours_required,
            'Last_Cleaning': last_cleaning,
            'Next_Cleaning': next_cleaning,
            'Stabling_Bay': stabling_bay
        })

    return pd.DataFrame(trains)

@st.cache_data
def generate_feasible_test_data(required_trains=18):
    """Generate test data that GUARANTEES optimization feasibility."""
    np.random.seed(123)  # Different seed for test data
    trains = []
    now = datetime.now()
    
    # Ensure we have enough trains with valid certificates and no critical maintenance
    for i in range(1, 26):
        train_id = f"TK{i:03d}"  # Test trains with TK prefix
        
        # For the first 'required_trains + 5' trains, ensure they're eligible for service
        if i <= required_trains + 5:
            # Valid certificates (well into the future)
            signal_cert = now + timedelta(days=np.random.randint(30, 90))
            brake_cert  = now + timedelta(days=np.random.randint(30, 90)) 
            safety_cert = now + timedelta(days=np.random.randint(30, 90))
            
            # No critical maintenance for eligible trains
            has_maintenance = np.random.random() < 0.1  # Lower chance
            maintenance_priority = np.random.choice(['LOW', 'MEDIUM']) if has_maintenance else None
        else:
            # Remaining trains can have issues
            signal_cert = now + timedelta(days=np.random.randint(-10, 60))
            brake_cert  = now + timedelta(days=np.random.randint(-5, 45))
            safety_cert = now + timedelta(days=np.random.randint(-3, 30))
            has_maintenance = np.random.random() < 0.3
            maintenance_priority = np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']) if has_maintenance else None

        # Reasonable mileage distribution
        total_mileage = np.random.randint(50_000, 100_000)
        monthly_mileage = np.random.randint(3_000, 7_000)

        # Branding (moderate levels)
        has_branding = np.random.random() < 0.25
        brand_name = np.random.choice(['TestBrand1', 'TestBrand2', 'TestBrand3']) if has_branding else None
        brand_hours_required = np.random.randint(3, 6) if has_branding else 0

        # Cleaning (not too many requiring cleaning)
        last_cleaning = now - timedelta(days=np.random.randint(1, 10))
        next_cleaning = last_cleaning + timedelta(days=14)

        # Stabling bays (good distribution across 1-25)
        stabling_bay = np.random.randint(1, 26)

        trains.append({
            'Train_ID': train_id,
            'Signal_Cert_Expiry': signal_cert,
            'Brake_Cert_Expiry': brake_cert,
            'Safety_Cert_Expiry': safety_cert,
            'Has_Maintenance': has_maintenance,
            'Maintenance_Priority': maintenance_priority,
            'Total_Mileage': total_mileage,
            'Monthly_Mileage': monthly_mileage,
            'Has_Branding': has_branding,
            'Brand_Name': brand_name,
            'Brand_Hours_Required': brand_hours_required,
            'Last_Cleaning': last_cleaning,
            'Next_Cleaning': next_cleaning,
            'Stabling_Bay': stabling_bay
        })

    return pd.DataFrame(trains)

def calculate_train_score(train, weights):
    """Heuristic score used only for dashboard heatmap & what-if."""
    score = 100
    today = datetime.now()

    # Hard disqualifiers
    if train['Signal_Cert_Expiry'] < today:  return 0
    if train['Brake_Cert_Expiry']  < today:  return 0
    if train['Safety_Cert_Expiry'] < today:  return 0
    if train.get('Telecom_Cert_Expiry') is not None and train['Telecom_Cert_Expiry'] < today:
        return 0
    if train['Has_Maintenance'] and train['Maintenance_Priority'] == 'CRITICAL':
        return 0

    # Mileage (balance around 80k)
    avg_mileage = 80_000
    mileage_diff = abs(train['Total_Mileage'] - avg_mileage)
    mileage_score = max(0, 100 - (mileage_diff / 1000))
    score += mileage_score * weights['mileage']

    # Certificates: min days margin
    cert_margins = [
        (train['Signal_Cert_Expiry'] - today).days,
        (train['Brake_Cert_Expiry'] - today).days,
        (train['Safety_Cert_Expiry'] - today).days
    ]
    if train.get('Telecom_Cert_Expiry') is not None:
        cert_margins.append((train['Telecom_Cert_Expiry'] - today).days)
    cert_days = min(cert_margins)
    cert_score = min(100, max(0, cert_days) * 2)
    score += cert_score * weights['certificates']

    # Branding boost
    if train['Has_Branding']:
        score += 50 * weights['branding']

    # Cleaning freshness
    days_since_cleaning = (today - train['Last_Cleaning']).days
    cleaning_score = max(0, 100 - (days_since_cleaning * 7))
    score += cleaning_score * weights['cleaning']

    # Stabling ease (updated for 1-25 bays)
    if train['Stabling_Bay'] <= 8:
        stabling_score = 100
    elif train['Stabling_Bay'] <= 16:
        stabling_score = 60
    else:
        stabling_score = 30
    score += stabling_score * weights['stabling']

    # Maintenance penalty
    if train['Has_Maintenance']:
        if train['Maintenance_Priority'] == 'HIGH':   score -= 30
        elif train['Maintenance_Priority'] == 'MEDIUM': score -= 15
        elif train['Maintenance_Priority'] == 'LOW':    score -= 5

    return max(0, score)

# ───────────────────────────────────────────────────────────────────────────────
# Optimisation helpers (adds Telecom cert, cleaning need, and builds tonight’s inputs)
# ───────────────────────────────────────────────────────────────────────────────
def generate_operational_inputs(required_trains:int):
    """Synthetic operational inputs for diagrams, cleaning slots, and bays."""
    now = datetime.now()

    # Diagrams: spaced departures ~05:00–08:30
    start = (now + timedelta(days=1)).replace(hour=5, minute=0, second=0, microsecond=0)
    diagrams = []
    t = start
    for i in range(1, required_trains + 1):
        hop = np.random.randint(10, 16)
        t = t + timedelta(minutes=hop) if i > 1 else t
        km = np.random.randint(120, 220)
        exposure_h = np.random.randint(3, 6)
        diagrams.append({
            "Diagram_ID": f"D{i:02d}",
            "Depart_Time": t,
            "Planned_Km": km,
            "Exposure_Hours": exposure_h
        })
    df_diagrams = pd.DataFrame(diagrams).sort_values("Depart_Time").reset_index(drop=True)

    # Cleaning slots and capacities (20:30–23:00)
    slots = [
        {"Slot_ID": "C1", "Start": now.replace(hour=20, minute=30, second=0, microsecond=0), "End": now.replace(hour=21, minute=30, second=0, microsecond=0), "Capacity": 6},
        {"Slot_ID": "C2", "Start": now.replace(hour=21, minute=30, second=0, microsecond=0), "End": now.replace(hour=22, minute=30, second=0, microsecond=0), "Capacity": 6},
        {"Slot_ID": "C3", "Start": now.replace(hour=22, minute=30, second=0, microsecond=0), "End": now.replace(hour=23, minute=0, second=0, microsecond=0),  "Capacity": 5},
    ]
    df_slots = pd.DataFrame(slots)

    # Bays (1–25) on ladders A (1–12) and B (13–25) - enough for all trains
    bays = []
    for b in range(1, 26):  # Increased from 16 to 26 to accommodate all 25 trains
        ladder = "A" if b <= 12 else "B"  # Adjusted ladder split
        base_cost = 1 if b <= 8 else 3 if b <= 16 else 6  # Extended cost tiers
        bays.append({"Bay": b, "Ladder": ladder, "ExitCost": base_cost})
    df_bays = pd.DataFrame(bays)

    return df_diagrams, df_slots, df_bays

def extend_train_schema_for_constraints(df_trains: pd.DataFrame) -> pd.DataFrame:
    """Add Telecom certificate, normalise maintenance priority, and compute cleaning need."""
    rng = np.random.default_rng(7)
    df = df_trains.copy()

    # Add Telecom certificate
    df["Telecom_Cert_Expiry"] = [
        datetime.now() + timedelta(days=int(rng.integers(-7, 40))) for _ in range(len(df))
    ]

    # Ensure a priority string exists if a train has maintenance
    def _fix_priority(row):
        if row["Has_Maintenance"] and row["Maintenance_Priority"] is None:
            return np.random.choice(["LOW","MEDIUM","HIGH","CRITICAL"], p=[0.4,0.35,0.2,0.05])
        return row["Maintenance_Priority"]
    df["Maintenance_Priority"] = df.apply(_fix_priority, axis=1)

    # Cleaning need: due if Next_Cleaning <= tomorrow 05:00
    tomorrow_5 = (datetime.now() + timedelta(days=1)).replace(hour=5, minute=0, second=0, microsecond=0)
    df["Needs_Deep_Clean"] = df["Next_Cleaning"] <= tomorrow_5

    return df

def _cert_valid_for_service(row: pd.Series, window_end: datetime) -> bool:
    return (row["Signal_Cert_Expiry"] >= window_end and
            row["Brake_Cert_Expiry"]  >= window_end and
            row["Safety_Cert_Expiry"] >= window_end and
            row["Telecom_Cert_Expiry"]>= window_end)


# ───────────────────────────────────────────────────────────────────────────────
# Optimiser (CP-SAT): service assignment, bays, cleaning, branding, mileage
# ───────────────────────────────────────────────────────────────────────────────
def optimize_schedule(df_trains: pd.DataFrame,
                      df_diagrams: pd.DataFrame,
                      df_slots: pd.DataFrame,
                      df_bays: pd.DataFrame,
                      weights: dict):
    """
    CP-SAT model with flexible constraints for real-world scenarios:
    - Assign each diagram to exactly one eligible train (x[t,d])
    - One mode per train: service OR standby OR IBL
    - Cleaning: if due and serving → must book one slot within capacity (q[t,s])
    - Bays: unique assignment for all trains (y[t,b])
    - Branding: minimise shortfall vs. tonight's required hours
    - Mileage: minimise L1 deviation around target μ
    """
    # Index sets
    T = list(df_trains.index)
    D = list(df_diagrams.index)
    S = list(df_slots.index)
    B = list(df_bays.index)

    # Window end surrogate: min departure + 18h
    service_window_end = (df_diagrams["Depart_Time"].min() + timedelta(hours=18))
    
    # Check feasibility and adjust requirements if needed
    total_trains = len(df_trains)
    required_diagrams = len(df_diagrams)
    
    # Eligibility levels (from strictest to most lenient)
    def get_eligibility_levels():
        levels = {}
        
        # Level 1: Perfect trains (all certs valid, no critical maintenance)
        levels['perfect'] = {t: (_cert_valid_for_service(df_trains.loc[t], service_window_end) and 
                                not (df_trains.loc[t, "Has_Maintenance"] and df_trains.loc[t, "Maintenance_Priority"] == "CRITICAL"))
                           for t in T}
        
        # Level 2: Valid certs but allow high maintenance (not critical)
        levels['valid_certs'] = {t: (_cert_valid_for_service(df_trains.loc[t], service_window_end) and 
                                    not (df_trains.loc[t, "Has_Maintenance"] and df_trains.loc[t, "Maintenance_Priority"] == "CRITICAL"))
                                for t in T}
        
        # Level 3: Allow slightly expired certs (up to 7 days) but no critical maintenance
        relaxed_window = service_window_end - timedelta(days=7)
        levels['relaxed_certs'] = {t: (_cert_valid_for_service(df_trains.loc[t], relaxed_window) and 
                                      not (df_trains.loc[t, "Has_Maintenance"] and df_trains.loc[t, "Maintenance_Priority"] == "CRITICAL"))
                                  for t in T}
        
        # Level 4: Emergency mode - only exclude trains with critical maintenance
        levels['emergency'] = {t: not (df_trains.loc[t, "Has_Maintenance"] and df_trains.loc[t, "Maintenance_Priority"] == "CRITICAL")
                              for t in T}
        
        return levels
    
    eligibility_levels = get_eligibility_levels()
    
    # Find the most restrictive level that provides enough trains
    chosen_eligibility = None
    eligibility_name = ""
    
    for level_name, eligibility in [('perfect', 'perfect'), ('valid_certs', 'valid_certs'), 
                                   ('relaxed_certs', 'relaxed_certs'), ('emergency', 'emergency')]:
        eligible_count = sum(1 for eligible in eligibility_levels[eligibility].values() if eligible)
        if eligible_count >= required_diagrams:
            chosen_eligibility = eligibility_levels[eligibility]
            eligibility_name = level_name
            break
    
    # If even emergency mode doesn't work, use it anyway and let solver handle partial assignment
    if chosen_eligibility is None:
        chosen_eligibility = eligibility_levels['emergency']
        eligibility_name = 'emergency'

    # Use the chosen eligibility level
    valid = chosen_eligibility
    crit = {t: (df_trains.loc[t, "Has_Maintenance"] and df_trains.loc[t, "Maintenance_Priority"] == "CRITICAL") for t in T}
    needs_clean = {t: bool(df_trains.loc[t, "Needs_Deep_Clean"]) for t in T}

    # Branding contracts
    brands = sorted(set(df_trains.loc[df_trains["Has_Branding"], "Brand_Name"].dropna().tolist()))
    P = list(range(len(brands)))
    brand_to_idx = {b:i for i,b in enumerate(brands)}
    Ereq = np.zeros(len(P), dtype=int)
    for _, r in df_trains.iterrows():
        if r["Has_Branding"] and r["Brand_Name"] in brand_to_idx:
            Ereq[brand_to_idx[r["Brand_Name"]]] += int(r["Brand_Hours_Required"])

    # Diagram attributes
    h_d = df_diagrams["Exposure_Hours"].astype(int).tolist()
    km_d = df_diagrams["Planned_Km"].astype(int).tolist()
    mu_target = int(np.mean(km_d)) if len(km_d) else 0
    
    # Branding incidence
    alpha = {(t,p): 1 if (df_trains.loc[t, "Has_Branding"] and df_trains.loc[t, "Brand_Name"]==brands[p]) else 0
             for t in T for p in P}

    # Bay costings
    # Build model
    model = cp_model.CpModel()

    # Vars
    x = {(t,d): model.NewBoolVar(f"x_t{t}_d{d}") for t in T for d in D}
    standby = {t: model.NewBoolVar(f"u_t{t}") for t in T}
    ibl     = {t: model.NewBoolVar(f"v_t{t}") for t in T}
    y = {(t,b): model.NewBoolVar(f"y_t{t}_b{b}") for t in T for b in B}
    q = {(t,s): model.NewBoolVar(f"q_t{t}_s{s}") for t in T for s in S}

    m      = {t: model.NewIntVar(0, 2000, f"m_t{t}") for t in T}
    dplus  = {t: model.NewIntVar(0, 2000, f"dplus_t{t}") for t in T}
    dminus = {t: model.NewIntVar(0, 2000, f"dminus_t{t}") for t in T}
    short  = {p: model.NewIntVar(0, 500, f"short_p{p}") for p in P}

    # Constraints
    # 1) Cover diagrams exactly once
    for d in D:
        model.Add(sum(x[(t,d)] for t in T) == 1)

    # 2) One mode per train (service≤1 diagram) OR standby OR IBL
    for t in T:
        model.Add(sum(x[(t,d)] for d in D) + standby[t] + ibl[t] == 1)

    # 3) Eligibility gates
    for t in T:
        for d in D:
            if not valid[t] or crit[t]:
                model.Add(x[(t,d)] == 0)

    # 4) Flexible cleaning: if due and serving → try to book slot, but allow postponing if capacity insufficient
    cleaning_penalty = {}  # Track trains that need cleaning but can't get slots
    for t in T:
        service_t = model.NewBoolVar(f"service_t{t}")
        model.Add(sum(x[(t,d)] for d in D) == service_t)
        
        if needs_clean[t]:
            # Create a penalty variable for not getting cleaning when needed
            no_cleaning_penalty = model.NewBoolVar(f"no_clean_penalty_t{t}")
            cleaning_penalty[t] = no_cleaning_penalty
            
            # Either get cleaning slot OR incur penalty (but only if in service)
            slots_booked = sum(q[(t,s)] for s in S)
            model.Add(slots_booked + no_cleaning_penalty >= service_t)
            model.Add(slots_booked <= service_t)  # Can only book if in service
        else:
            model.Add(sum(q[(t,s)] for s in S) == 0)
            cleaning_penalty[t] = model.NewBoolVar(f"dummy_clean_penalty_t{t}")
            model.Add(cleaning_penalty[t] == 0)  # No penalty if cleaning not needed

    # 5) Slot capacities
    for s in S:
        model.Add(sum(q[(t,s)] for t in T) <= int(df_slots.loc[s, "Capacity"]))

    # 6) Bay assignment: one per train; at most one train per bay
    for t in T:
        model.Add(sum(y[(t,b)] for b in B) == 1)
    for b in B:
        model.Add(sum(y[(t,b)] for t in T) <= 1)

    # 7) Mileage balancing
    for t in T:
        model.Add(m[t] == sum(int(km_d[d]) * x[(t,d)] for d in D))
        model.Add(m[t] - int(mu_target) == dplus[t] - dminus[t])

    # 8) Branding shortfall
    for p in P:
        planned = sum(int(h_d[d]) * alpha[(t,p)] * x[(t,d)] for t in T for d in D)
        model.Add(short[p] >= Ereq[p] - planned)
        model.Add(short[p] >= 0)

    # Objective weights
    W = {
        "branding" : int(100 * weights.get("branding", 1.0)),
        "mileage"  : int(50  * weights.get("mileage", 1.0)),
        "certs"    : int(40  * weights.get("certificates", 1.0)),
        "cleaning" : int(20  * weights.get("cleaning", 1.0)),
        "stabling" : int(15  * weights.get("stabling", 1.0)),
    }

    # Stabling cost: discourage placing serving trains in hard-exit bays
    stabling_cost_terms = []
    for t in T:
        for b in B:
            per_pair = int(10 * df_bays.loc[b, "ExitCost"])
            stabling_cost_terms.append(per_pair * sum(x[(t,d)] for d in D))

    # Mileage L1 deviation
    mileage_cost = sum(dplus[t] + dminus[t] for t in T)

    # Branding shortfall
    branding_cost = sum(short[p] for p in P)

    # Certificate margin reward (prefer longer validity among serving trains)
    cert_reward = 0
    today = datetime.now()
    for t in T:
        days_list = []
        for col in ["Signal_Cert_Expiry","Brake_Cert_Expiry","Safety_Cert_Expiry","Telecom_Cert_Expiry"]:
            days_list.append(max(0, (df_trains.loc[t, col] - today).days))
        margin = min(days_list) if len(days_list) else 0
        margin = min(30, margin)
        cert_reward += margin * sum(x[(t,d)] for d in D)

    # Cleaning reward if due and booked
    cleaning_reward = sum(int(needs_clean[t]) * sum(q[(t,s)] for s in S) for t in T)
    cleaning_penalty_cost = sum(50 * cleaning_penalty[t] for t in T)  # Penalty for not cleaning when needed

    # Also keep bays “reasonable” even for non-serving modes (light penalty)
    bay_smoothing = 5 * sum(int(df_bays.loc[b, "ExitCost"]) * y[(t,b)] for t in T for b in B)

    objective = (
        - W["certs"]    * cert_reward
        - W["cleaning"] * cleaning_reward
        + W["cleaning"] * cleaning_penalty_cost  # Penalty for missed cleaning
        + W["stabling"] * sum(stabling_cost_terms)
        + W["mileage"]  * mileage_cost
        + W["branding"] * branding_cost
        + bay_smoothing
    )
    model.Minimize(objective)

    # Solve with more generous time limit for complex scenarios
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0  # Increased from 8.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    plan = {
        "status": status,
        "service": [],
        "standby": [],
        "ibl": [],
        "kpis": {},
        "eligibility_mode": eligibility_name,  # Add info about what constraints were used
        "cleaning_issues": []  # Track cleaning problems
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return plan

    # Extract chosen bays & slots
    bay_of = {}
    for t in T:
        for b in B:
            if solver.Value(y[(t,b)]) == 1:
                bay_of[t] = int(df_bays.loc[b, "Bay"])

    slot_of = {}
    for t in T:
        for s in S:
            if solver.Value(q[(t,s)]) == 1:
                slot_of[t] = df_slots.loc[s, "Slot_ID"]

    # Service assignments
    for d in D:
        for t in T:
            if solver.Value(x[(t,d)]) == 1:
                row = df_trains.loc[t]
                brand = row["Brand_Name"] if row["Has_Branding"] else "-"
                plan["service"].append({
                    "Diagram_ID"   : df_diagrams.loc[d, "Diagram_ID"],
                    "Depart_Time"  : df_diagrams.loc[d, "Depart_Time"].strftime("%H:%M"),
                    "Train_ID"     : row["Train_ID"],
                    "Bay"          : bay_of.get(t, None),
                    "Cleaning_Slot": slot_of.get(t, "-") if needs_clean[t] else "-",
                    "Planned_Km"   : int(df_diagrams.loc[d, "Planned_Km"]),
                    "Brand"        : brand
                })

    # Non-service modes
    for t in T:
        service_assigned = any(solver.Value(x[(t,d)]) for d in D)
        if service_assigned:
            continue
        row = df_trains.loc[t]
        entry = {
            "Train_ID": row["Train_ID"],
            "Bay": bay_of.get(t, None),
            "Brand": row["Brand_Name"] if row["Has_Branding"] else "-",
        }
        if solver.Value(standby[t]) == 1:
            plan["standby"].append(entry)
        elif solver.Value(ibl[t]) == 1:
            plan["ibl"].append(entry)

    # Track cleaning issues
    for t in T:
        if needs_clean[t] and solver.Value(cleaning_penalty[t]) == 1:
            service_assigned = any(solver.Value(x[(t,d)]) for d in D)
            if service_assigned:
                train_id = df_trains.loc[t, "Train_ID"]
                plan["cleaning_issues"].append(f"{train_id}: Needs cleaning but no slot available")

    # KPIs
    plan["kpis"]["BrandContracts"]   = len(brands)
    plan["kpis"]["BrandShortfallHours"] = sum(solver.Value(short[p]) for p in P) if P else 0
    plan["kpis"]["AvgServingKm"]     = int(np.mean([r["Planned_Km"] for r in plan["service"]])) if plan["service"] else 0
    plan["kpis"]["CleaningBooked"]   = sum(1 for t in T if t in slot_of)
    plan["kpis"]["CleaningIssues"]   = len(plan["cleaning_issues"])
    plan["kpis"]["StandbyCount"]     = len(plan["standby"])
    plan["kpis"]["IBLCount"]         = len(plan["ibl"])

    # Why strings
    def _ladder_of_bay(bay_num):
        if pd.isna(bay_num): return "?"
        if bay_num <= 12: return "A"
        return "B"

    def _explain(t_idx, diagram_idx):
        row = df_trains.loc[t_idx]
        msgs = []
        # Certificates
        days = []
        for c in ["Signal_Cert_Expiry","Brake_Cert_Expiry","Safety_Cert_Expiry","Telecom_Cert_Expiry"]:
            days.append(max(0, (row[c] - datetime.now()).days))
        min_days = min(days) if days else 0
        if min_days >= 30: msgs.append(f"✅ Certs valid ≥30d")
        elif min_days >= 7: msgs.append(f"⚠️ Certs expire in {min_days}d")
        else: msgs.append(f"⚠️ Certs expiring soon ({min_days}d)")

        # Branding
        if row["Has_Branding"]:
            msgs.append(f"🎯 Branding: {row['Brand_Name']}")

        # Cleaning
        if needs_clean[t_idx]:
            msgs.append(f"🧽 Cleaning booked: {slot_of.get(t_idx, '—')}")

        # Mileage vs μ
        msgs.append(f"📏 Duty {int(df_diagrams.loc[diagram_idx, 'Planned_Km'])}km vs μ={mu_target}km")

        # Bay/Ladder
        bay_num = bay_of.get(t_idx, None)
        msgs.append(f"🚪 Bay {bay_num} (Ladder {_ladder_of_bay(bay_num)})")

        # Maintenance
        if row["Has_Maintenance"]:
            msgs.append(f"🛠 JC: {row['Maintenance_Priority']}")

        return " | ".join(msgs)

    for s in plan["service"]:
        t_idx = df_trains.index[df_trains["Train_ID"] == s["Train_ID"]][0]
        d_idx = df_diagrams.index[df_diagrams["Diagram_ID"] == s["Diagram_ID"]][0]
        s["Why"] = _explain(t_idx, d_idx)

    return plan
