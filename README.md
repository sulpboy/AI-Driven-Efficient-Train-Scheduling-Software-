# AI-Driven-Efficient-Train-Scheduling-Software-

A fleet optimization engine combining heterogeneous telemetry ingestion, a Google OR-Tools CP-SAT multi-objective scheduler, SHAP-inspired Explainable AI (XAI) attribution, and an online gradient-descent preference feedback loop.

## System Architecture & Core Features

### 1. Heterogeneous Data Ingestion Engine

Processes raw, unformatted, multi-source inputs — including ISO strings, legacy timestamps, nested JSON configurations, and unstructured free-text operator maintenance logs.

- **Text Mining Layer**: Scans raw telemetry logs using pattern-matching pipelines to classify high, medium, and critical safety maintenance states.
- **Fault Isolation**: Built-in validation limits isolate corrupt payload elements, keeping data streaming stable without application crashes.

### 2. Multi-Objective CP-SAT Scheduler

Replaces heuristic scorecards with a unified mathematical optimization model using the Google OR-Tools CP-SAT solver. Conflicting scheduling goals — mileage balance, stabling-bay friction, and certificate runway — are combined into a single objective function:

$$\min \left( W_{\text{mileage}} \sum_{t \in T} (d^+_t + d^-_t) + W_{\text{stabling}} \sum_{t \in T} \sum_{b \in B} C_b y_{tb} - W_{\text{certs}} \sum_{t \in T} M_t \sum_{d \in D} x_{td} \right)$$

Where:
- $x_{td} \in \{0, 1\}$: whether Train $t$ is assigned to Diagram $d$
- $y_{tb} \in \{0, 1\}$: whether Train $t$ is placed in Stabling Bay $b$
- $d^+_t, d^-_t$: positive/negative mileage deviation from the fleet target average ($\mu$)
- $C_b$: cost coefficient penalizing deployment friction in restricted stabling-bay ladders
- $M_t$: remaining runway buffer before safety/signaling certificate expiry

### 3. Explainable AI (XAI) Attribution Mapping

For every scheduled assignment, the engine extracts the final penalty values directly from the solver's decision variables and renders them as a Plotly waterfall chart, showing the feature attributions behind each choice — so an operator can see why a specific asset was selected over another. The attribution method is SHAP-inspired rather than a direct SHAP implementation.

### 4. Online Machine Learning Feedback Loop

Implements an inverse-optimization learning policy: when a supervisor overrides or rejects a generated schedule, the system runs an online gradient-descent update over the multi-objective weight vector $W$:

$$W_{new} = \text{clip}\left( W_{old} + \eta \cdot \nabla R, \; 0.1, \; 5.0 \right)$$

This lets the scheduler adapt to implicit human operational preferences over time without full model retraining.

## Repository Structure

```
├── src
│   ├── app.py              # Streamlit frontend & dashboard views
│   ├── engine.py           # Backend core logic
│       ├── HeterogeneousIngestionEngine  # Data normalization layer
│       ├── MultiObjectiveScheduler       # Google OR-Tools solver mapping
│       └── OnlineFeedbackLearningLoop    # Gradient update state machine
├── README.md               # System documentation
└── requirements.txt        # Package dependencies
```

## Quick Start

### Prerequisites
Python 3.9 or higher

### 1. Clone & Install Dependencies
```bash
pip install -r requirements.txt           # After downloading the zip/cloning the repo
```

### 2. Dependencies (`requirements.txt`)
```
ortools>=9.6.0
streamlit>=1.25.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
```

### 3. Launch the Application
```bash
cd src   
streamlit run app.py
```

## Design Notes for Production Deployment
The following notes are relevant for anyone looking to move this from prototype to production.

This prototype currently runs as a Streamlit app with in-memory `st.session_state` storage, suitable for local runs and demos. A few design choices were made with a future production deployment in mind:

- **Deterministic solve time**: The CP-SAT solver is configured with a hard 10-second timeout (`solver.parameters.max_time_in_seconds = 10.0`), giving bounded, predictable execution — a property that would suit serverless/containerized environments (e.g., AWS Fargate, Google Cloud Run) if scaled up in the future.
- **State persistence**: For a real deployment, `st.session_state` would need to be replaced with a persistent store (e.g., PostgreSQL or Redis) to retain weight-configuration history and track system performance over time. This is not yet implemented.
