# Task B — Synthetic Data Generation (10%)

**Domain:** Healthcare — Emergency Department (Colombia)  
**Tool:** `EDDataGenerator` — generates realistic pseudorandom patient encounter records

---

## What it generates

**36 columns × variable row count** (default: 2,000 rows)

| Category | Columns |
|---|---|
| Keys | `patient_id`, `encounter_id` |
| Temporal | `encounter_date`, `encounter_time` |
| Demographics | `age`, `sex`, `department_of_origin`, `blood_type`, `weight_kg`, `height_cm`, `bmi` |
| Triage | `triage_level`, `triage_description`, `chief_complaint` |
| Diagnosis | `icd10_code`, `icd10_description`, `severity_score` |
| Vital signs | `systolic_bp`, `diastolic_bp`, `heart_rate`, `respiratory_rate`, `temperature_c`, `spo2_pct`, `pain_scale_0_10` |
| Labs | `glucose_mgdl`, `creatinine_mgdl`, `hemoglobin_gdl`, `wbc_thousand_ul` |
| Operations | `triage_to_md_min`, `length_of_stay_hrs`, `primary_medication` |
| Administrative | `insurance_type`, `billing_amount_cop` |
| Outcomes | `readmission_30d`, `mortality_flag`, `disposition` |

---

## Distributions used

| Variable | Distribution | Justification |
|---|---|---|
| Age | Mixture of Gaussians (bimodal) | Young adults (accidents) + elderly (chronic disease) |
| Systolic BP | Mixture Normal | Normotensive pop. + hypertensive subgroup |
| Weight | Log-Normal | Right-skewed: few very heavy patients |
| Glucose | Gamma | Positive, right-skewed; diabetics shifted |
| Creatinine, WBC | Log-Normal | Strictly positive, heavy right tail |
| Severity, Pain, SpO2 | Beta (scaled) | Bounded variable with realistic mode |
| Billing | Pareto (power-law) | Most bills small, few catastrophically large |
| Diagnoses, Medications | Zipf | Rank-frequency: few categories dominate |
| Triage wait | Exponential | Memoryless arrival process |
| Length of stay | Log-Normal | Short for most, very long for few (ICU) |
| Mortality, Readmission | Logistic Bernoulli | Binary outcome conditioned on age + severity |

---

## Structure

```
task_b_healthcare/
├── generator.py       ← Main script / EDDataGenerator class
├── data/
│   ├── ed_patients.csv    ← generated on first run
│   └── ed_patients.json
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate 2,000 records
python generator.py --rows 2000

# Generate + Expert Check via Anthropic API
export ANTHROPIC_API_KEY="your-key-here"
python generator.py --rows 2000 --check

# Custom seed and size
python generator.py --rows 5000 --seed 99 --check
```

---

## Expert Check

The `--check` flag calls **Claude claude-sonnet-4-6** acting as *Dr. Alejandro Reyes*,
a senior emergency medicine physician from Hospital Universitario San Ignacio (Bogotá).
The AI evaluates the dataset summary for clinical realism and returns a score 0–100
with specific corrections.

Requires: `ANTHROPIC_API_KEY` environment variable.

---

## Realism Targets (MSPS Colombia 2022)

| Metric | Generated | Target |
|---|---|---|
| Mortality rate | ~1.5% | 1.5–3% |
| 30-day readmission | ~17% | 10–18% |
| Admitted to ward | ~22% | 20–35% |
| Discharged home | ~65% | 55–70% |
| Triage Level 3 (Urgent) | ~39% | most common |
