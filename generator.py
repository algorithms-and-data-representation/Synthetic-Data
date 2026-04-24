"""
Task B — Synthetic Data Generation (10%)
==========================================
Domain  : Healthcare — Emergency Department (Colombia)
Tool    : EDDataGenerator — generates realistic pseudorandom patient records

Distributions used (justification inside each method)
------------------------------------------------------
  Mixture of Gaussians  → age (bimodal: young adults + elderly)
  Truncated Normal      → vitals (physiologically bounded)
  Log-Normal            → creatinine, WBC, length of stay (right-skewed)
  Gamma                 → glucose, triage wait time (positive, skewed)
  Beta (scaled)         → severity score, pain scale, SpO2 (bounded)
  Pareto                → billing amount (power-law tail)
  Zipf                  → diagnoses, medications, complaints (rank-frequency)
  Bernoulli (logistic)  → mortality, readmission (conditioned on covariates)
  Exponential           → triage-to-doctor wait (memoryless arrival process)
  Bivariate correlation → BP systolic ↔ diastolic, weight ↔ BMI

References
----------
  - MSPS Colombia (2022). Indicadores hospitalarios de urgencias.
  - SGSSS (2023). Estadísticas de aseguradoras EPS.
  - Harrison's Principles of Internal Medicine, 21st ed.
"""

import csv
import json
import math
import os
import random
import datetime
import argparse
import statistics

import anthropic   # for Expert Check


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def truncated_normal(mu, sigma, lo, hi, rng):
    """Normal(mu,sigma) rejection-sampled to [lo, hi]. O(1) average."""
    for _ in range(200):
        v = rng.gauss(mu, sigma)
        if lo <= v <= hi:
            return v
    return max(lo, min(hi, mu))          # fallback if very tight bounds


def log_normal(mu_log, sigma_log, rng):
    """X = exp(N(mu_log, sigma_log)).  E[X] = exp(mu + σ²/2).
    Good for: creatinine, WBC, billing, LOS — strictly positive, right-skewed."""
    return math.exp(rng.gauss(mu_log, sigma_log))


def gamma_sample(shape, scale, rng):
    """Gamma(shape, scale).  E[X] = shape*scale.  Skewness = 2/sqrt(shape).
    Good for: glucose, wait times — positive, unimodal, right tail."""
    return rng.gammavariate(shape, scale)


def beta_scaled(alpha, beta_p, lo, hi, rng):
    """Beta(alpha,beta_p) rescaled to [lo,hi].
    Good for: severity score, pain scale, SpO2 — strictly bounded."""
    return lo + rng.betavariate(alpha, beta_p) * (hi - lo)


def pareto_sample(alpha, x_min, rng):
    """Pareto: P(X>x) = (x_min/x)^alpha.  Heavy right tail.
    Good for: billing amounts — most are small, a few are enormous."""
    u = rng.random()
    return x_min / (u ** (1.0 / alpha))


def mixture_normal(components, rng):
    """Gaussian mixture [(weight, mu, sigma), ...].
    Good for: age — bimodal distribution (young adults + elderly)."""
    total = sum(w for w, _, _ in components)
    r = rng.random() * total
    cumulative = 0.0
    for w, mu, sigma in components:
        cumulative += w
        if r <= cumulative:
            return rng.gauss(mu, sigma)
    return rng.gauss(components[-1][1], components[-1][2])


def zipf_index(n, exponent, rng):
    """Zipf/power-law rank sampling over n items.  P(rank k) ∝ 1/k^exponent.
    Good for: diagnoses, medications — few categories dominate."""
    weights = [1.0 / (k ** exponent) for k in range(1, n + 1)]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return n - 1


def logistic_bernoulli(log_odds, rng):
    """Bernoulli with logistic probability from log-odds.
    Good for: mortality, readmission — binary outcome conditioned on covariates."""
    p = 1.0 / (1.0 + math.exp(-log_odds))
    return int(rng.random() < p)


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP TABLES  (calibrated against MSPS / Harrison's)
# ─────────────────────────────────────────────────────────────────────────────

ICD10 = [
    ("J06.9", "Acute upper respiratory infection"),
    ("R51",   "Headache"),
    ("M54.5", "Low back pain"),
    ("R07.9", "Chest pain, unspecified"),
    ("I10",   "Essential hypertension"),
    ("E11.9", "Type 2 diabetes mellitus"),
    ("K59.0", "Constipation"),
    ("S09.90", "Head injury, unspecified"),
    ("J18.9", "Pneumonia, unspecified"),
    ("N39.0", "Urinary tract infection"),
    ("R55",   "Syncope and collapse"),
    ("I21.9", "Acute myocardial infarction"),
    ("A09",   "Gastroenteritis"),
    ("F32.9", "Major depressive episode"),
    ("G43.909","Migraine, unspecified"),
    ("K37",   "Appendicitis, unspecified"),
    ("T14.90", "Injury, unspecified"),
    ("J44.1", "COPD with acute exacerbation"),
    ("I63.9", "Cerebral infarction, unspecified"),
    ("Z00.00", "General medical examination"),
]

COMPLAINTS = [
    "Chest pain", "Shortness of breath", "Abdominal pain", "Headache",
    "Back pain", "Fever", "Nausea/vomiting", "Dizziness", "Trauma/injury",
    "Syncope", "Palpitations", "Urinary symptoms", "Altered mental status",
    "Laceration", "Fracture/fall", "Allergic reaction", "Seizure",
    "Eye pain/redness", "Ear pain", "Anxiety/panic",
]

MEDICATIONS = [
    "Acetaminophen 1g IV", "Ketorolac 30mg IV", "Ondansetron 4mg IV",
    "Metoclopramide 10mg IV", "Morphine 4mg IV", "Ibuprofen 400mg PO",
    "Omeprazole 40mg IV", "Amoxicillin 500mg PO", "Ceftriaxone 1g IV",
    "Salbutamol nebulized", "Lorazepam 2mg IV", "Metoprolol 25mg PO",
    "Aspirin 300mg PO", "Enoxaparin 60mg SC", "Furosemide 40mg IV",
]

DISPOSITIONS = [
    "Discharged home", "Admitted to ward", "Admitted to ICU",
    "Transfer to another facility", "Left without being seen",
    "Left against medical advice", "Deceased in ED",
]

EPS_OPTIONS = [
    "EPS Sura", "EPS Sanitas", "Nueva EPS", "EPS Compensar",
    "EPS Famisanar", "Sisben C1", "Sisben C2", "Particular",
    "ARL (occupational)", "Medicina prepagada",
]

DEPARTMENTS_COL = [
    "Cundinamarca", "Antioquia", "Valle del Cauca", "Atlántico",
    "Santander", "Bolívar", "Nariño", "Boyacá", "Cauca", "Huila",
    "Tolima", "Risaralda", "Caldas", "Meta", "Córdoba",
]

TRIAGE_NAMES = {
    1: "Resuscitation", 2: "Emergent", 3: "Urgent",
    4: "Semi-urgent",   5: "Non-urgent",
}

BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
BLOOD_FREQ  = [0.38, 0.07, 0.30, 0.06, 0.09, 0.02, 0.03, 0.02, 0.01, 0.02]


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class EDDataGenerator:
    """
    Emergency Department Synthetic Patient Record Generator.

    Parameters
    ----------
    n_rows  : int   — number of encounters to generate
    seed    : int   — random seed for reproducibility
    start   : str   — earliest encounter date  (YYYY-MM-DD)
    end     : str   — latest encounter date    (YYYY-MM-DD)

    Usage
    -----
    gen = EDDataGenerator(n_rows=2000, seed=42)
    records = gen.generate()
    gen.to_csv(records, "data/ed_patients.csv")
    gen.schema_report()
    gen.summary_stats(records)
    """

    # 36 columns — documented schema
    SCHEMA = [
        ("patient_id",            "Unique patient identifier (UUID-style)"),
        ("encounter_id",          "Unique encounter key per visit"),
        ("encounter_date",        "Date of ED visit (YYYY-MM-DD)"),
        ("encounter_time",        "Hour of arrival (HH:MM)  — bimodal peaks 10h & 19h"),
        ("age",                   "Age in years — Mixture of Gaussians (bimodal)"),
        ("sex",                   "Biological sex — Bernoulli(0.51 female)"),
        ("department_of_origin",  "Colombian department — Zipf"),
        ("blood_type",            "ABO + Rh — population-frequency weighted"),
        ("weight_kg",             "Body weight kg — Log-Normal (right-skewed)"),
        ("height_cm",             "Height cm — Truncated Normal"),
        ("bmi",                   "Body mass index = weight / (height/100)^2"),
        ("triage_level",          "ESI 1-5 — Zipf (most are level 3)"),
        ("triage_description",    "Text label for triage level"),
        ("chief_complaint",       "Primary reason for visit — Zipf"),
        ("icd10_code",            "Primary diagnosis code — Zipf"),
        ("icd10_description",     "Diagnosis text"),
        ("severity_score",        "Internal severity 0-10 — Beta(2,3) scaled"),
        ("systolic_bp",           "Systolic BP mmHg — Mixture Normal (hypertensive)"),
        ("diastolic_bp",          "Diastolic BP mmHg — correlated with systolic"),
        ("heart_rate",            "Heart rate bpm — Beta scaled [40,160]"),
        ("respiratory_rate",      "Resp rate /min — Truncated Normal, ↑ with severity"),
        ("temperature_c",         "Body temp °C — Truncated Normal; fever if complaint"),
        ("spo2_pct",              "O2 saturation % — Beta scaled, ↓ with severity"),
        ("pain_scale_0_10",       "Self-reported pain — Beta(3,2) scaled [0,10]"),
        ("glucose_mgdl",          "Blood glucose mg/dL — Gamma (diabetics shifted)"),
        ("creatinine_mgdl",       "Creatinine mg/dL — Log-Normal, ↑ with age"),
        ("hemoglobin_gdl",        "Hemoglobin g/dL — Truncated Normal, sex-dependent"),
        ("wbc_thousand_ul",       "WBC count k/µL — Log-Normal, ↑ with severity"),
        ("triage_to_md_min",      "Wait for physician min — Exponential(λ∝triage)"),
        ("length_of_stay_hrs",    "Total ED stay hrs — Log-Normal, ↑ if admitted"),
        ("primary_medication",    "Main drug administered — Zipf"),
        ("insurance_type",        "Health coverage — categorical"),
        ("billing_amount_cop",    "Billed amount COP — Pareto(alpha=1.5, power-law)"),
        ("readmission_30d",       "Readmission in 30 days — logistic Bernoulli"),
        ("mortality_flag",        "ED/in-hospital death — logistic(age, severity)"),
        ("disposition",           "ED outcome — Zipf"),
    ]

    def __init__(self, n_rows=1000, seed=42,
                 start="2020-01-01", end="2024-12-31"):
        self.n_rows = n_rows
        self.rng    = random.Random(seed)
        self._start = datetime.date.fromisoformat(start)
        self._range = (datetime.date.fromisoformat(end) - self._start).days
        self._pid_pool = self._make_pid_pool(max(n_rows * 3, 10_000))

    # ── internal helpers ──────────────────────────────────────────────

    def _make_pid_pool(self, size):
        pool = set()
        while len(pool) < size:
            pool.add(f"PT{self.rng.randint(1_000_000, 9_999_999)}")
        return list(pool)

    def _encounter_id(self, pid, date_str, seq):
        return f"EC{pid[2:]}{date_str.replace('-','')}{seq:03d}"

    def _random_date(self):
        return (self._start +
                datetime.timedelta(days=self.rng.randint(0, self._range))
                ).isoformat()

    def _random_time(self):
        """Bimodal: peaks at 10h (morning) and 19h (evening)."""
        hour = int(mixture_normal(
            [(0.45, 10, 2.5), (0.40, 19, 2.0), (0.15, 2, 1.5)],
            self.rng
        ))
        hour = max(0, min(23, hour))
        return f"{hour:02d}:{self.rng.randint(0,59):02d}"

    # ── column generators ─────────────────────────────────────────────

    def _age(self):
        """Bimodal mixture: 40% young adults (mean 32), 60% elderly (mean 67).
        Reflects typical ED burden: accidents + chronic disease."""
        raw = mixture_normal([(0.40, 32, 12), (0.60, 67, 13)], self.rng)
        return max(0, min(110, int(raw)))

    def _weight(self, age, sex):
        """Log-Normal: realistic right skew (few very heavy patients).
        mu_log calibrated so median ≈ 70 kg (men) / 63 kg (women)."""
        mu = math.log(73 if sex == "Male" else 65)
        return round(max(30.0, min(200.0, log_normal(mu, 0.18, self.rng))), 1)

    def _height(self, sex):
        """Truncated Normal.  Colombian DANE averages: 170cm (M), 158cm (F)."""
        if sex == "Male":
            return round(truncated_normal(170, 7, 145, 200, self.rng), 1)
        return round(truncated_normal(158, 6, 140, 185, self.rng), 1)

    def _severity(self):
        """Beta(2,3) → mode ≈ 0.25 (most patients mild).  Scaled to [0,10]."""
        return round(beta_scaled(2, 3, 0, 10, self.rng), 1)

    def _systolic_bp(self, age):
        """Mixture: 65% normotensive N(118,12) + 35% hypertensive N(155,18).
        Hypertensive weight grows with age (cardiovascular burden)."""
        w_hyp  = min(0.75, 0.10 + (age - 18) * 0.007)
        w_norm = 1 - w_hyp
        sbp = mixture_normal(
            [(w_norm, 118, 12), (w_hyp, 155, 18)], self.rng)
        return round(max(60, min(250, sbp)))

    def _diastolic_bp(self, sbp):
        """Correlated with SBP: DBP ≈ 0.6*SBP ± noise.
        Maintains physiological pulse pressure ~40 mmHg."""
        return round(truncated_normal(sbp * 0.60, 8, 35, 130, self.rng))

    def _heart_rate(self):
        """Beta(3,4) scaled to [40,160].  Mode ≈ 76 bpm."""
        return round(beta_scaled(3, 4, 40, 160, self.rng))

    def _respiratory_rate(self, severity):
        """Truncated Normal.  Mean = 16 + 0.5*severity (tachypnea with illness)."""
        return round(truncated_normal(16 + 0.5 * severity, 3, 8, 40, self.rng), 1)

    def _temperature(self, complaint):
        """Fever complaints → Truncated Normal shifted to 38.8°C.
        Otherwise normal body temperature ≈ 36.9°C."""
        if "Fever" in complaint or "fever" in complaint:
            return round(truncated_normal(38.8, 0.7, 37.5, 41.5, self.rng), 1)
        return round(truncated_normal(36.9, 0.4, 35.0, 38.5, self.rng), 1)

    def _spo2(self, severity):
        """Beta scaled: healthy ≈ 98%, declines with severity.
        Bounds [70,100] — physiologically impossible below 70."""
        mu = 98.0 - severity * 0.5
        return round(min(100.0, max(70.0,
            truncated_normal(mu, 2, 70, 100, self.rng))), 1)

    def _pain(self, complaint):
        """Beta(4,2) for pain complaints (bias toward moderate-high pain).
        Beta(2,3) otherwise (bias toward low pain)."""
        if any(k in complaint for k in ["pain", "Pain", "ache"]):
            return round(beta_scaled(4, 2, 2, 10, self.rng), 1)
        return round(beta_scaled(2, 3, 0, 8, self.rng), 1)

    def _glucose(self, icd10_code):
        """Gamma(shape=4, scale=25) → mean=100 for general patients.
        Diabetics (E11) get Gamma(5,42) → mean≈210 mg/dL."""
        if "E11" in icd10_code:
            return round(gamma_sample(5, 42, self.rng))
        return round(gamma_sample(4, 25, self.rng))

    def _creatinine(self, age):
        """Log-Normal with mu_log increasing with age (renal decline).
        Expected range: 0.6–1.2 (normal) to >2.0 (renal failure)."""
        mu = 0.05 + age * 0.003
        return round(max(0.3, log_normal(mu, 0.35, self.rng)), 2)

    def _hemoglobin(self, sex):
        """Truncated Normal, sex-dependent.
        Males: mean 14.5, Females: mean 12.8 g/dL  (WHO reference)."""
        if sex == "Male":
            return round(truncated_normal(14.5, 1.5, 5, 20, self.rng), 1)
        return round(truncated_normal(12.8, 1.4, 5, 20, self.rng), 1)

    def _wbc(self, severity):
        """Log-Normal WBC.  Infection/inflammation → higher count.
        Normal: ~7.5 k/µL.  Infected: ~14+ k/µL."""
        mu = math.log(max(7.5 + severity * 0.5, 1))
        return round(max(1.0, log_normal(mu, 0.4, self.rng)), 2)

    def _triage_wait(self, triage_level):
        """Exponential (memoryless arrival process).
        Higher urgency → shorter mean wait (level 1: 2 min, level 5: 55 min)."""
        means = {1: 2, 2: 8, 3: 20, 4: 35, 5: 55}
        return round(self.rng.expovariate(
            1.0 / means.get(triage_level, 25)), 1)

    def _los(self, disposition):
        """Log-Normal length of stay.
        ICU admissions: ~24h median.  Ward: ~10h.  Discharge: ~2-3h."""
        if "ICU" in disposition:
            return round(max(1.0, log_normal(2.8, 0.6, self.rng)), 1)
        if "Admitted" in disposition:
            return round(max(1.0, log_normal(2.1, 0.5, self.rng)), 1)
        return round(max(0.2, log_normal(0.9, 0.6, self.rng)), 1)

    def _billing(self, disposition):
        """Pareto(alpha=1.5, x_min=50k COP).  Power-law tail:
        most bills small, a few catastrophically large (ICU, surgery)."""
        x_min = 300_000 if "ICU" in disposition else 50_000
        return round(min(pareto_sample(1.5, x_min, self.rng), 60_000_000))

    def _readmission(self, age, severity):
        """Logistic Bernoulli: older + sicker → higher readmission risk.
        Calibrated for overall rate ≈ 10-15% (realistic for Colombia)."""
        log_odds = 0.03 * (age - 50) + 0.25 * severity - 2.8
        return logistic_bernoulli(log_odds, self.rng)

    def _mortality(self, age, severity):
        """Logistic Bernoulli: conditioned on age and severity.
        Calibrated for ED mortality ≈ 1.5-3%."""
        log_odds = 0.04 * (age - 60) + 0.40 * severity - 6.0
        return logistic_bernoulli(log_odds, self.rng)

    # ── main generation ───────────────────────────────────────────────

    def generate(self):
        """Generate n_rows encounter records. Returns list of dicts."""
        records = []
        enc_counter = {}

        for _ in range(self.n_rows):
            pid      = self.rng.choice(self._pid_pool)
            date_str = self._random_date()
            seq      = enc_counter.get(pid, 0) + 1
            enc_counter[pid] = seq
            eid      = self._encounter_id(pid, date_str, seq)

            # independent categorical fields
            age        = self._age()
            sex        = self.rng.choice(["Male", "Female"])
            dept       = DEPARTMENTS_COL[zipf_index(len(DEPARTMENTS_COL), 1.3, self.rng)]
            blood      = self.rng.choices(BLOOD_TYPES, weights=[38,7,30,6,9,2,3,2,1,2][:len(BLOOD_TYPES)])[0]
            # Realistic Colombian ED triage (MSPS 2022):
            # Level 3 most common (~40%), Level 1 rarest (~2%)
            triage     = self.rng.choices(
                [1, 2, 3, 4, 5], weights=[2, 12, 40, 35, 11])[0]
            complaint  = COMPLAINTS[zipf_index(len(COMPLAINTS), 1.6, self.rng)]
            diag_idx   = zipf_index(len(ICD10), 1.5, self.rng)
            icd_code, icd_desc = ICD10[diag_idx]
            disp_idx   = zipf_index(len(DISPOSITIONS), 2.0, self.rng)
            disposition = DISPOSITIONS[disp_idx]
            med        = MEDICATIONS[zipf_index(len(MEDICATIONS), 1.7, self.rng)]
            insurance  = self.rng.choice(EPS_OPTIONS)

            # derived / correlated fields
            severity   = self._severity()
            weight     = self._weight(age, sex)
            height     = self._height(sex)
            bmi        = round(weight / (height / 100) ** 2, 1)
            sbp        = self._systolic_bp(age)
            dbp        = self._diastolic_bp(sbp)
            hr         = self._heart_rate()
            rr         = self._respiratory_rate(severity)
            temp       = self._temperature(complaint)
            spo2       = self._spo2(severity)
            pain       = self._pain(complaint)
            glucose    = self._glucose(icd_code)
            creatinine = self._creatinine(age)
            hemoglobin = self._hemoglobin(sex)
            wbc        = self._wbc(severity)
            wait       = self._triage_wait(triage)
            los        = self._los(disposition)
            billing    = self._billing(disposition)
            readmit    = self._readmission(age, severity)
            mortality  = self._mortality(age, severity)

            records.append({
                "patient_id":           pid,
                "encounter_id":         eid,
                "encounter_date":       date_str,
                "encounter_time":       self._random_time(),
                "age":                  age,
                "sex":                  sex,
                "department_of_origin": dept,
                "blood_type":           blood,
                "weight_kg":            weight,
                "height_cm":            height,
                "bmi":                  bmi,
                "triage_level":         triage,
                "triage_description":   TRIAGE_NAMES[triage],
                "chief_complaint":      complaint,
                "icd10_code":           icd_code,
                "icd10_description":    icd_desc,
                "severity_score":       severity,
                "systolic_bp":          sbp,
                "diastolic_bp":         dbp,
                "heart_rate":           hr,
                "respiratory_rate":     rr,
                "temperature_c":        temp,
                "spo2_pct":             spo2,
                "pain_scale_0_10":      pain,
                "glucose_mgdl":         glucose,
                "creatinine_mgdl":      creatinine,
                "hemoglobin_gdl":       hemoglobin,
                "wbc_thousand_ul":      wbc,
                "triage_to_md_min":     wait,
                "length_of_stay_hrs":   los,
                "primary_medication":   med,
                "insurance_type":       insurance,
                "billing_amount_cop":   billing,
                "readmission_30d":      readmit,
                "mortality_flag":       mortality,
                "disposition":          disposition,
            })

        return records

    # ── export ────────────────────────────────────────────────────────

    def to_csv(self, records, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"✅ CSV saved → {path}  ({len(records):,} rows × {len(records[0])} cols)")

    def to_json(self, records, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON saved → {path}")

    def schema_report(self):
        print("\n" + "=" * 72)
        print(f"SCHEMA — EDDataGenerator   ({len(self.SCHEMA)} columns)")
        print("=" * 72)
        for i, (col, desc) in enumerate(self.SCHEMA, 1):
            print(f"  {i:>2}. {col:<28} {desc}")
        print("=" * 72)

    def summary_stats(self, records):
        """Print numeric summary and realism checks."""
        numeric = ["age", "severity_score", "systolic_bp", "diastolic_bp",
                   "heart_rate", "glucose_mgdl", "creatinine_mgdl",
                   "length_of_stay_hrs", "billing_amount_cop"]
        print("\n=== Numeric Summary ===")
        print(f"{'Column':<24} {'min':>8} {'mean':>10} {'median':>10} {'max':>10}")
        print("-" * 66)
        for col in numeric:
            vals = [r[col] for r in records]
            print(f"{col:<24} {min(vals):>8.1f} {statistics.mean(vals):>10.1f} "
                  f"{statistics.median(vals):>10.1f} {max(vals):>10.1f}")

        print("\n=== Realism Checks ===")
        n = len(records)
        mortality_rate = sum(r["mortality_flag"] for r in records) / n * 100
        readmit_rate   = sum(r["readmission_30d"] for r in records) / n * 100
        pct_admitted   = sum("Admitted" in r["disposition"] for r in records) / n * 100
        pct_discharge  = sum("Discharged" in r["disposition"] for r in records) / n * 100
        print(f"  Mortality rate:      {mortality_rate:.2f}%  (target 1.5–3%)")
        print(f"  30-day readmission:  {readmit_rate:.2f}%  (target 10–18%)")
        print(f"  Admitted:            {pct_admitted:.1f}%  (target 20–35%)")
        print(f"  Discharged:          {pct_discharge:.1f}%  (target 55–70%)")

        print("\n=== Triage Distribution ===")
        from collections import Counter
        triage_counts = Counter(r["triage_level"] for r in records)
        for lvl in sorted(triage_counts):
            pct = triage_counts[lvl] / n * 100
            print(f"  Level {lvl} ({TRIAGE_NAMES[lvl]:<16}) "
                  f"{triage_counts[lvl]:>5}  ({pct:4.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT CHECK  — Anthropic API
# ─────────────────────────────────────────────────────────────────────────────

EXPERT_SYSTEM_PROMPT = """You are Dr. Alejandro Reyes, a senior emergency medicine physician
with 20 years of experience at Hospital Universitario San Ignacio in Bogotá, Colombia.
You also hold a Master's in Clinical Epidemiology and have published research on
Colombian emergency department utilization patterns and MSPS statistics.

Your task: evaluate whether a synthetic ED dataset looks clinically and
epidemiologically realistic for a Colombian urban hospital. Be specific.
Point out any values, distributions, or correlations that seem off, and
suggest concrete corrections. Give an overall score from 0-100."""


def expert_check(records, n_sample=200):
    """
    Send a sample of records to Anthropic Claude acting as Dr. Alejandro Reyes
    and ask if the data looks real.

    Returns the expert's evaluation as a string.
    """
    import statistics as st
    n = len(records)

    # Build a concise statistical summary for the expert
    def pct(col, val=1):
        return sum(1 for r in records if r[col] == val) / n * 100

    summary = f"""
DATASET SUMMARY  ({n:,} patient encounters, Colombian ED 2020-2024)

DEMOGRAPHICS
  Age:        mean {st.mean(r['age'] for r in records):.1f}, median {st.median(r['age'] for r in records):.0f}, range {min(r['age'] for r in records)}–{max(r['age'] for r in records)}
  Sex:        {sum(r['sex']=='Female' for r in records)/n*100:.1f}% female

TRIAGE
  Level 1: {sum(r['triage_level']==1 for r in records)/n*100:.1f}%  (Resuscitation)
  Level 2: {sum(r['triage_level']==2 for r in records)/n*100:.1f}%  (Emergent)
  Level 3: {sum(r['triage_level']==3 for r in records)/n*100:.1f}%  (Urgent)
  Level 4: {sum(r['triage_level']==4 for r in records)/n*100:.1f}%  (Semi-urgent)
  Level 5: {sum(r['triage_level']==5 for r in records)/n*100:.1f}%  (Non-urgent)

VITAL SIGNS (means)
  Systolic BP:  {st.mean(r['systolic_bp'] for r in records):.0f} mmHg
  Diastolic BP: {st.mean(r['diastolic_bp'] for r in records):.0f} mmHg
  Heart rate:   {st.mean(r['heart_rate'] for r in records):.0f} bpm
  Resp rate:    {st.mean(r['respiratory_rate'] for r in records):.1f} /min
  Temperature:  {st.mean(r['temperature_c'] for r in records):.2f} °C
  SpO2:         {st.mean(r['spo2_pct'] for r in records):.1f}%

LABS (means)
  Glucose:      {st.mean(r['glucose_mgdl'] for r in records):.0f} mg/dL
  Creatinine:   {st.mean(r['creatinine_mgdl'] for r in records):.2f} mg/dL
  Hemoglobin:   {st.mean(r['hemoglobin_gdl'] for r in records):.1f} g/dL
  WBC:          {st.mean(r['wbc_thousand_ul'] for r in records):.1f} k/µL

OUTCOMES
  Mortality:         {pct('mortality_flag'):.2f}%
  30-day readmission:{pct('readmission_30d'):.1f}%
  Admitted to ward:  {sum('Admitted to ward' in r['disposition'] for r in records)/n*100:.1f}%
  Admitted to ICU:   {sum('Admitted to ICU' in r['disposition'] for r in records)/n*100:.1f}%
  Discharged home:   {sum('Discharged home' in r['disposition'] for r in records)/n*100:.1f}%

OPERATIONS
  Median triage-to-MD wait: {st.median(r['triage_to_md_min'] for r in records):.0f} min
  Median length of stay:    {st.median(r['length_of_stay_hrs'] for r in records):.1f} hrs
  Median billing (COP):     {st.median(r['billing_amount_cop'] for r in records):,.0f}

TOP 3 CHIEF COMPLAINTS
{chr(10).join(f'  {k}: {v/n*100:.1f}%' for k,v in sorted({r['chief_complaint']:sum(1 for x in records if x['chief_complaint']==r['chief_complaint']) for r in records}.items(), key=lambda x:-x[1])[:3])}

TOP 3 ICD-10 DIAGNOSES
{chr(10).join(f'  {k}: {v/n*100:.1f}%' for k,v in sorted({r['icd10_code']:sum(1 for x in records if x['icd10_code']==r['icd10_code']) for r in records}.items(), key=lambda x:-x[1])[:3])}
"""

    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment
    print("\n" + "=" * 68)
    print("EXPERT CHECK — Dr. Alejandro Reyes, Emergency Medicine (Bogotá)")
    print("=" * 68)
    print("Sending dataset summary to AI expert for evaluation …\n")

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=EXPERT_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Please evaluate this synthetic Colombian ED dataset.\n\n"
                    f"{summary}\n\n"
                    "For each section: (1) Does it look realistic? "
                    "(2) What is wrong or unusual? "
                    "(3) What should be corrected? "
                    "End with an overall realism score 0-100 and "
                    "the top 3 most important corrections needed."
                )
            }]
        )
    except Exception as e:
        print(f"❌ Expert Check failed: {e}")
        print("\n💡 To enable Expert Check, set your API key:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("   Then run: python generator.py --rows 2000 --check")
        return ""

    evaluation = message.content[0].text
    print(evaluation)
    print("=" * 68)
    return evaluation


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALDA 2026 — Task B: ED Synthetic Data Generator")
    parser.add_argument("--rows",  type=int, default=2000)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="Run Expert Check via Anthropic API")
    args = parser.parse_args()

    print("=" * 60)
    print("ALDA 2026 — Task B: Healthcare Synthetic Data Generator")
    print(f"  Domain : Emergency Department (Colombia)")
    print(f"  Rows   : {args.rows:,}")
    print(f"  Seed   : {args.seed}")
    print("=" * 60)

    gen = EDDataGenerator(n_rows=args.rows, seed=args.seed)
    gen.schema_report()

    print(f"\nGenerating {args.rows:,} records …")
    records = gen.generate()
    print(f"✅ Generated {len(records):,} records with {len(records[0])} columns.")

    gen.summary_stats(records)

    os.makedirs("data", exist_ok=True)
    gen.to_csv(records,  "data/ed_patients.csv")
    gen.to_json(records, "data/ed_patients.json")

    if args.check:
        expert_check(records)
    else:
        print("\n💡 Run with --check to get Expert Check via Anthropic API")
        print("   python generator.py --rows 2000 --check")
