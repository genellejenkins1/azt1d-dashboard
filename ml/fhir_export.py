"""
fhir_export.py
--------------
Serialise one or more AZT1D patients' CGM records into valid FHIR R4 resources.

Produces:
  - A FHIR Patient resource
  - One FHIR Observation per CGM reading (LOINC 15074-8 — glucose [Moles/volume]
    in Blood, converted to mmol/L; also encodes mg/dL in a secondary component)
  - One FHIR Bundle (type = collection) containing all resources
  - Written as a pretty-printed JSON file compatible with any FHIR R4 server

Demonstrates: HL7 FHIR R4, clinical data interoperability, LOINC coding,
healthcare data standards.  Educational only — not PHI.

Install (optional for schema validation):
    pip install fhir.resources

Usage:
    python ml/fhir_export.py                             # subject 1, template
    python ml/fhir_export.py --subject 2 --max-obs 200  # first 200 readings
    python ml/fhir_export.py --out results/fhir/         # write to custom dir
    python ml/fhir_export.py --validate                  # validate with fhir.resources
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FHIR R4 helpers — pure dict construction (no external library required)
# ---------------------------------------------------------------------------

MG_DL_TO_MMOL = 0.0555174   # 1 mg/dL = 0.0555174 mmol/L

# LOINC 15074-8: Glucose [Moles/volume] in Blood (primary FHIR glucose code)
LOINC_GLUCOSE = {
    "system": "http://loinc.org",
    "code": "15074-8",
    "display": "Glucose [Moles/volume] in Blood"
}
# LOINC 2339-0: Glucose [Mass/volume] in Blood (mg/dL secondary)
LOINC_GLUCOSE_MGDL = {
    "system": "http://loinc.org",
    "code": "2339-0",
    "display": "Glucose [Mass/volume] in Blood"
}
UCUM_MMOL = {"system": "http://unitsofmeasure.org", "code": "mmol/L", "unit": "mmol/L"}
UCUM_MGDL = {"system": "http://unitsofmeasure.org", "code": "mg/dL", "unit": "mg/dL"}

# ADA interpretation thresholds
def glucose_interpretation(mg_dl: float) -> dict:
    if mg_dl < 54:
        return {"text": "Critically Low", "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "LL", "display": "Critical low"}]}
    if mg_dl < 70:
        return {"text": "Low", "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "L", "display": "Low"}]}
    if mg_dl <= 180:
        return {"text": "Normal", "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "N", "display": "Normal"}]}
    if mg_dl <= 250:
        return {"text": "High", "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H", "display": "High"}]}
    return {"text": "Critically High", "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "HH", "display": "Critical high"}]}


def fhir_patient(subject_id: int) -> dict:
    """FHIR R4 Patient resource (de-identified demo patient)."""
    return {
        "resourceType": "Patient",
        "id": f"azt1d-patient-{subject_id:03d}",
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"],
            "tag": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
                     "code": "HTEST", "display": "test health data"}]
        },
        "text": {
            "status": "generated",
            "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">De-identified AZT1D study participant {subject_id:03d}</div>"
        },
        "identifier": [{
            "use": "usual",
            "system": "urn:oid:2.16.840.1.113883.4.3.4",   # ASU study namespace (demo)
            "value": f"AZT1D-{subject_id:04d}"
        }],
        "active": True,
        "name": [{"use": "anonymous", "text": f"AZT1D Study Participant {subject_id:03d}"}],
        "gender": "unknown",
        "birthDate": "1990-01-01",    # anonymised
        "extension": [{
            "url": "http://hl7.org/fhir/StructureDefinition/patient-clinicalTrial",
            "valueString": "AZT1D 2025 — ASU Automated Insulin Delivery Study"
        }]
    }


def fhir_device() -> dict:
    """FHIR R4 Device resource for the CGM sensor."""
    return {
        "resourceType": "Device",
        "id": "cgm-sensor-aid",
        "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Device"]},
        "status": "active",
        "manufacturer": "Demo (AID System)",
        "type": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "714749008",
                "display": "Continuous glucose monitoring system"
            }]
        },
        "note": [{"text": "Automated Insulin Delivery (AID) system CGM component. "
                           "Device identity anonymised for portfolio demonstration."}]
    }


def fhir_observation(subject_id: int, reading_id: str,
                     dt: datetime, mg_dl: float) -> dict:
    """FHIR R4 Observation — one CGM glucose reading."""
    mmol = round(mg_dl * MG_DL_TO_MMOL, 2)
    return {
        "resourceType": "Observation",
        "id": reading_id,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]
        },
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory",
                "display": "Laboratory"
            }]
        }],
        "code": {"coding": [LOINC_GLUCOSE], "text": "Blood Glucose (CGM)"},
        "subject": {"reference": f"Patient/azt1d-patient-{subject_id:03d}"},
        "device": {"reference": "Device/cgm-sensor-aid"},
        "effectiveDateTime": dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "issued": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Primary value: mmol/L (FHIR preferred for glucose)
        "valueQuantity": {
            "value": mmol,
            **UCUM_MMOL
        },
        "interpretation": [glucose_interpretation(mg_dl)],
        # Secondary component: mg/dL (common in US clinical practice)
        "component": [{
            "code": {"coding": [LOINC_GLUCOSE_MGDL], "text": "Blood Glucose (mg/dL)"},
            "valueQuantity": {"value": round(mg_dl, 1), **UCUM_MGDL}
        }],
        "referenceRange": [{
            "low": {"value": 3.9, **UCUM_MMOL},     # 70 mg/dL
            "high": {"value": 10.0, **UCUM_MMOL},   # 180 mg/dL
            "text": "ADA target range: 70-180 mg/dL (3.9-10.0 mmol/L)"
        }]
    }


def fhir_bundle(resources: list[dict]) -> dict:
    """FHIR R4 Bundle — collection of all resources."""
    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"]},
        "type": "collection",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(resources),
        "entry": [{"resource": r} for r in resources]
    }


# ---------------------------------------------------------------------------
# Validation (optional — requires fhir.resources)
# ---------------------------------------------------------------------------

def validate_bundle(bundle: dict) -> None:
    try:
        from fhir.resources.bundle import Bundle
        b = Bundle.model_validate(bundle)
        print(f"[FHIR] Validation passed — {b.total} resources in bundle")
    except ImportError:
        print("[FHIR] fhir.resources not installed; skipping schema validation.")
        print("       pip install fhir.resources")
    except Exception as e:
        print(f"[FHIR] Validation error: {e}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cgm(data_root: Path, subject_id: int) -> pd.DataFrame:
    folder = data_root / f"Subject {subject_id}"
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {folder}")
    df = pd.concat([pd.read_csv(c, parse_dates=["EventDateTime"]) for c in csvs],
                   ignore_index=True)
    df = df.dropna(subset=["CGM", "EventDateTime"])
    df["CGM"] = pd.to_numeric(df["CGM"], errors="coerce")
    df = df.dropna(subset=["CGM"])
    df = df.sort_values("EventDateTime").reset_index(drop=True)
    return df


def synthetic_cgm(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-12-01", periods=n, freq="5min", tz="UTC")
    glucose = np.clip(
        145 + 35 * rng.standard_normal(n) + 20 * np.sin(np.linspace(0, 6 * np.pi, n)),
        38, 380
    )
    return pd.DataFrame({"EventDateTime": times, "CGM": glucose})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Export CGM data as FHIR R4 Bundle")
    ap.add_argument("--data", default="sample_data/CGM Records")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--max-obs", type=int, default=500,
                    help="Max CGM readings to export (keeps file manageable)")
    ap.add_argument("--out", default="results/fhir")
    ap.add_argument("--validate", action="store_true",
                    help="Validate bundle with fhir.resources library")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CGM data
    try:
        df = load_cgm(Path(args.data), args.subject)
        print(f"Loaded {len(df)} CGM readings for Subject {args.subject}")
    except FileNotFoundError:
        print("CGM data not found — using synthetic 200-reading demo cohort")
        df = synthetic_cgm(200)
        args.subject = 1

    # Subsample if requested
    if len(df) > args.max_obs:
        df = df.iloc[: args.max_obs]
        print(f"Capped to {args.max_obs} readings (--max-obs)")

    # Build resources
    resources: list[dict] = [fhir_patient(args.subject), fhir_device()]
    for i, row in df.iterrows():
        rid = f"obs-{args.subject:03d}-{i:06d}"
        dt = row["EventDateTime"]
        if not isinstance(dt, datetime):
            dt = pd.Timestamp(dt).to_pydatetime()
        resources.append(fhir_observation(args.subject, rid, dt, float(row["CGM"])))

    bundle = fhir_bundle(resources)

    # Write
    out_path = out_dir / f"subject_{args.subject:03d}_cgm_bundle.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Bundle written: {out_path}")
    print(f"  Resources: 1 Patient + 1 Device + {len(df)} Observations = {len(resources)} total")
    print(f"  FHIR version: R4")
    print(f"  Glucose coding: LOINC 15074-8 (mmol/L primary, mg/dL component)")

    if args.validate:
        validate_bundle(bundle)

    # Print a sample observation for reference
    print("\n--- Sample Observation (first reading) ---")
    print(json.dumps(resources[2], indent=2)[:800] + "\n...")


if __name__ == "__main__":
    main()
