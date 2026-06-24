#!/usr/bin/env python3
"""Build the frozen eval corpus + queryset (#1). Real corpus, real known answers: every
gold_answer is derivable from the corpus and every gold_chunks [file,idx] is located by running
the sidecar's own chunk() over the source file (so the gold passage provably contains the
supporting string). Deterministic — no randomness."""
import os, sys, json, zipfile
from pathlib import Path
sys.path.insert(0, "server")
from dataroom_service import chunk  # the real chunker the sidecar uses (CHUNK_SIZE/OVERLAP)

ROOT = Path("data/prd/benchmarks/eval")
CORPUS = ROOT / "corpus"
CORPUS.mkdir(parents=True, exist_ok=True)

FILES = {
"company_overview.md": """# Meridian Robotics — Company Overview

Meridian Robotics is a robotics company founded in 2019 in Portland, Oregon. The CEO is Dana
Whitfield. As of 2026 the company employs 140 people across engineering, operations, and sales.

Meridian builds two product lines: the Atlas humanoid robots for general-purpose manipulation,
and the Orion line of autonomous mobile robots (AMRs) for warehouse logistics. The company's
mission is to make general-purpose robots affordable for mid-size manufacturers.

Headquarters is at 410 Industry Way, Portland, Oregon. The company is privately held.
""",
"products/atlas_spec.md": """# Atlas-7 Technical Specification

The Atlas-7 is Meridian's flagship humanoid robot. Key specifications:

- Height: 1.7 meters
- Weight: 62 kilograms
- Battery life: 8 hours of continuous operation
- Payload capacity: 15 kilograms per arm
- Top walking speed: 1.8 meters per second
- Degrees of freedom: 28
- Ingress protection rating: IP54
- Onboard compute: NVIDIA Jetson Orin
- List price: 74,000 US dollars

The Atlas-7 ships with ROS 2 Humble and supports tactile fingertip sensing.
""",
"products/atlas_changelog.md": """# Atlas-7 Release Changelog

- v1.0 — shipped 2021-03-15. First production release.
- v2.0 — shipped 2022-09-01. Added stereo vision and improved balance control.
- v3.0 — shipped 2024-06-20. Added tactile fingertips. This release was owned by the Locomotion team.
- v3.1 — shipped 2025-02-10. Extended battery life from 6 hours to 8 hours.
- v3.2 — shipped 2025-11-05. Migrated the software stack to ROS 2 Humble.

All Atlas-7 releases are validated by the QA team before shipment.
""",
"products/orion_spec.md": """# Orion-2 Technical Specification

The Orion-2 is Meridian's autonomous mobile robot for warehouse logistics.

- Payload capacity: 1200 kilograms
- Battery life: 12 hours
- Navigation: LiDAR combined with visual SLAM (VSLAM)
- Maximum speed: 2.5 meters per second
- Fleet size: up to 50 units coordinated by one fleet manager
- List price: 38,500 US dollars

The Orion-2 runs firmware version 4.2.1 as of the latest release.
""",
"people/team.md": """# Meridian Robotics — Team Directory

- Dana Whitfield — Chief Executive Officer — joined 2019-01.
- Marcus Lin — VP of Engineering — joined 2019-04.
- Priya Nair — Locomotion team lead — joined 2020-07.
- Sofia Reyes — Perception team lead — joined 2021-02.
- Nina Petrov — Finance lead — joined 2022-05.
- Grace Kim — QA team lead — joined 2023-08.
- Raj Patel — Perception engineer — joined 2025-01.
- Tom Becker — Firmware engineer — joined 2025-03.
- Aisha Khan — Firmware engineer — joined 2025-06.
- Liam Foster — Sales representative — joined 2025-09.
- Omar Said — QA engineer — joined 2022-11, left the company 2026-01.

The Locomotion team and the Perception team both report to the VP of Engineering.
""",
"finance/funding.md": """# Meridian Robotics — Funding History

- Seed round (2019): raised 2.5 million US dollars, led by Cascade Ventures.
- Series A (2021-05): raised 18 million US dollars, led by Northwind Capital.
- Series B (2023-10): raised 55 million US dollars, led by Meridian Growth Partners,
  at a post-money valuation of 310 million US dollars.

Total capital raised to date is 75.5 million US dollars. The company has not raised debt.
""",
"ops/incidents.md": """# Operations — Incident Reports

- INC-2041 (2025-04-12): An Atlas-7 arm fault occurred during a customer demo. Error code:
  ERR_TORQUE_OVERFLOW. Root cause: faulty torque-sensor calibration. Resolved by recalibration.
- INC-2047 (2025-07-03): An Orion-2 unit lost navigation in a warehouse. Error code:
  ERR_LIDAR_TIMEOUT. Root cause: a firmware race condition. Fixed in firmware version 4.2.1.
- INC-2052 (2025-10-19): A battery thermal warning triggered on an Atlas-7. Error code:
  ERR_THERM_LIMIT. Root cause: battery cell imbalance. Mitigated by a balancing firmware update.

All incidents are reviewed in the weekly operations meeting.
""",
"policies/security.md": """# Security Policy

All employee access to internal systems is granted through single sign-on (SSO) with mandatory
multi-factor authentication (MFA). Production system access additionally requires connection
through the corporate VPN. Secrets and API keys are stored in HashiCorp Vault, never in source
control. Access reviews are conducted quarterly by the security team.
""",
"policies/data_retention.md": """# Data Retention Policy

- Customer telemetry data is retained for 90 days, then deleted.
- Incident logs are retained for 2 years.
- Database backups are taken daily and retained for 30 days.

Data deletion requests from customers are honored within 30 days under the company privacy policy.
""",
"research/benchmarks.md": """# Internal Benchmark Results

- Atlas-7 grasp success rate: 94.2 percent on the internal manipulation benchmark.
- Orion-2 navigation accuracy: 99.1 percent on the warehouse navigation benchmark.
- Atlas-7 walking energy efficiency: cost of transport of 0.42.

These benchmarks are re-run before every major release.
""",
}

for rel, content in FILES.items():
    p = CORPUS / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

# (id, type, query, gold_answer, [(file, supporting_substring_verbatim_in_file), ...])
QA = [
 # ---- factoid (single clear fact) ----
 ("q01","factoid","What is the battery life of the Atlas-7?","8 hours",[("products/atlas_spec.md","Battery life: 8 hours")]),
 ("q02","factoid","How much does the Atlas-7 cost?","74,000 US dollars",[("products/atlas_spec.md","List price: 74,000 US dollars")]),
 ("q03","factoid","What is the payload capacity of the Orion-2?","1200 kilograms",[("products/orion_spec.md","Payload capacity: 1200 kilograms")]),
 ("q04","factoid","Who is the CEO of Meridian Robotics?","Dana Whitfield",[("company_overview.md","The CEO is Dana")]),
 ("q05","factoid","In what year was Meridian Robotics founded?","2019",[("company_overview.md","founded in 2019")]),
 ("q06","factoid","What onboard compute does the Atlas-7 use?","NVIDIA Jetson Orin",[("products/atlas_spec.md","Onboard compute: NVIDIA Jetson Orin")]),
 ("q07","factoid","How long is customer telemetry data retained?","90 days",[("policies/data_retention.md","retained for 90 days")]),
 ("q08","factoid","What is the Atlas-7 grasp success rate on the internal benchmark?","94.2 percent",[("research/benchmarks.md","grasp success rate: 94.2 percent")]),
 ("q09","factoid","How many degrees of freedom does the Atlas-7 have?","28",[("products/atlas_spec.md","Degrees of freedom: 28")]),
 ("q10","factoid","What is the Orion-2 list price?","38,500 US dollars",[("products/orion_spec.md","List price: 38,500 US dollars")]),
 # ---- literal (exact entity / code / id string) ----
 ("q11","literal","What error code did incident INC-2041 report?","ERR_TORQUE_OVERFLOW",[("ops/incidents.md","ERR_TORQUE_OVERFLOW")]),
 ("q12","literal","Which incident reported error code ERR_LIDAR_TIMEOUT?","INC-2047",[("ops/incidents.md","INC-2047")]),
 ("q13","literal","What firmware version fixed the Orion-2 navigation race condition?","firmware version 4.2.1",[("ops/incidents.md","Fixed in firmware version 4.2.1")]),
 ("q14","literal","Where are Meridian's secrets and API keys stored?","HashiCorp Vault",[("policies/security.md","stored in HashiCorp Vault")]),
 ("q15","literal","What is the error code for the Atlas-7 battery thermal warning incident?","ERR_THERM_LIMIT",[("ops/incidents.md","ERR_THERM_LIMIT")]),
 ("q16","literal","What ingress protection rating does the Atlas-7 have?","IP54",[("products/atlas_spec.md","Ingress protection rating: IP54")]),
 ("q17","literal","Which firm led Meridian's Series B round?","Meridian Growth Partners",[("finance/funding.md","led by Meridian Growth Partners")]),
 ("q18","literal","What is the company headquarters address?","410 Industry Way, Portland, Oregon",[("company_overview.md","410 Industry Way, Portland, Oregon")]),
 # ---- dispersed (aggregate / count across one or more docs) ----
 ("q19","dispersed","How many people joined Meridian in 2025?","4",[("people/team.md","joined 2025-01"),("people/team.md","joined 2025-03"),("people/team.md","joined 2025-06"),("people/team.md","joined 2025-09")]),
 ("q20","dispersed","What is the total capital Meridian has raised to date?","75.5 million US dollars",[("finance/funding.md","Total capital raised to date is 75.5 million US dollars")]),
 ("q21","dispersed","How many Atlas-7 releases have shipped in total?","5",[("products/atlas_changelog.md","v1.0"),("products/atlas_changelog.md","v3.2")]),
 ("q22","dispersed","Which two product lines does Meridian build?","Atlas and Orion",[("company_overview.md","Atlas humanoid robots"),("company_overview.md","Orion line of autonomous mobile robots")]),
 ("q23","dispersed","How many funding rounds has Meridian completed?","3",[("finance/funding.md","Seed round"),("finance/funding.md","Series A"),("finance/funding.md","Series B")]),
 ("q24","dispersed","How long are incident logs retained compared to customer telemetry?","Incident logs 2 years vs telemetry 90 days",[("policies/data_retention.md","Incident logs are retained for 2 years"),("policies/data_retention.md","retained for 90 days")]),
 ("q25","dispersed","How many engineers on the Firmware team joined in 2025?","2",[("people/team.md","Tom Becker — Firmware engineer — joined 2025-03"),("people/team.md","Aisha Khan — Firmware engineer — joined 2025-06")]),
 # ---- multi-hop (chain across docs) ----
 ("q26","multi-hop","Who led the team that shipped Atlas-7 v3.0?","Priya Nair",[("products/atlas_changelog.md","owned by the Locomotion team"),("people/team.md","Priya Nair — Locomotion team lead")]),
 ("q27","multi-hop","What was the post-money valuation at the round that raised 55 million dollars?","310 million US dollars",[("finance/funding.md","raised 55 million US dollars"),("finance/funding.md","valuation of 310 million US dollars")]),
 ("q28","multi-hop","Which firmware version is referenced both by the Orion-2 spec and by incident INC-2047?","4.2.1",[("products/orion_spec.md","firmware version 4.2.1"),("ops/incidents.md","Fixed in firmware version 4.2.1")]),
 ("q29","multi-hop","Who is the team lead that the QA team's manager also manages alongside Locomotion?","Sofia Reyes",[("people/team.md","report to the VP of Engineering"),("people/team.md","Sofia Reyes — Perception team lead")]),
 ("q30","multi-hop","Which Atlas-7 release extended the battery life to the value listed in the current spec?","v3.1",[("products/atlas_spec.md","Battery life: 8 hours"),("products/atlas_changelog.md","Extended battery life from 6 hours to 8 hours")]),
 ("q31","multi-hop","Who joined earlier: the Locomotion team lead or the Perception team lead?","Priya Nair (Locomotion, 2020-07) joined before Sofia Reyes (Perception, 2021-02)",[("people/team.md","Priya Nair — Locomotion team lead — joined 2020-07"),("people/team.md","Sofia Reyes — Perception team lead — joined 2021-02")]),
 ("q32","multi-hop","What error code was reported in the same incident whose root cause was a firmware race condition?","ERR_LIDAR_TIMEOUT",[("ops/incidents.md","ERR_LIDAR_TIMEOUT"),("ops/incidents.md","a firmware race condition")]),
]

# Locate gold_chunks: for each support (file, substr), find the chunk index in that file.
def locate(rel, substr):
    text = (CORPUS / rel).read_text()
    if substr not in text:
        raise SystemExit(f"FATAL: support substring not in {rel}: {substr!r}")
    chunks = chunk(text)  # the real sidecar chunker
    for i, c in enumerate(chunks):
        if substr in c:
            return [rel, i]
    raise SystemExit(f"FATAL: support not in any chunk of {rel}: {substr!r}")

lines = []
seen = set()
for qid, typ, query, gold, supports in QA:
    if qid in seen:
        raise SystemExit(f"FATAL duplicate id {qid}")
    seen.add(qid)
    gold_chunks = [locate(f, s) for f, s in supports]
    lines.append({"id": qid, "query": query, "gold_answer": gold,
                  "gold_chunks": gold_chunks, "type": typ})

out = ROOT / "queryset.jsonl"
out.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
print(f"wrote {out} ({len(lines)} questions)")

# Build the default dataroom zip from the corpus (for #2 smoke / Phase-2 evals).
dz = Path("data/default-dataroom.zip")
with zipfile.ZipFile(dz, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(CORPUS.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(CORPUS))
print(f"wrote {dz} ({dz.stat().st_size} bytes, {sum(1 for p in CORPUS.rglob('*') if p.is_file())} files)")
