#!/bin/bash
# TEST ON A REAL SCAN: run the submission model on a knee MRI from outside the competition and open the report
# (scripts/external_study.py). First put the hospital viewer's DICOM download (zip or folder) in "external scan/dicom/".
# Optional: fill in "external scan/truth.csv" from the radiologist's report and run again to see the model next to it.
# It shows MRI images: for you only, never paste what it shows into a chat.
#
#   start/5_test_real_scan.command                       # "external scan"
#   SCAN="other data" start/5_test_real_scan.command     # another folder with the same layout
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/external_study.py "${SCAN:-external scan}"
