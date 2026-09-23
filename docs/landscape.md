# Clinical AI landscape and open resources (researched 2026-09-22)

## What radiologists actually use
- As of March 2026 the FDA had authorized 1,524 AI devices, **76% of them in radiology** (1,164). GE, Siemens, Philips and Canon lead; Aidoc (33) is the largest pure-AI vendor. Sources: [Imaging Wire](https://theimagingwire.com/2026/03/11/numbers-from-the-fda-show-radiology-is-maintaining-its-lead/), [FDA list](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices?SiteId=DP_Other).
- Adoption: 48% of European radiologists used AI in 2024. Only 23% of emergency radiologists use it daily.
- Product types:
  - **Triage** (Aidoc: flags brain bleeds / pulmonary embolisms on CT and reorders the worklist).
  - **Detection** (Gleamer BoneView: fracture boxes on X-ray, +10.4% sensitivity).
  - **Grading** (IB Lab KOALA: knee OA Kellgren–Lawrence grade on X-ray).
  - **Reconstruction** (GE AIR Recon DL, Siemens Deep Resolve: about 5-minute knee MRI, 80 M+ patients).
  - **LLM reporting** (Rad AI, PowerScribe Smart Impression).
  - **Autonomous** (LumineticsCore / IDx-DR and EyeArt: diabetic retinopathy screening, the same field as the earlier DR project).
- **No FDA-cleared product that diagnoses knee MRI findings was found.** That is still a research gap (e.g. MRNet: meniscus AUC about 0.85).
- All the commercial models are closed source.

## Open resources useful here
- **The RSNA 2024 lumbar winning solutions** — studied in [../reference/STUDY.md](../reference/STUDY.md).
- **RadImageNet** — CNNs pretrained on about 1.35 M radiology images, including MSK MRI. A drop-in encoder weight to test.
- **Knee MRI datasets** (check licences / competition rulings before using):
  - fastMRI + fastMRI+ (about 1.5k knees, pathology boxes);
  - MRNet (about 1.37k, ACL / meniscus / abnormal; registration, non-commercial);
  - OAI (thousands, OA grades; registration);
  - KneeMRI (about 900, ACL);
  - SKM-TEA (segmentation).
- **Multilingual report labelling:** Qwen2.5-7B/14B, Llama-3.1-8B. CheXbert-style labellers only cover English chest X-ray.
- **Tools:**
  - MONAI (PyTorch medical imaging);
  - TotalSegmentator-MRI (bone masks, for joint cropping);
  - pydicom + pylibjpeg / gdcm (decoding).
