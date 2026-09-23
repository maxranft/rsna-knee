# The 12 targets

What each label means clinically, what it looks like on MRI, which series show it best, how
reports phrase it, and what that implies for the model. Written 2026-09-22 from domain knowledge,
before looking at the data — **every prevalence and labelling-rule guess below must be checked
against `train.csv` gold labels.**

The metric is macro AUC, so each of these 12 is worth exactly 1/12 of the score no matter how rare.

## Quick reference

| Label | What it is | Best plane(s) | Best sequence | Main sign | Likely prevalence* | Difficulty |
|---|---|---|---|---|---|---|
| ACL | Anterior cruciate ligament tear/injury | Sagittal (+ coronal, axial) | PD / T2 fat-sat | Fibres discontinuous or not seen; abnormal angle | Common (sports cohort) | Medium |
| MCL | Medial collateral ligament injury | **Coronal** (+ axial) | PD/T2 fat-sat | Oedema around / thickening / discontinuity of the medial band | Moderate | Medium |
| Medial Meniscus | Medial meniscus tear | Sagittal + coronal | PD (± fat-sat) | Signal line reaching the articular surface; shape change | **High** (commonest finding) | Medium–hard |
| Lateral Meniscus | Lateral meniscus tear | Sagittal + coronal | PD (± fat-sat) | Same as medial | Moderate | Hard |
| Medial OA | Medial tibiofemoral osteoarthritis | Coronal + sagittal | PD fat-sat, T1 | Cartilage loss, osteophytes, subchondral oedema/cysts | Moderate–high (older) | Medium |
| Lateral OA | Lateral tibiofemoral osteoarthritis | Coronal + sagittal | PD fat-sat, T1 | Same, lateral side | Lower | Medium |
| PF OA | Patellofemoral osteoarthritis | **Axial** + sagittal | PD fat-sat | Patellar / trochlear cartilage loss, osteophytes | Moderate–high | Medium |
| Effusion | Excess joint fluid | Sagittal (suprapatellar pouch) + axial | Any fluid-sensitive | Distended bright fluid in the suprapatellar recess | **High** | **Easy** |
| Synovitis | Inflamed joint lining | Axial + sagittal | Fluid-sensitive; ideally contrast (not given) | Thickened synovium, Hoffa fat-pad signal, debris | Low–moderate | **Hard, noisy** |
| Baker's | Popliteal (Baker's) cyst | **Axial** + sagittal | Fluid-sensitive | Fluid sac between medial gastrocnemius and semimembranosus | Moderate | Easy–medium |
| Contusion | Bone bruise | Any, esp. coronal + sagittal | **Fat-sat PD/T2 or STIR** | Ill-defined bright marrow oedema | Moderate (acute trauma) | Medium |
| Fracture | Bone fracture | Coronal + sagittal | **T1 (non-fat-sat)** + fat-sat | Dark fracture line on T1 inside the oedema | **Rare** | **Hard, high variance** |

*Guesses. Real numbers come from the gold labels.

## Per label

### ACL — anterior cruciate ligament injury
- **Anatomy:** runs from the back of the lateral femoral condyle, in the notch, to the front of the tibia. It is the main restraint against the tibia sliding forward.
- **On MRI:** a normal ACL is a taut dark band parallel to the roof of the notch (Blumensaat's line) on sagittal images. A tear shows as fibres that are missing, wavy or discontinuous, a mass of bright oedema, or a band that is too flat.
- **Secondary signs worth letting the model learn:**
  - bone bruises on the lateral femoral condyle and the back of the lateral tibial plateau (the "pivot-shift" pattern);
  - the tibia translated forward;
  - a buckled PCL;
  - a deep lateral femoral notch;
  - a Segond fracture.
- **Labelling ambiguity:** partial tears, sprains, mucoid degeneration, and a *reconstructed* ACL graft. "Injury" suggests partial tears count as positive. Graft status is unknown. Check the gold labels on studies whose report says "reconstruction" or "graft".
- **Model implications:** the sagittal fluid-sensitive series is the key slot. The ACL is thin and oblique, so resolution matters more than for most labels.

### MCL — medial collateral ligament injury
- **On MRI:** the MCL is a thin dark band along the medial side of the knee, seen on coronal images.
  - grade 1: oedema around an intact ligament;
  - grade 2: partial tear / thickening;
  - grade 3: complete discontinuity.
- **Model implications:** the coronal fat-sat slot matters most. The sign is at the *edge* of the joint, so centre crops must not cut it off.
- **Confusers:** a chronically thickened MCL after an old injury; oedema from an MCL bursa or medial OA; "pes anserine bursitis".

### Medial Meniscus / Lateral Meniscus — meniscal tears
- **On MRI:** a normal meniscus is a dark triangle (bow-tie on peripheral sagittal slices).
  - The main rule is *bright signal that clearly reaches an articular surface*, seen on ≥2 slices.
  - Signal that stays inside the meniscus is degeneration, not a tear.
- **Tear types in reports:** horizontal, radial, vertical/longitudinal, complex, root, bucket-handle (look for a "double PCL" sign), flap / displaced fragment.
- **Medial vs lateral:**
  - Medial tears are more common, especially the posterior horn and in degenerative knees.
  - Lateral tears go with acute ACL tears.
- **Ambiguity:** reports often hedge ("possible tear", "cannot exclude", "degenerative signal, no discrete tear") or describe post-surgical menisci. Hedged phrases are where report-derived labels will be noisiest.
- **Model implications:**
  - Sagittal + coronal PD.
  - Medial vs lateral is a left/right question in the image, so **orientation handling is critical** (see below).
  - A model shared across both menisci with a side indicator is worth trying, following lumbar 3rd place's "same model for every side" idea.

### Medial OA / Lateral OA / PF OA — osteoarthritis by compartment
- **On MRI:**
  - cartilage thinning or full-thickness loss;
  - osteophytes (bone spurs at the joint margins);
  - subchondral bone marrow oedema and cysts;
  - often meniscal extrusion and degeneration in the same compartment.
- **Compartments:**
  - Medial / lateral tibiofemoral are best seen on coronal images.
  - Patellofemoral means the patella and femoral trochlea, best on axial and midline sagittal images.
- **Ambiguity:** threshold. "Early chondral changes", "grade 1 chondromalacia", "mild degenerative change" may or may not be labelled positive. Check the gold labels to learn the threshold.
- **Correlations:**
  - Medial OA ↔ Medial Meniscus tear (degenerative) ↔ Effusion ↔ Baker's.
  - OA rises with age, and there is no age field. The model will pick up age proxies in the image.

### Effusion — excess joint fluid
- **On MRI:** bright fluid distending the suprapatellar recess (above the patella) on sagittal fluid-sensitive images, and on axial images. A small physiological amount is normal. Positive means "more than physiological", and where readers draw that line varies.
- **Model implications:** the easiest label, visible on almost any fluid-sensitive series. Expect a high AUC. Bank it and don't over-invest.
- **Correlations:** strong with almost every other finding (trauma and OA both cause it). It is a good *auxiliary* signal.

### Synovitis — inflammation of the joint lining
- **On MRI:**
  - Properly assessed with IV contrast, which we don't have.
  - On non-contrast MRI readers infer it from thickened synovium, oedema-like signal in Hoffa's fat pad, and fluid with debris/fronds ("effusion-synovitis").
- **Ambiguity:** heavily overlaps with Effusion. Many reports write "effusion-synovitis" as one phrase. Radiologists disagree a lot, so labels are noisy and the rules for turning report text into labels need care.
- **Model implications:** probably the lowest-AUC label. Its per-label AUC is the metric to watch for easy gains. Hoffa's fat pad (the front of the knee, sagittal) is the region to make sure isn't cropped away.

### Baker's — popliteal (Baker's) cyst
- **On MRI:** a fluid-filled sac at the back of the knee, on the medial side. Its neck passes between the medial head of gastrocnemius and the semimembranosus tendon, which is diagnostic on axial images. It may be ruptured (fluid tracking down the calf).
- **Model implications:**
  - The axial fluid-sensitive slot matters.
  - The cyst is *posterior and medial*. Tight central crops could cut it off, so keep the full field of view for this label.
- **Confusers:** ganglion cysts, parameniscal cysts, semimembranosus bursa, popliteal vessels (bright on some sequences).

### Contusion — bone contusion / bone bruise
- **On MRI:** ill-defined bright marrow oedema on fat-suppressed fluid-sensitive images (dark on T1, but no discrete line).
- **Known patterns:**
  - pivot-shift (lateral femoral condyle + posterolateral tibia → ACL);
  - patellar dislocation (medial patella + lateral femoral condyle);
  - dashboard injury (anterior tibia).
- **Ambiguity:** separating bruise from fracture (a line = fracture) and from OA-related subchondral oedema (degenerative, not traumatic). The label presumably means traumatic bruise, and reports usually say which.
- **Model implications:** **needs the fat-sat slot.** Without fat suppression the oedema is barely visible. Correlates strongly with ACL.

### Fracture
- **On MRI:** a dark linear line on **T1 non-fat-sat** images, surrounded by oedema on fat-sat images. It may be subtle:
  - tibial plateau fractures;
  - Segond fracture (small avulsion at the lateral tibia, pathognomonic for ACL tear);
  - osteochondral fractures;
  - subchondral insufficiency fractures in older patients.
- **Model implications:**
  - **Rarest label, so the noisiest AUC.** A few correct rankings swing 1/12 of the score.
  - This is the one label that argues for keeping a **non-fluid-sensitive (T1) series slot**. The v0 blueprint only kept fluid-sensitive slots, so add a T1 slot (coronal or sagittal, `Fluid_Sensitive == 0`).
  - Upweight positives, or oversample fracture-positive studies, during training.

## Cross-cutting implications

**1. Series slots (update to the v0 blueprint).** Keep up to four slots per study, "best available" for each:

| Slot | Plane | Flags | Main labels served |
|---|---|---|---|
| S1 | Sagittal | fluid-sensitive, fat-sat preferred | ACL, menisci, effusion, synovitis (Hoffa), Baker's, contusion |
| S2 | Coronal | fluid-sensitive, fat-sat preferred | MCL, menisci, medial/lateral OA, contusion, fracture |
| S3 | Axial | fluid-sensitive, fat-sat preferred | PF OA, Baker's, effusion, synovitis, MCL |
| S4 | Coronal or sagittal | **not** fluid-sensitive (T1) | Fracture, osteophytes (OA), marrow |

If a study has no series for a slot, use zeros plus a mask, and randomly drop a slot about 10% of the time in training (lumbar 6th place).

**2. Medial vs lateral orientation.**
- In DICOM patient coordinates, +x points to the patient's left. The medial side is toward the midline:
  - for a **right** knee, medial = patient-left = +x;
  - for a **left** knee, medial = patient-right = −x.
- How to find which knee it is: check whether a laterality tag (`Laterality` / `ImageLaterality` / `BodyPartExamined`) survived the 86-tag allowlist. Otherwise infer from anatomy: the fibular head is always lateral.
- Two safe options:
  1. **Never flip left-right.** The model learns orientation from anatomy. Simplest, and the v0 default.
  2. **Canonicalise every knee to one orientation first.** Then flipping is a valid augmentation, provided you swap Medial↔Lateral labels (Meniscus, OA) at the same time.
- Lumbar 7th place lost score to exactly this bug.

**3. The labels are strongly correlated** — model them jointly, and use the correlations:
- ACL ↔ Contusion (pivot-shift) ↔ Lateral Meniscus ↔ MCL ↔ Effusion ↔ Fracture (Segond). This is the acute-injury cluster.
- Medial OA ↔ Medial Meniscus (degenerative) ↔ PF OA ↔ Baker's ↔ Effusion ↔ Synovitis. This is the degenerative cluster.
- So use one shared trunk with 12 heads, not 12 separate models. Lumbar 5th place gained a lot by modelling related targets together.
- A small "acute vs degenerative" auxiliary target (derivable from report text) may help.

**4. Report-derived labels — rules to write before labelling.**
- Negation: "no", "without", "intact", "no evidence of", "kein", "sin", "yok", "не", "nema", and the same in every source language.
- Hedging: "possible", "cannot exclude", "suspected", "equivocal". Decide positive / negative / soft 0.5 by checking against the gold labels.
- Chronic vs acute and post-operative states (ACL graft, meniscectomy).
- Severity thresholds (trace effusion, mild chondral change).
- Output a probability per label rather than a hard 0/1, and train on it as a soft label.
- Report languages are unknown until we look. The contributing sites span English, German, French, Italian, Spanish (Argentina, Chile), Portuguese?, Turkish, Greek, Bulgarian, Croatian/Bosnian, Maltese/English, Thai, Chinese (Taiwan), Mongolian and Arabic/French (Morocco). A multilingual LLM (Qwen2.5) is the right tool.

**5. Where the AUC will come from.**
- Easy: Effusion, Baker's.
- Solid with a good sagittal slot: ACL.
- Medium: menisci, OA, MCL, Contusion.
- Hard / noisy: Synovitis, Fracture, Lateral Meniscus.
- Budget extra effort for the hard three. Gains there move the macro average most.
