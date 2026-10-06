"""Report -> soft labels (S1). An open-weights LLM on Kaggle answers one yes/no question per target;
score = P(yes) / (P(yes) + P(no)) from the first answer token. Report text never leaves Kaggle.

Each report's prompts share the prefix (system + report); only the question at the end differs, so vLLM's
prefix cache computes the report once. Question wording is the tuning surface: every change is
re-scored against the 58 gold studies (docs/labels.md has the clinical rationale).
"""
import math

SYSTEM = (
    "You read knee MRI radiology reports, which may be written in any language. "
    "Answer the question about the report with a single word: yes or no. "
    "Answer yes only if the report says the finding is present in the examined knee. "
    "Findings that are negated, excluded or described as normal are no."
)

# v1 wording. Keys 0-11 are the competition labels; the rest are auxiliary (laterality, post-op).
QUESTIONS = {
    "ACL": "Does the report describe a tear or injury of the anterior cruciate ligament (ACL), partial or complete?",
    "MCL": "Does the report describe an injury of the medial collateral ligament (MCL): sprain, partial tear or complete tear?",
    "Medial Meniscus": "Does the report describe a tear of the medial meniscus (not only degeneration or intrasubstance signal)?",
    "Lateral Meniscus": "Does the report describe a tear of the lateral meniscus (not only degeneration or intrasubstance signal)?",
    "Medial OA": "Does the report describe osteoarthritis or cartilage degeneration/loss in the medial tibiofemoral compartment (medial femoral condyle or medial tibial plateau)?",
    "Lateral OA": "Does the report describe osteoarthritis or cartilage degeneration/loss in the lateral tibiofemoral compartment (lateral femoral condyle or lateral tibial plateau)?",
    "PF OA": "Does the report describe osteoarthritis or cartilage degeneration/loss in the patellofemoral compartment (patella or trochlea), including chondromalacia patellae?",
    "Effusion": "Does the report describe a joint effusion (more joint fluid than a physiological amount)?",
    "Synovitis": "Does the report describe synovitis, synovial thickening or synovial proliferation (including effusion-synovitis or inflammation of Hoffa's fat pad)?",
    "Baker's": "Does the report describe a Baker's cyst (popliteal cyst)?",
    "Contusion": "Does the report describe a bone contusion, bone bruise or traumatic bone marrow oedema?",
    "Fracture": "Does the report describe a fracture (including avulsion, osteochondral or insufficiency fracture)?",
    "aux_right_knee": "Is the examined knee the right knee?",
    "aux_left_knee": "Is the examined knee the left knee?",
    "aux_prior_surgery": "Does the report mention prior surgery on this knee (for example ligament reconstruction, meniscectomy, meniscal repair or arthroplasty)?",
}
LABELS = list(QUESTIONS)[:12]

# Alternative wordings, keyed "<label>@<variant>"; the eval scores each against its base label.
# Tried and rejected in gold-v2 (all lower AUC, gold uses strict definitions): Effusion@any, Synovitis@broad,
# Lateral OA@any, PF OA@any, Contusion@oedema. See docs/labeller.md.
QUESTIONS.update({
    # adopted in gold-v2 (+0.107): gold counts small effusions as negative. Training uses this column for Effusion.
    "Effusion@moderate": "Does the report describe a moderate or large joint effusion (more than a small, minimal, trace or physiological amount of fluid)?",
    # +0.031, within noise on 58 studies: carried as a column, not adopted
    "ACL@abnormal": "Is the anterior cruciate ligament (ACL) abnormal in the report: torn (partial or complete), sprained, insufficient, or a reconstructed ACL graft?",
    # v3: manual review found cartilage fissures ("fisura") scored as fractures (docs/labeller.md, 2026-09-24)
    "Fracture@bone": "Does the report describe a fracture, meaning a break in a bone (including avulsion, osteochondral, impaction or insufficiency fracture), not a cartilage fissure or chondral defect?",
})
ADOPTED = {"Effusion": "Effusion@moderate"}  # label -> column the trainer should use (--label-map)


MAX_REPORT_CHARS = 6000  # longest training report is 4,743 characters


def base_label(key):
    return key.split("@")[0]


def build_prompts(report, tokenizer):
    """-> list of (key, prompt string), one per question, sharing the report prefix."""
    report = str(report)[:MAX_REPORT_CHARS]
    out = []
    for key, q in QUESTIONS.items():
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f'Report:\n"""\n{report}\n"""\n\nQuestion: {q}'}]
        out.append((key, tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
    return out


def yes_probability(top_logprobs):
    """top_logprobs: iterable of (decoded_token, logprob) for the first answer token.
    -> P(yes) / (P(yes) + P(no)), or 0.5 if neither word is among the candidates."""
    p = {"yes": 0.0, "no": 0.0}
    for tok, lp in top_logprobs:
        w = str(tok).strip().lower()
        if w in p:
            p[w] += math.exp(lp)
    total = p["yes"] + p["no"]
    return p["yes"] / total if total > 0 else 0.5
