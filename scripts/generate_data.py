"""
Generates synthetic but realistic medical research paper data.
Run once: python scripts/generate_data.py
Produces: data/papers.json
"""

import json, random, os

RESEARCHERS = [
    ("Dr. Anika Mehta",       "AIIMS New Delhi"),
    ("Prof. Rajiv Sundaram",  "IIT Bombay"),
    ("Dr. Chen Wei",          "Peking Union Medical College"),
    ("Dr. Sarah Okonkwo",     "University of Lagos"),
    ("Prof. James Whitfield", "Johns Hopkins"),
    ("Dr. Yuki Tanaka",       "Kyoto University"),
    ("Dr. Lena Fischer",      "Charité Berlin"),
    ("Prof. Ali Hassan",      "Aga Khan University"),
    ("Dr. Maria Santos",      "USP São Paulo"),
    ("Dr. Tom Bergström",     "Karolinska Institute"),
    ("Dr. Priya Krishnan",    "CMC Vellore"),
    ("Prof. Fatima Al-Rashid","King Faisal Specialist Hospital"),
]

COMPOUNDS = [
    ("GX-471",   "antimicrobial"),
    ("CMP-88",   "antiviral"),
    ("BRD-2201", "anticancer"),
    ("NXP-334",  "anti-inflammatory"),
    ("ZLT-904",  "antimicrobial"),
    ("VKR-115",  "antifungal"),
    ("MDX-557",  "anticancer"),
    ("PLQ-772",  "antiviral"),
    ("TRF-090",  "neuroprotective"),
    ("GSK-441",  "anti-inflammatory"),
    ("AMR-623",  "antimicrobial"),
    ("HCV-309",  "antiviral"),
]

DISEASES = [
    "H. pylori infection", "Hepatitis C", "Lung cancer",
    "Rheumatoid arthritis", "MRSA infection", "Candida albicans",
    "Glioblastoma", "SARS-CoV-2", "Alzheimer's disease",
    "Crohn's disease", "Tuberculosis", "Breast cancer",
    "Pancreatic cancer", "HIV", "E. coli infection",
]

JOURNALS = [
    "Lancet Infectious Diseases",
    "Nature Medicine",
    "NEJM",
    "Cell Host & Microbe",
    "Journal of Clinical Oncology",
    "Gut",
    "JAMA Oncology",
    "Antiviral Research",
    "Journal of Medicinal Chemistry",
    "Cancer Research",
]

METHODS = [
    "randomized controlled trial",
    "in vitro cell line study",
    "mouse xenograft model",
    "retrospective cohort study",
    "phase II clinical trial",
    "systematic review and meta-analysis",
    "CRISPR gene editing screen",
    "single-cell RNA sequencing",
    "molecular docking simulation",
    "proteomics profiling",
]

TEMPLATES = [
    """{researcher} and colleagues at {institution} investigated the efficacy of {compound} against {disease}. 
Using a {method}, the team demonstrated {efficacy}% inhibition at a minimum inhibitory concentration of {mic} μg/mL. 
The compound showed a favorable safety profile in preliminary toxicity assays. 
This work was conducted in collaboration with {collab_institution} and partially funded by the {funder}. 
Results suggest {compound} may represent a novel therapeutic candidate warranting phase {phase} clinical evaluation.""",

    """A {method} conducted at {institution} examined the molecular mechanisms by which {compound} exerts 
activity against {disease}. {researcher} reported that the compound disrupts {target} signaling, 
leading to {effect} in treated samples. The IC50 was measured at {ic50} nM. 
Co-authors from {collab_institution} contributed proteomic analysis. 
The study was published in {journal} and received {citations} citations within the first year.""",

    """This paper from {institution} presents a {method} evaluating {compound} as a treatment for {disease}. 
Lead author {researcher} demonstrated {efficacy}% response rate in the treatment arm compared to {control_rate}% in controls. 
Adverse events were mild to moderate. Biomarker analysis revealed that patients with elevated {biomarker} levels 
responded significantly better. A larger multi-center trial is currently recruiting across 
{institution}, {collab_institution}, and two additional sites.""",
]

FUNDERS = [
    "NIH National Cancer Institute",
    "Wellcome Trust",
    "Bill & Melinda Gates Foundation",
    "ICMR",
    "European Research Council",
    "DBT India",
]

TARGETS = [
    "PI3K/AKT/mTOR", "NF-κB", "VEGFR2", "PD-L1/PD-1",
    "EGFR", "JAK/STAT", "BCL-2", "Wnt/β-catenin",
]

BIOMARKERS = ["IL-6", "TNF-α", "EGFR mutation", "PD-L1 expression", "CEA", "CA-125"]


def make_paper(i):
    researcher, institution = random.choice(RESEARCHERS)
    collab_researcher, collab_institution = random.choice(RESEARCHERS)
    while collab_institution == institution:
        collab_researcher, collab_institution = random.choice(RESEARCHERS)

    compound, ctype = random.choice(COMPOUNDS)
    disease = random.choice(DISEASES)
    method = random.choice(METHODS)
    journal = random.choice(JOURNALS)
    template = random.choice(TEMPLATES)
    year = random.randint(2019, 2024)
    efficacy = random.randint(55, 98)

    text = template.format(
        researcher=researcher,
        institution=institution,
        compound=compound,
        disease=disease,
        method=method,
        efficacy=efficacy,
        mic=round(random.uniform(0.5, 32), 1),
        collab_institution=collab_institution,
        funder=random.choice(FUNDERS),
        phase=random.choice([1, 2, 3]),
        journal=journal,
        citations=random.randint(5, 340),
        ic50=round(random.uniform(10, 500), 1),
        target=random.choice(TARGETS),
        effect=random.choice(["apoptosis", "cell cycle arrest", "autophagy", "ferroptosis"]),
        control_rate=random.randint(10, 35),
        biomarker=random.choice(BIOMARKERS),
    )

    return {
        "id": f"paper_{i:03d}",
        "title": f"Study of {compound} in {disease} — {method.title()} ({year})",
        "year": year,
        "journal": journal,
        "lead_author": researcher,
        "institution": institution,
        "collaborating_institution": collab_institution,
        "compound": compound,
        "compound_type": ctype,
        "disease": disease,
        "method": method,
        "efficacy_pct": efficacy,
        "text": text.strip(),
    }


if __name__ == "__main__":
    random.seed(42)
    papers = [make_paper(i) for i in range(30)]
    os.makedirs("data", exist_ok=True)
    with open("data/papers.json", "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Generated {len(papers)} papers → data/papers.json")
    # Print a sample
    print("\nSample paper:")
    print(json.dumps(papers[0], indent=2))
