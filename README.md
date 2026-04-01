# VERDICT v2 - AI Decision Firewall

## Problem Statement
AI systems used in hiring, lending, insurance, education, and criminal justice can produce biased outcomes against protected groups. Most teams only discover these issues after deployment, and existing tools are often too technical, fragmented, or static for decision-makers to use effectively. VERDICT v2 solves this by giving users an interactive fairness audit platform that detects demographic bias, exposes proxy-feature risks, compares model behavior, simulates fairer predictions, and generates compliance-ready audit evidence.

## Project Description
VERDICT v2 is a full-stack fairness auditing platform built to inspect decision datasets and model outputs through a guided six-step workflow:

1. Ingestion: Users upload a CSV and preview the dataset.
2. The Mirror: The platform scans for disparity across sensitive attributes, checks proxy-feature correlations, tests statistical significance, and performs intersectional analysis.
3. The Face-Off: Users compare real records from different groups to understand how bias appears in practice.
4. The Fix: The debiasing engine removes sensitive signals and proxy leakage, then shows before/after fairness impact.
5. Live Shield: Users submit a single live record and get biased vs. fair prediction outputs side by side.
6. Verdict: The system produces a final audit verdict, maps findings to regulations, and exports a branded PDF certificate with integrity hash and QR code.
 
What makes it useful is that it combines fairness diagnostics, explainability, interactive testing, model comparison, and exportable audit documentation in one product experience.
## Google AI Usage


### Tools / Models Used
- Google Gemini API / Google AI Studio
- Gemini model for product ideation, UX iteration, fairness explanation drafting, and audit-language refinement
 

### How Google AI Was Used
Google AI was used as an intelligence layer during the product design and build process. It helped:

- refine the Decision Firewall concept and product workflow
- improve how fairness findings are explained in plain English
- shape interactive UX ideas for the six-screen flow
- generate and improve audit-report wording and compliance-oriented summaries
- support iteration on debiasing, live prediction, and user-facing explanation design
AI is integrated into the project workflow as a co-pilot for reasoning, product framing, and fairness communication, helping turn complex bias metrics into more understandable outputs for users.

## Proof of Google AI Usage
Attach screenshots in a `/proof` folder:
<img width="1910" height="847" alt="Screenshot 2026-04-01 070012" src="https://github.com/user-attachments/assets/111bd6ad-e8b3-4842-90f8-77456c1bc529" />


<img width="1920" height="1080" alt="Screenshot 2026-04-01 065453" src="https://github.com/user-attachments/assets/2a0b1021-201c-4a6f-b5ab-0246a33f4486" />

## Screenshots 
<img width="1920" height="892" alt="Screenshot (649)" src="https://github.com/user-attachments/assets/9969f932-05ec-4d32-b04d-fa1ed7946dc2" />
<img width="1920" height="1080" alt="Screenshot (653)" src="https://github.com/user-attachments/assets/f477243a-3d8f-4f31-9c8e-6beab212a64c" />
<img width="1920" height="885" alt="Screenshot (655)" src="https://github.com/user-attachments/assets/f3fe2952-5dd8-43f0-821a-872f4188d6ca" />
<img width="1920" height="884" alt="Screenshot (656)" src="https://github.com/user-attachments/assets/180cac7a-50ea-4bc4-a499-8b776268bd32" />
<img width="1913" height="875" alt="Screenshot 2026-04-01 064946" src="https://github.com/user-attachments/assets/3be2aee0-7304-4c5c-8e41-5462ab960abf" />
<img width="1871" height="882" alt="Screenshot 2026-04-01 065055" src="https://github.com/user-attachments/assets/43d45f2e-855a-403b-ab62-79dd5bf4355a" />

## Demo Video
https://drive.google.com/drive/folders/1To1Orvvt_6ShxTyWhX91IwV36dNAtgvO?usp=sharing # watch the demo

## Installation Steps

```bash
# Clone the repository
git clone <your-repo-link>

# Go to project folder
cd verdict-v2

# Backend setup
cd backend
.\venv\Scripts\python.exe -m uvicorn verdict_api:app --host 127.0.0.1 --port 8000 --reload

# Open a second terminal and go to frontend
cd ../frontend

# Install dependencies
npm install

# Run the project
npm run dev
