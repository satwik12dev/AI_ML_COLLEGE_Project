# 🚀 AI-Powered Resume Screening and Job Matching System

**Tags:** `NLP` · `Similarity Matching` · `Feature Engineering` · `Ethical AI` · `Supervised Machine Learning` · `Gemini API`

---

## 👥 Team Details
- **Dhruv Jain** (TCA2357019) – Feature Engineering, Skill Taxonomy & Supervised Domain Classifier
- **Hardik Chauhan** (TCA2357022) – TF-IDF Vectorization, Cosine Similarity & Ranking Engine
- **Satwik Saxena** (TCA2357046) – Ingestion Pipeline (PDF & CSV) & Ethical AI (PII/Demographic Scrubber)
- **Sanya Khare** (TCA2357045) – Explainable AI, Visualizations & Gemini API Hybrid Layer

---

## 📁 Repository Structure
```
AI Resume Analyse/
├── app.py                        # Streamlit Interactive Web Application
├── resume_screener_model.ipynb   # Complete, executed Jupyter Notebook (98.96% accuracy)
├── build_notebook.py             # Notebook builder & executor script
├── category_distribution.png     # EDA plot: 25 Job Category distributions
├── candidate_scores.png          # Ranking & similarity comparison plot
├── src/
│   └── core_engine.py            # Reusable NLP, Ethical AI, & ML Matcher Engine
├── data/
│   ├── UpdatedResumeDataSet.csv  # Benchmark dataset: 962 real-world resumes across 25 domains
│   └── sample_resumes_pdf/       # Folder of real PDF resumes for live testing
│       ├── Candidate_Alice_AI.pdf
│       ├── Candidate_Bob_FullStack.pdf
│       └── Candidate_Charlie_DevOps.pdf
├── models/                       # Exported trained ML models & artifacts
│   ├── resume_classifier.pkl     # Logistic Regression Domain Classifier (98.96% accuracy)
│   ├── classifier_vectorizer.pkl # TF-IDF Vocabulary on 962 Resumes
│   ├── label_encoder.pkl         # 25 Category Encoders
│   ├── tfidf_vectorizer.pkl      # Job Match Vectorizer
│   └── skill_taxonomy.pkl        # Domain Technical Taxonomy
└── README.md
```

---

## 🌐 How to Launch the Web Application

Run this single command in your terminal:
```bash
streamlit run app.py
```
This automatically opens the Web Dashboard in your browser at `http://localhost:8501`.

### App Features:
1. **Multi-Format Resume Ingestion:** Upload multiple `.pdf`, `.docx`, or `.txt` resumes, or load pre-existing candidates with one click.
2. **Interactive Job Description:** Choose from built-in job presets (Data Scientist, Full Stack, DevOps) or enter a custom JD.
3. **Real-Time Ethical AI Toggle:** Watch how names, phone numbers, emails, gender pronouns (*he/she/him/her*), age, and nationalities are scrubbed before ranking.
4. **Interactive Leaderboard & Chart:** Ranked candidate table with Plotly horizontal match score visualization.
5. **Skill Gap Cards:** Visual color-coded badges for **Matched Skills** (green) and **Missing Skills** (red).
6. **Gemini AI Integration:** Generates an executive screening summary, 3 targeted interview questions, and candidate development advice.

---

## 🔬 How to Run the Jupyter Notebook

### In VS Code:
1. Open [`resume_screener_model.ipynb`](file:///c:/Users/SATWIK/OneDrive/Desktop/AI Resume Analyse/resume_screener_model.ipynb).
2. Select your Python 3.11 kernel and click **Run All**.

### In Jupyter Terminal:
```bash
jupyter notebook resume_screener_model.ipynb
```
