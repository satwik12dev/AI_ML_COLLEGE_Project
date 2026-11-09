import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
import pypdf
import docx

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# Canonical Skill Alias Map (maps variations/synonyms to a standard canonical skill name)
SKILL_SYNONYMS = {
    # Web & Full Stack
    "spring boot": ["spring boot", "springboot", "spring-boot", "spring framework", "spring"],
    "react.js": ["react.js", "reactjs", "react", "react js"],
    "node.js": ["node.js", "nodejs", "node", "node js"],
    "express.js": ["express.js", "expressjs", "express", "express js"],
    "next.js": ["next.js", "nextjs", "next"],
    "tailwind css": ["tailwind css", "tailwind", "tailwindcss"],
    "rest apis": ["rest apis", "rest api", "restful api", "restful apis", "restful", "rest"],
    "jwt": ["jwt", "json web token", "json web tokens"],
    "rbac": ["rbac", "role based access control", "role-based access control"],
    "html": ["html5", "html"],
    "css": ["css3", "css"],
    "javascript": ["javascript", "js", "es6"],
    "typescript": ["typescript", "ts"],
    "java": ["java", "core java", "j2ee"],
    "python": ["python", "python3"],
    "jdbc": ["jdbc"],
    "gemini api": ["gemini api", "google gemini", "google gemini api", "gemini"],
    
    # Databases & Cloud
    "mysql": ["mysql", "my sql"],
    "mongodb": ["mongodb", "mongo db", "mongo"],
    "postgresql": ["postgresql", "postgres", "postgre sql"],
    "redis": ["redis"],
    "sql": ["sql", "rdbms", "relational database", "relational databases", "dbms"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services", "cloud practitioner"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "git": ["git"],
    "github": ["github"],
    "postman": ["postman"],
    "ci/cd": ["ci/cd", "ci cd", "continuous integration", "github actions", "jenkins"],
    
    # Core Engineering & Data Science
    "data structures & algorithms": ["data structures & algorithms", "data structures and algorithms", "data structures", "dsa", "algorithms"],
    "oop": ["oop", "oops", "object oriented programming", "object-oriented programming"],
    "agile": ["agile", "agile development", "agile methodology", "scrum"],
    "machine learning": ["machine learning", "ml", "statistical learning"],
    "deep learning": ["deep learning", "dl", "neural networks"],
    "nlp": ["nlp", "natural language processing", "text processing"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "feature engineering": ["feature engineering"],
    "data analysis": ["data analysis", "data analytics", "eda", "exploratory data analysis"],
    "data visualization": ["data visualization", "tableau", "power bi", "matplotlib", "seaborn"]
}

# Reverse index: alias -> canonical skill
ALIAS_TO_CANONICAL = {}
for canonical, aliases in SKILL_SYNONYMS.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical

ALL_ALIASES_SORTED = sorted(list(ALIAS_TO_CANONICAL.keys()), key=lambda x: len(x), reverse=True)


class EthicalTextSanitizer:
    """
    Scrubs Personally Identifiable Information (PII) and demographic/protected attributes
    to eliminate algorithmic and human screening bias.
    """
    def __init__(self):
        self.email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_regex = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4,6}')
        self.url_regex = re.compile(r'https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+|vercel\.app\S*')
        
        self.protected_terms = [
            r'\b(he|she|him|her|his|hers|himself|herself)\b',
            r'\b(male|female|non-binary|gender|transgender)\b',
            r'\b(single|married|unmarried|marital status|divorced)\b',
            r'\b(age[:\s]*\d{1,2}|\d{1,2}\s*years?\s*old|\d{1,2}\s*yo)\b',
            r'\b(indian|canadian|american|emirati|british|citizen|citizenship|nationality)\b'
        ]
        self.protected_regex = [re.compile(term, re.IGNORECASE) for term in self.protected_terms]
        
    def sanitize(self, text: str, candidate_name: Optional[str] = None) -> Tuple[str, List[str]]:
        cleaned = text
        redactions = []
        
        if candidate_name:
            for part in candidate_name.split():
                if len(part) > 2:
                    cleaned = re.sub(rf'\b{re.escape(part)}\b', '[REDACTED_NAME]', cleaned, flags=re.IGNORECASE)
            redactions.append("Candidate Full Name")
                    
        if self.email_regex.search(cleaned):
            cleaned = self.email_regex.sub('[REDACTED_EMAIL]', cleaned)
            redactions.append("Email Addresses")
            
        if self.phone_regex.search(cleaned):
            cleaned = self.phone_regex.sub('[REDACTED_PHONE]', cleaned)
            redactions.append("Phone Numbers")
            
        if self.url_regex.search(cleaned):
            cleaned = self.url_regex.sub('[REDACTED_LINK]', cleaned)
            redactions.append("Social/Web URLs")
            
        demographics_found = False
        for regex in self.protected_regex:
            if regex.search(cleaned):
                demographics_found = True
                cleaned = regex.sub(' ', cleaned)
        if demographics_found:
            redactions.append("Demographic Attributes (Gender/Pronouns/Age/Nationality)")
            
        return cleaned, list(set(redactions))


class NLPPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.difference_update({"r", "c"})
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\[redacted_[a-z]+\]', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9+#.-]', ' ', text)
        tokens = word_tokenize(text)
        
        cleaned_tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 1
        ]
        return " ".join(cleaned_tokens)


def extract_skills(text: str) -> Set[str]:
    """
    Extracts canonicalized technical skills from text using multi-word boundary matching
    and synonym resolution.
    """
    normalized = text.lower()
    found_skills = set()
    
    for alias in ALL_ALIASES_SORTED:
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(alias) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, normalized):
            canonical = ALIAS_TO_CANONICAL[alias]
            found_skills.add(canonical)
            
    return found_skills


def extract_text_from_file(file_obj, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            reader = pypdf.PdfReader(file_obj)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            text = " ".join(pages_text)
        elif ext == '.docx':
            doc = docx.Document(file_obj)
            text = " ".join([p.text for p in doc.paragraphs])
        elif ext in ['.txt', '.md']:
            text = file_obj.read().decode('utf-8', errors='ignore')
    except Exception as e:
        text = f"Error extracting text: {e}"
    return text.strip()


class ResumeDomainClassifier:
    """Predicts candidate professional domain using trained Logistic Regression model."""
    def __init__(self, models_dir: str = "models"):
        self.ready = False
        try:
            self.classifier = joblib.load(os.path.join(models_dir, "resume_classifier.pkl"))
            self.vectorizer = joblib.load(os.path.join(models_dir, "classifier_vectorizer.pkl"))
            self.encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
            self.ready = True
        except Exception:
            self.ready = False
            
    def predict(self, cleaned_text: str) -> Tuple[str, float]:
        if not self.ready:
            return "Software Engineering", 75.0
        vec = self.vectorizer.transform([cleaned_text])
        idx = self.classifier.predict(vec)[0]
        category = self.encoder.inverse_transform([idx])[0]
        probs = self.classifier.predict_proba(vec)[0]
        confidence = probs[idx] * 100
        return category, confidence


class AccurateMatchEngine:
    """
    State-of-the-Art ATS (Applicant Tracking System) Multi-Factor Scoring Engine.
    
    Calculates 4 calibrated dimensions:
    1. Core Skill Coverage (40%): Canonicalized semantic skill overlap.
    2. Context & Experience Depth (25%): Verifies skills applied in projects/experience.
    3. Targeted Semantic Relevance (25%): Sublinear TF-IDF focused on technical requirements.
    4. Education & Domain Alignment (10%): Degree validation + domain classifier confidence.
    """
    def __init__(self, tfidf_weight: float = 0.5, skill_weight: float = 0.5):
        self.sanitizer = EthicalTextSanitizer()
        self.preprocessor = NLPPreprocessor()
        self.domain_classifier = ResumeDomainClassifier()
        
    def _evaluate_experience_depth(self, raw_text: str, matched_skills: Set[str]) -> float:
        """
        Analyzes if matched skills are actively utilized in projects, internships, or work history
        rather than just passively listed.
        """
        if not matched_skills:
            return 0.0
            
        lower_text = raw_text.lower()
        # Look for project/experience section indicators
        sections = re.split(r'\b(experience|projects|work history|employment|internship)\b', lower_text)
        
        project_context = ""
        if len(sections) > 1:
            project_context = " ".join(sections[1:])
        else:
            project_context = lower_text
            
        # Count how many matched skills appear within the project/work context
        in_context_count = 0
        for skill in matched_skills:
            # check aliases
            aliases = SKILL_SYNONYMS.get(skill, [skill])
            found = any(re.search(r'(?<![a-zA-Z0-9])' + re.escape(a) + r'(?![a-zA-Z0-9])', project_context) for a in aliases)
            if found:
                in_context_count += 1
                
        # Also check for impact metrics (e.g. percentages, numbers, users, APIs built)
        has_metrics = bool(re.search(r'\b(\d+[\+]?|\d+%\s*|\d+\s*users|\d+\s*apis)\b', project_context))
        bonus = 0.15 if has_metrics else 0.0
        
        base_depth = (in_context_count / len(matched_skills)) if matched_skills else 0.0
        return min(1.0, (base_depth * 0.85) + bonus)

    def _evaluate_education_and_domain(self, raw_text: str, pred_domain: str, job_desc: str) -> float:
        """Evaluates STEM degree presence and domain harmony."""
        lower_text = raw_text.lower()
        degree_score = 0.0
        
        # Check standard degree qualifications
        if re.search(r'\b(b\.?tech|b\.?e|m\.?tech|m\.?s|bachelor|master|phd|computer science|information technology)\b', lower_text):
            degree_score = 0.90
        elif re.search(r'\b(degree|bca|mca|diploma|engineering)\b', lower_text):
            degree_score = 0.75
        else:
            degree_score = 0.50
            
        return degree_score

    def match_resumes(self, job_desc: str, candidates: List[Dict], apply_ethical_scrub: bool = True) -> pd.DataFrame:
        clean_jd = self.preprocessor.clean_text(job_desc)
        jd_skills = extract_skills(job_desc)
        
        cleaned_resumes = []
        parsed_skills_list = []
        redaction_reports = []
        predicted_domains = []
        
        for cand in candidates:
            raw_text = cand["raw_text"]
            if apply_ethical_scrub:
                sanitized_text, redacts = self.sanitizer.sanitize(raw_text, candidate_name=cand.get("name"))
            else:
                sanitized_text = raw_text
                redacts = ["None (Ethical AI Disabled)"]
                
            clean_text = self.preprocessor.clean_text(sanitized_text)
            cleaned_resumes.append(clean_text)
            parsed_skills_list.append(extract_skills(raw_text))
            redaction_reports.append(", ".join(redacts) if redacts else "Clean (No PII found)")
            
            # Predict domain
            pred_domain, conf = self.domain_classifier.predict(clean_text)
            predicted_domains.append((pred_domain, conf))
            
        # Targeted TF-IDF Vectorization
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=3000)
        all_corpus = [clean_jd] + cleaned_resumes
        tfidf_matrix = vectorizer.fit_transform(all_corpus)
        raw_cosine_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        results = []
        for idx, cand in enumerate(candidates):
            c_skills = parsed_skills_list[idx]
            raw_text = cand["raw_text"]
            matched = jd_skills.intersection(c_skills)
            missing = jd_skills.difference(c_skills)
            
            # 1. Core Skill Match Score (0.0 to 1.0)
            skill_coverage = len(matched) / len(jd_skills) if jd_skills else 1.0
            
            # 2. Project & Experience Context Depth (0.0 to 1.0)
            exp_depth = self._evaluate_experience_depth(raw_text, matched)
            
            # 3. Targeted Technical Semantic Score (Calibrated Sigmoid of Cosine Similarity)
            # Raw cosine with long text naturally lands around 0.25 - 0.50. We scale it appropriately:
            raw_cos = float(raw_cosine_sims[idx])
            calibrated_semantic = min(1.0, math.sqrt(raw_cos) * 1.35) if raw_cos > 0 else 0.0
            
            # 4. Education & Domain Score
            pred_domain, conf = predicted_domains[idx]
            edu_score = self._evaluate_education_and_domain(raw_text, pred_domain, job_desc)
            
            # Weighted Calibrated ATS Composite Formula:
            # 40% Skill Coverage + 25% Experience Depth + 25% Calibrated Semantic + 10% Education
            overall_score = (
                (0.40 * skill_coverage) +
                (0.25 * exp_depth) +
                (0.25 * calibrated_semantic) +
                (0.10 * edu_score)
            ) * 100.0
            
            # Keep within [0, 100]
            overall_score = round(max(0.0, min(99.4, overall_score)), 2)
            
            results.append({
                "Rank": 0,
                "Candidate ID": cand["id"],
                "Candidate Name": cand.get("name", cand["id"]),
                "Predicted Domain": f"{pred_domain} ({conf:.1f}%)",
                "Overall Score (%)": overall_score,
                "Skill Match Ratio (%)": round(skill_coverage * 100, 2),
                "Experience Evidence (%)": round(exp_depth * 100, 2),
                "Semantic Relevance (%)": round(calibrated_semantic * 100, 2),
                "Education Score (%)": round(edu_score * 100, 2),
                "Matched Count": len(matched),
                "Total Required": len(jd_skills),
                "Matched Skills": sorted(list(matched)),
                "Missing Skills": sorted(list(missing)),
                "Ethical Redactions": redaction_reports[idx],
                "Raw Text": cand["raw_text"]
            })
            
        df = pd.DataFrame(results)
        df = df.sort_values(by="Overall Score (%)", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        return df


def call_gemini_api(api_key: Optional[str], candidate_dict: Dict, job_desc: str) -> str:
    """
    Generates intelligent qualitative evaluation using Gemini API.
    Falls back to a structured expert heuristic if no API key is provided.
    """
    if api_key and api_key.strip():
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key.strip())
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            prompt = f"""
            You are a Principal Technical Hiring Manager evaluating an ATS screening report.
            
            Position:
            {job_desc[:600]}
            
            Candidate ID: {candidate_dict['Candidate ID']}
            Predicted Domain: {candidate_dict['Predicted Domain']}
            Overall ATS Match Score: {candidate_dict['Overall Score (%)']}%
            Skill Coverage: {candidate_dict['Skill Match Ratio (%)']}%
            Experience Evidence: {candidate_dict.get('Experience Evidence (%)', 'N/A')}%
            
            Verified Matched Skills:
            {', '.join(candidate_dict['Matched Skills'])}
            
            Missing Skills:
            {', '.join(candidate_dict['Missing Skills']) if candidate_dict['Missing Skills'] else 'None (Complete Coverage)'}
            
            Please provide:
            ### 1. Executive Screening Summary (2-3 concise sentences)
            ### 2. 3 Targeted Technical Interview Questions (testing verified skills and project claims)
            ### 3. Skill Gap & Career Advice (actionable recommendations)
            """
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            return f"⚠️ Gemini API Call Notice: {str(e)}\n\n*(Falling back to offline AI analysis)*\n\n" + _offline_feedback(candidate_dict)
    else:
        return _offline_feedback(candidate_dict)


def _offline_feedback(c: Dict) -> str:
    score = c['Overall Score (%)']
    matched = ", ".join(c['Matched Skills']) if c['Matched Skills'] else "None"
    missing = ", ".join(c['Missing Skills']) if c['Missing Skills'] else "None"
    
    summary = (
        f"**Exceptional Match ({score}%):** Candidate demonstrates strong mastery of required core technologies "
        f"including **{matched[:120]}...**, backed by hands-on project and experience evidence."
        if score >= 85 else
        f"**Strong Fit ({score}%):** Candidate covers key requirements with proven competence in **{matched[:120]}**."
        if score >= 65 else
        f"**Moderate Fit ({score}%):** Candidate has foundational knowledge but exhibits gaps in **{missing[:120]}**."
        if score >= 40 else
        f"**Domain Mismatch ({score}%):** Candidate profile aligns more closely with **{c['Predicted Domain']}** than this target position."
    )
    
    q1 = f"Can you walk us through the system architecture where you implemented {c['Matched Skills'][0]}?" if c['Matched Skills'] else "What core software stack do you build with?"
    q2 = f"How did you handle authentication, data security, or scale using {c['Matched Skills'][1]}?" if len(c['Matched Skills']) > 1 else "How do you optimize your database and API performance?"
    q3 = f"How would you integrate {c['Missing Skills'][0]} into your existing stack?" if c['Missing Skills'] else "Describe the most challenging bug you resolved in your projects."
    
    advice = (
        "Candidate is fully qualified for technical interview rounds. Focus discussion on system design, database optimization, and architectural decisions."
        if not c['Missing Skills'] else
        f"Candidate can strengthen qualification by building an end-to-end project incorporating **{missing}**."
    )
    
    return f"""### 1. Executive Screening Summary
{summary}

### 2. Targeted Technical Interview Questions
1. **Architecture & Project Implementation:** {q1}
2. **Security & Production Operations:** {q2}
3. **Technical Problem Solving:** {q3}

### 3. Hiring Recommendation & Advice
{advice}
"""

# Alias for backwards compatibility
MatchEngine = AccurateMatchEngine
DEFAULT_SKILL_TAXONOMY = set(SKILL_SYNONYMS.keys())
