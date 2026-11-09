import os
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict

from src.core_engine import (
    MatchEngine,
    EthicalTextSanitizer,
    NLPPreprocessor,
    extract_skills,
    extract_text_from_file,
    call_gemini_api,
    DEFAULT_SKILL_TAXONOMY
)

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening & Matching System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .badge-matched {
        display: inline-block;
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-missing {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    .ethical-pill {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Preset Job Descriptions
PRESET_JDS = {
    "Senior Data Scientist / ML Engineer": """Job Title: Senior Data Scientist / Machine Learning Engineer
Department: AI & Data Analytics
Requirements:
- Strong proficiency in Python, SQL, and data analysis.
- Proven experience with Machine Learning (scikit-learn, XGBoost, Random Forest).
- Experience in Natural Language Processing (NLP), NLTK, spaCy, and Transformers.
- Deep Learning using PyTorch or TensorFlow.
- Experience with Data Visualization using Matplotlib, Seaborn, Tableau.
- Familiarity with Docker, Git, and AWS cloud deployment.
- Bachelor's or Master's degree in Computer Science or Data Science.""",

    "Full Stack Web Developer (Java & Modern Web)": """Job Title: Full Stack Web Developer
Department: Engineering & Product Development
Requirements:
- Hands-on experience developing web applications using Java, Spring Boot, Node.js, and Express.js.
- Frontend development experience with React.js, Tailwind CSS, HTML, and CSS.
- Proven expertise designing REST APIs, microservices, and CRUD workflows.
- Implementation of secure authentication using JWT (JSON Web Tokens) and Role-Based Access Control (RBAC).
- Database design, optimization, and querying with MySQL and MongoDB.
- Practical experience with Git, GitHub, Docker containerization, and Postman API testing.
- Understanding of AWS cloud fundamentals (AWS Cloud Practitioner), OOP, and Agile methodology.
- Bachelor's degree in Computer Science, Software Engineering, or related STEM field.""",

    "Cloud & DevOps Engineer": """Job Title: Cloud DevOps & Infrastructure Engineer
Department: Platform Infrastructure
Requirements:
- Hands-on experience managing AWS, Azure, or GCP cloud platforms.
- Deep expertise in Docker containerization and Kubernetes orchestration.
- Infrastructure as Code (IaC) using Terraform or Ansible.
- Automated CI/CD pipeline development using Jenkins, GitHub Actions, or GitLab CI.
- System administration and automation with Linux, Bash, and Python scripting.
- Experience with monitoring tools like Prometheus and Grafana."""
}

# Header Section
st.markdown('<div class="main-header">🎯 AI-Powered Resume Screening & Job Matching</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>NLP · TF-IDF & Cosine Similarity · Feature Engineering · Ethical AI · Gemini API Layer</b></div>', unsafe_allow_html=True)

with st.expander("👥 Project Team & Architecture Details", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Satwik Saxena** (TCA2357046)\n\n*Ingestion & Ethical AI (PII Scrubber)*")
    with col2:
        st.markdown("**Dhruv Jain** (TCA2357019)\n\n*Feature Engineering & Domain Classifier*")
    with col3:
        st.markdown("**Hardik Chauhan** (TCA2357022)\n\n*TF-IDF Vectorization & Matching Engine*")
    with col4:
        st.markdown("**Sanya Khare** (TCA2357045)\n\n*Explainable AI & Gemini Integration*")

st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Screening Configuration")

# Ethical AI Toggle
ethical_toggle = st.sidebar.toggle(
    "🛡️ Ethical AI Mode",
    value=True,
    help="Redacts names, contact info, gender pronouns, age, marital status, and nationality to eliminate personal bias."
)

# Weight Sliders
st.sidebar.subheader("⚖️ Scoring Formula Weights")
tfidf_pct = st.sidebar.slider("TF-IDF Cosine Similarity (%)", min_value=10, max_value=90, value=50, step=5)
skill_pct = 100 - tfidf_pct
st.sidebar.caption(f"Skill Match Ratio: **{skill_pct}%**")

# Gemini API Key (Optional)
st.sidebar.subheader("✨ Gemini AI Integration")
gemini_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    type="password",
    help="Enter your Google AI Studio API key for live Gemini 2.5 Flash generations. If left empty, built-in intelligent evaluation is used."
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 *Model vectorizer fitted across 962 real resumes with 98.96% domain classification accuracy.*")

# Initialize Engine
engine = MatchEngine(tfidf_weight=tfidf_pct/100.0, skill_weight=skill_pct/100.0)

# Main Form: Two Column Layout
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Job Description")
    jd_choice = st.selectbox("Select Preset or Custom:", list(PRESET_JDS.keys()) + ["Custom Job Description"])
    
    if jd_choice != "Custom Job Description":
        default_jd = PRESET_JDS[jd_choice]
    else:
        default_jd = ""
        
    jd_text = st.text_area("Job Requirements & Keywords:", value=default_jd, height=260)
    
    # Extract JD skills live
    if jd_text.strip():
        extracted_jd_skills = extract_skills(jd_text)
        st.caption(f"**Required Skills Detected ({len(extracted_jd_skills)}):**")
        badges = " ".join([f"<span class='badge-matched'>{s}</span>" for s in sorted(list(extracted_jd_skills))])
        st.markdown(badges, unsafe_allow_html=True)

with col_right:
    st.subheader("2. Candidate Resumes")
    ingest_mode = st.radio("Resume Source:", ["📁 Upload PDF / DOCX Resumes", "📚 Load Sample Benchmark Candidates"], horizontal=True)
    
    candidate_list: List[Dict] = []
    
    if ingest_mode == "📁 Upload PDF / DOCX Resumes":
        uploaded_files = st.file_uploader(
            "Upload resumes (.pdf, .docx, .txt):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                text = extract_text_from_file(f, f.name)
                candidate_list.append({
                    "id": f.name,
                    "name": os.path.splitext(f.name)[0],
                    "raw_text": text
                })
            st.success(f"Loaded {len(candidate_list)} resume files successfully.")
    else:
        # Load sample benchmark resumes from disk
        sample_dir = "data/sample_resumes_pdf"
        if os.path.exists(sample_dir):
            for fname in sorted(os.listdir(sample_dir)):
                if fname.lower().endswith('.pdf'):
                    fpath = os.path.join(sample_dir, fname)
                    with open(fpath, "rb") as fp:
                        text = extract_text_from_file(fp, fname)
                        candidate_list.append({
                            "id": fname,
                            "name": os.path.splitext(fname)[0].replace("_", " "),
                            "raw_text": text
                        })
            st.info(f"Loaded {len(candidate_list)} benchmark candidates from `{sample_dir}`.")

# Action Button
st.markdown("")
run_button = st.button("🚀 Screen & Rank Candidates", type="primary", use_container_width=True)

# Run Screening
if run_button:
    if not jd_text.strip():
        st.error("Please provide a Job Description to compare against.")
    elif not candidate_list:
        st.warning("Please upload at least one candidate resume or select benchmark candidates.")
    else:
        with st.spinner("Processing resumes, applying ethical sanitization, and calculating similarity..."):
            results_df = engine.match_resumes(
                job_desc=jd_text,
                candidates=candidate_list,
                apply_ethical_scrub=ethical_toggle
            )
            
        st.session_state["results_df"] = results_df
        st.session_state["current_jd"] = jd_text

# Display Results
if "results_df" in st.session_state and not st.session_state["results_df"].empty:
    results_df = st.session_state["results_df"]
    current_jd = st.session_state.get("current_jd", "")
    
    st.markdown("---")
    st.subheader("🏆 Candidate Screening Leaderboard")
    
    # Ethical AI Banner
    if ethical_toggle:
        st.markdown("""
        <div class="ethical-pill">
            🛡️ <b>Ethical AI Active:</b> Names, personal contact information, gender pronouns (<i>he/she/him/her</i>), age, and nationalities were stripped prior to vectorization to guarantee an unbiased technical evaluation.
        </div>
        """, unsafe_allow_html=True)
        
    # Metric Summary Cards
    top_cand = results_df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates Evaluated", len(results_df))
    m2.metric("Top Match Candidate", top_cand["Candidate ID"])
    m3.metric("Highest Match Score", f"{top_cand['Overall Score (%)']}%")
    m4.metric("Avg Candidate Alignment", f"{results_df['Overall Score (%)'].mean():.1f}%")
    
    # Leaderboard Table
    display_df = results_df[[
        "Rank", "Candidate ID", "Predicted Domain", "Overall Score (%)",
        "Skill Match Ratio (%)", "Experience Evidence (%)", "Semantic Relevance (%)",
        "Matched Count", "Total Required"
    ]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Interactive Score Bar Chart
    st.subheader("📊 Match Score Breakdown")
    fig = px.bar(
        results_df,
        x="Overall Score (%)",
        y="Candidate ID",
        color="Overall Score (%)",
        orientation="h",
        color_continuous_scale="Viridis",
        title="Candidate Overall Match Score (%)",
        text="Overall Score (%)"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Candidate Inspector
    st.subheader("🔍 Deep-Dive Candidate Inspector & AI Feedback")
    
    for idx, row in results_df.iterrows():
        score = row["Overall Score (%)"]
        rank_badge = "🥇" if row["Rank"] == 1 else "🥈" if row["Rank"] == 2 else "🥉" if row["Rank"] == 3 else f"#{row['Rank']}"
        
        with st.expander(f"{rank_badge} **{row['Candidate ID']}** — Match: **{score}%** | Domain: **{row['Predicted Domain']}**"):
            c_col1, c_col2 = st.columns([1, 1])
            
            with c_col1:
                col_sub1, col_sub2 = st.columns(2)
                col_sub1.metric("🎯 Skill Coverage", f"{row['Skill Match Ratio (%)']}%", f"{row['Matched Count']}/{row['Total Required']} skills")
                col_sub2.metric("💼 Experience Evidence", f"{row.get('Experience Evidence (%)', 'N/A')}%")
                
                col_sub3, col_sub4 = st.columns(2)
                col_sub3.metric("🧠 Semantic Relevance", f"{row.get('Semantic Relevance (%)', 'N/A')}%")
                col_sub4.metric("🎓 Education & Domain", f"{row.get('Education Score (%)', 'N/A')}%")
                
                st.markdown(f"**Ethical Redactions Applied:** `{row['Ethical Redactions']}`")
                
                st.markdown("**✅ Verified Matched Skills:**")
                if row["Matched Skills"]:
                    m_html = " ".join([f"<span class='badge-matched'>{s}</span>" for s in row["Matched Skills"]])
                    st.markdown(m_html, unsafe_allow_html=True)
                else:
                    st.caption("No direct skill matches detected.")
                    
                st.markdown("**⚠️ Missing Required Skills:**")
                if row["Missing Skills"]:
                    miss_html = " ".join([f"<span class='badge-missing'>{s}</span>" for s in row["Missing Skills"]])
                    st.markdown(miss_html, unsafe_allow_html=True)
                else:
                    st.caption("All job required skills satisfied!")
                    
            with c_col2:
                st.markdown("#### ✨ AI Recruiter & Gemini Insight")
                gen_btn = st.button(f"Generate AI Analysis for Candidate {row['Candidate ID']}", key=f"gemini_btn_{idx}")
                
                if gen_btn:
                    with st.spinner("Analyzing candidate profile with Gemini API..."):
                        ai_feedback = call_gemini_api(gemini_key, row.to_dict(), current_jd)
                    st.markdown(ai_feedback)
                else:
                    st.info("Click above to generate executive screening feedback, tailored interview questions, and candidate improvement advice.")
                    
            # Raw Text vs Sanitized Text Preview
            with st.expander("📄 View Candidate Raw Text"):
                st.text(row["Raw Text"][:1000] + ("..." if len(row["Raw Text"]) > 1000 else ""))

