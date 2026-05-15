# app.py
import streamlit as st
import json
import re
import plotly.graph_objects as go
from utils import (
    extract_text_from_pdf,
    extract_skills,
    extract_skills_by_category,
    calculate_similarity,
    get_missing_skills,
    get_semantic_skill_matches,
    extract_tfidf_keywords,
    keyword_coverage_score,
    extract_resume_entities,
    calculate_ats_score,
    get_category_coverage,
    calculate_composite_score,
    generate_specific_suggestions,
    extract_jd_requirements,
    calculate_section_scores,
    get_sentence_matches,
    generate_match_explanation,
    generate_pdf_report,
    rewrite_resume_with_openrouter
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — NLP Matching System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp { background: #0b0f19; color: #e2e8f0; } /* Dark navy */
    .main { background-color: transparent; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    /* Typography */
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: #f8fafc !important; font-weight: 700; letter-spacing: -0.5px; }
    p, span, div, label { color: #94a3b8; }

    /* Hero Section */
    .hero-card {
        background: linear-gradient(145deg, #1e1b4b, #111827);
        border: 1px solid #312e81;
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        margin: 20px auto 30px auto;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        max-width: 400px;
    }
    .hero-verdict { font-size: 18px; font-weight: 600; margin-top: 10px; color: #818cf8; letter-spacing: 1px; text-transform: uppercase;}

    /* Metric Cards */
    .metric-card {
        background: #111827;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1f2937;
        margin-bottom: 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.4); border-color: #374151; }
    .metric-title { font-size: 12px; letter-spacing: 1px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; justify-content: center; gap: 6px;}
    .metric-value { font-size: 28px; font-weight: 700; color: #f1f5f9; }
    
    /* Progress Bars inside cards */
    .prog-mini-track { background: #1f2937; border-radius: 99px; height: 6px; margin-top: 12px; overflow: hidden; }
    .prog-mini-fill { height: 100%; border-radius: 99px; }

    /* General Progress */
    .prog-wrap { margin: 12px 0; }
    .prog-label { display: flex; justify-content: space-between; font-size: 13px; font-weight: 500; margin-bottom: 6px; color: #cbd5e1; }
    .prog-track { background: #1f2937; border-radius: 99px; height: 8px; overflow: hidden; }
    .prog-fill { height: 100%; border-radius: 99px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; padding: 6px; gap: 4px; border: 1px solid #1f2937; margin-bottom: 20px;}
    .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 500; font-size: 14px; color: #94a3b8 !important; padding: 8px 16px;}
    .stTabs [aria-selected="true"] { background: #4f46e5 !important; color: white !important; }

    /* Section Cards */
    .section-card { background: #111827; border-radius: 12px; padding: 24px; border: 1px solid #1f2937; margin-bottom: 16px; }

    /* Status Cards (ATS / Suggestions) */
    .status-card { font-size: 14px; margin: 10px 0; padding: 16px 20px; border-radius: 10px; line-height: 1.5; }
    .status-error { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; color: #fca5a5 !important; }
    .status-success { background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; color: #86efac !important; }
    .status-info { background: rgba(99, 102, 241, 0.1); border-left: 4px solid #6366f1; color: #a5b4fc !important; }

    /* Keyword Tags */
    .kw-matched { display:inline-block; background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 6px; padding: 4px 10px; margin: 4px; font-size: 12px; font-weight: 500; }
    .kw-missing { display:inline-block; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; padding: 4px 10px; margin: 4px; font-size: 12px; font-weight: 500; opacity: 0.8; }
    
    /* Entity Pills */
    .entity-pill { display: inline-block; background: #1f2937; border: 1px solid #374151; color: #e2e8f0; border-radius: 6px; padding: 4px 10px; margin: 2px; font-size: 12px; font-weight: 500; }

    /* Buttons */
    .stButton button {
        background: #4f46e5 !important; color: white !important; border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; padding: 12px 24px !important; transition: background 0.2s !important;
    }
    .stButton button:hover { background: #4338ca !important; }

    .stDownloadButton button { background: #10b981 !important; color: white !important; border-radius: 8px !important; border: none !important;}
    .stDownloadButton button:hover { background: #059669 !important; }

    /* Inputs */
    .stTextArea textarea { background: #111827 !important; border: 1px solid #1f2937 !important; border-radius: 8px !important; color: #f8fafc !important; }
    .stTextArea textarea:focus { border-color: #6366f1 !important; box-shadow: none !important;}
    
    hr { border-color: #1f2937 !important; margin: 32px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ResumeIQ")
    st.markdown("**NLP-Powered Resume Analysis**")
    st.divider()
    st.markdown("""
    ### How it works
    1. 📤 Upload or paste your **Resume**
    2. 📋 Upload or paste the **Job Description**
    3. 🔍 Click **Analyze** to run:
       - Sentence-BERT semantic similarity
       - TF-IDF keyword extraction
       - Skill gap analysis with taxonomy
       - ATS friendliness scoring
       - Named entity recognition
    """)
    st.divider()
    st.markdown("### Score Breakdown")
    st.markdown("**Weights dynamically adjust based on JD type:**")
    st.markdown("*Tech Jobs (>3 hard skills):*")
    st.markdown("🤖 Semantic 35% | 🛠️ Skills 35% | 🔑 Keywords 20% | 📋 ATS 10%")
    st.markdown("*Non-Tech Jobs:*")
    st.markdown("🤖 Semantic 40% | 🔑 Keywords 40% | 🛠️ Skills 10% | 📋 ATS 10%")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🧠 ResumeIQ — Automated Resume–JD Matching")
st.markdown("*Multi-signal NLP analysis: Semantic similarity · TF-IDF keywords · Skill taxonomy · ATS scoring*")
st.divider()


# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Resume Input")
    resume_mode = st.radio("Input mode:", ("Paste Text", "Upload PDF"), key="res_mode", horizontal=True)
    resume_text = ""
    if resume_mode == "Upload PDF":
        resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="res_file")
        if resume_file:
            resume_text = extract_text_from_pdf(resume_file)
            if resume_text and "Could not extract text properly" not in resume_text:
                st.success(f"✅ Extracted {len(resume_text.split())} words")
            else:
                st.error("Could not extract text.")
    else:
        resume_text = st.text_area("Paste resume here", height=280, key="res_text",
                                    placeholder="Paste your complete resume text...")

with col2:
    st.subheader("💼 Job Description Input")
    jd_mode = st.radio("Input mode:", ("Paste Text", "Upload PDF"), key="jd_mode", horizontal=True)
    jd_text = ""
    if jd_mode == "Upload PDF":
        jd_file = st.file_uploader("Upload JD PDF", type=["pdf"], key="jd_file")
        if jd_file:
            jd_text = extract_text_from_pdf(jd_file)
            if jd_text and "Could not extract text properly" not in jd_text:
                st.success(f"✅ Extracted {len(jd_text.split())} words")
            else:
                st.error("Could not extract text.")
    else:
        jd_text = st.text_area("Paste job description here", height=280, key="jd_text",
                                placeholder="Paste the complete job description...")

st.divider()

# ─────────────────────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────────────────────
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if st.button("🔍 Run Full NLP Analysis", use_container_width=True):
    if not resume_text.strip() or not jd_text.strip():
        st.error("⚠️ Please provide both Resume and Job Description.")
        st.stop()
    st.session_state.analysis_complete = True

if st.session_state.analysis_complete:

    # ── Step 1: NLP Analysis ──
    with st.spinner("🤖 Running Sentence-BERT semantic analysis..."):
        semantic_score = calculate_similarity(resume_text, jd_text)

    with st.spinner("🔑 Extracting TF-IDF keywords..."):
        jd_keywords = extract_tfidf_keywords(jd_text, top_n=15)
        resume_keywords = extract_tfidf_keywords(resume_text, top_n=15)
        kw_coverage, matched_kws, missing_kws = keyword_coverage_score(resume_text, jd_text, top_n=20)

    with st.spinner("🛠️ Analyzing skill taxonomy..."):
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        
        # Use new semantic matching instead of exact string matching
        matched_skills, missing_skills = get_semantic_skill_matches(resume_text, resume_skills, jd_skills)
        
        resume_skills_by_cat = extract_skills_by_category(resume_text)
        jd_skills_by_cat = extract_skills_by_category(jd_text)
        
        # Pass semantically matched skills to coverage calculator
        category_coverage = get_category_coverage(matched_skills, jd_skills_by_cat)
        
        skill_match_pct = round((len(matched_skills) / len(jd_skills) * 100) if jd_skills else 100.0, 1)

    with st.spinner("📋 Scoring ATS compatibility..."):
        ats_score, ats_issues, ats_tips = calculate_ats_score(resume_text)
        entities = extract_resume_entities(resume_text)
        jd_entities = extract_jd_requirements(jd_text)

    composite_score = calculate_composite_score(semantic_score, kw_coverage, skill_match_pct, ats_score, len(jd_skills))
    section_scores = calculate_section_scores(entities, jd_entities, skill_match_pct)

    # ─────────────────────────────────────────────
    # RESULTS — COMPOSITE SCORE + METRICS
    # ─────────────────────────────────────────────
    st.header("📊 Analysis Results")

    # ── Hero Section ──
    verdict_text = "Excellent" if composite_score >= 75 else "Good" if composite_score >= 50 else "Needs Work"
    
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 14px; font-weight: 600; color: #94a3b8; letter-spacing: 2px; text-transform: uppercase;">Composite Match Score</div>
        <div style="font-size: 84px; font-weight: 800; line-height: 1; color: #818cf8; margin: 10px 0;">{composite_score}%</div>
        <div class="hero-verdict">{verdict_text} Match</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Row ──
    m1, m2, m3, m4 = st.columns(4)
    signal_metrics = [
        ("Semantic", semantic_score, "%", "🧠"),
        ("Keywords", kw_coverage, "%", "🔑"),
        ("Skills", skill_match_pct, "%", "🛠️"),
        ("ATS Health", ats_score, "/100", "📋"),
    ]
    for (title, val, suffix, icon), col in zip(signal_metrics, [m1, m2, m3, m4]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{icon} {title}</div>
                <div class="metric-value">{val}{suffix}</div>
                <div class="prog-mini-track">
                    <div class="prog-mini-fill" style="width:{min(val,100)}%; background:#4f46e5;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # DETAILED ANALYSIS TABS
    # ─────────────────────────────────────────────
    tab_overview, tab_skills, tab_keywords, tab_ats, tab_suggestions = st.tabs([
        "Overview", "Skills", "Keywords", "ATS & Resume Health", "Suggestions & AI"
    ])

    with tab_overview:
        st.subheader("Overview & Breakdown")
        
        # Move section scores here
        def progress_bar(label, value):
            return f"""
            <div class="prog-wrap">
                <div class="prog-label"><span>{label}</span><span>{value}%</span></div>
                <div class="prog-track"><div class="prog-fill" style="width:{value}%;background:#4f46e5;"></div></div>
            </div>"""

        score_col1, score_col2 = st.columns([1, 1.5])
        with score_col1:
            st.markdown("#### Section Scores")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                progress_bar("Skills", section_scores['skills']) +
                progress_bar("Experience", section_scores['experience']) +
                progress_bar("Education", section_scores['education']),
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with score_col2:
            st.markdown("#### Match Explanation")
            with st.spinner("Analyzing match..."):
                explanations, strengths, weaknesses = generate_match_explanation(
                    composite_score, semantic_score, kw_coverage, skill_match_pct, ats_score,
                    matched_skills, missing_skills, matched_kws, missing_kws, len(jd_skills)
                )
                sentence_matches = get_sentence_matches(resume_text, jd_text, top_n=4)
            
            for exp in explanations:
                st.markdown(f'<div class="status-card status-info">ℹ️ {exp}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("##### Strengths")
            if strengths:
                for s in strengths[:3]: st.markdown(f'<div class="status-card status-success">✅ {s}</div>', unsafe_allow_html=True)
            else: st.info("No strong areas detected.")
        with s_col2:
            st.markdown("##### Weaknesses")
            if weaknesses:
                for w in weaknesses[:3]: st.markdown(f'<div class="status-card status-error">⚠️ {w}</div>', unsafe_allow_html=True)
            else: st.success("No significant weaknesses found.")

        if sentence_matches:
            st.markdown("<br>#### Top Matching Lines", unsafe_allow_html=True)
            for idx, (res_sent, jd_sent, sim_score) in enumerate(sentence_matches, 1):
                st.markdown(f"""
                <div class="section-card" style="padding: 16px;">
                    <div style="font-size:12px; color:#64748b; font-weight:600; margin-bottom:8px;">Match #{idx} · {sim_score}% Similarity</div>
                    <div style="margin-bottom:8px; font-size:14px;"><span style="color:#818cf8; font-weight:600;">Resume:</span> {res_sent}</div>
                    <div style="font-size:14px;"><span style="color:#94a3b8; font-weight:600;">JD:</span> {jd_sent}</div>
                </div>
                """, unsafe_allow_html=True)


    with tab_skills:
        radar_col, skill_col = st.columns([1, 1])
        with radar_col:
            st.markdown("#### Skill Category Radar")
            # Only show categories that the JD actually requires
            radar_cats = list(category_coverage.keys())
            if len(radar_cats) >= 3:
                vals = [category_coverage[c] for c in radar_cats]
                vals_closed = vals + [vals[0]]
                cats_closed = radar_cats + [radar_cats[0]]
                fig = go.Figure(go.Scatterpolar(
                    r=vals_closed, theta=cats_closed,
                    fill='toself',
                    fillcolor='rgba(99, 102, 241, 0.25)',
                    line=dict(color='#6366f1', width=2),
                    marker=dict(size=6, color='#4f46e5', symbol='circle')
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#64748b"), gridcolor="rgba(255,255,255,0.05)"),
                        angularaxis=dict(tickfont=dict(size=12, color="#cbd5e1"), gridcolor="rgba(255,255,255,0.05)")
                    ),
                    showlegend=False,
                    margin=dict(t=40, b=40, l=40, r=40),
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color="#f8fafc")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough skill categories detected in JD for radar chart (need at least 3).")

        with skill_col:
            st.markdown("#### Skills Overview")
            sub_tab_res, sub_tab_jd, sub_tab_gap = st.tabs(["Resume Skills", "JD Skills", "Skill Gap"])

            with sub_tab_res:
                if resume_skills:
                    pills = "".join([f'<span class="kw-matched">{s.title()}</span>' for s in sorted(resume_skills)])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.info("No technical skills found in Resume.")

            with sub_tab_jd:
                if jd_skills:
                    pills = "".join([f'<span class="entity-pill">{s.title()}</span>' for s in sorted(jd_skills)])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.info("No technical skills found in JD.")

            with sub_tab_gap:
                if missing_skills:
                    pills = "".join([f'<span class="kw-missing">{s.title()}</span>' for s in sorted(missing_skills)])
                    st.markdown(f"<div style='margin-bottom: 12px; color: #f87171; font-weight: 600;'>{len(missing_skills)} skills missing:</div>", unsafe_allow_html=True)
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-card status-success">🎉 No skill gaps detected!</div>', unsafe_allow_html=True)

    with tab_keywords:
        st.markdown("#### TF-IDF Keyword Analysis")

        # Coverage summary bar
        cov_color = "#22c55e" if kw_coverage >= 70 else "#f59e0b" if kw_coverage >= 40 else "#ef4444"
        st.markdown(f"""
        <div class="section-card" style="padding:20px 24px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <span style="font-weight:600;font-size:14px;color:#e2e8f0;">JD Keyword Coverage</span>
                <span style="font-size:22px;font-weight:800;color:{cov_color};">{kw_coverage}%</span>
            </div>
            <div style="background:#1f2937;border-radius:99px;height:8px;overflow:hidden;">
                <div style="width:{kw_coverage}%;height:100%;border-radius:99px;background:{cov_color};"></div>
            </div>
            <div style="margin-top:10px;font-size:13px;color:#94a3b8;">{len(matched_kws)} matched · {len(missing_kws)} missing out of {len(matched_kws)+len(missing_kws)} JD keywords</div>
        </div>
        """, unsafe_allow_html=True)

        kw1, kw2 = st.columns(2)
        with kw1:
            st.markdown("##### Matched Keywords")
            if matched_kws:
                tags = "".join([f'<span class="kw-matched">{k}</span>' for k in matched_kws[:15]])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.info("No keywords matched.")
        with kw2:
            st.markdown("##### Missing Keywords")
            if missing_kws:
                tags = "".join([f'<span class="kw-missing">{k}</span>' for k in missing_kws[:15]])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-card status-success">🎉 No keywords missing!</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Keyword importance bar chart
        if jd_keywords:
            st.markdown("##### Top JD Keywords by Importance (TF-IDF)")
            kw_names = [kw for kw, _ in jd_keywords[:10]]
            kw_scores = [sc for _, sc in jd_keywords[:10]]
            fig2 = go.Figure(go.Bar(
                x=kw_scores, y=kw_names, orientation='h',
                marker_color=['#4ade80' if kw in matched_kws else '#f87171' for kw in kw_names],
                text=[f"{s:.3f}" for s in kw_scores], textposition='auto'
            ))
            fig2.update_layout(
                height=350, margin=dict(t=20, b=20, l=20, r=20),
                xaxis_title="TF-IDF Weight", 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cbd5e1", family="Inter"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab_ats:
        ner_col1, ner_col2 = st.columns([1, 1])
        
        with ner_col1:
            st.markdown("#### ATS Compatibility")
            ats_color = "#22c55e" if ats_score >= 75 else "#f59e0b" if ats_score >= 50 else "#ef4444"
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-bottom:12px;">
                <span style="font-weight:600;font-size:16px;color:#e2e8f0;margin-right:12px;">ATS Score:</span>
                <span style="font-size:28px;font-weight:800;color:{ats_color};">{ats_score}/100</span>
            </div>
            <div style="background:#1f2937;border-radius:99px;height:10px;margin-bottom:20px;overflow:hidden;">
                <div style="width:{ats_score}%;height:100%;border-radius:99px;background:{ats_color};"></div>
            </div>
            """, unsafe_allow_html=True)

            if ats_issues:
                for issue in ats_issues:
                    st.markdown(f'<div class="status-card status-error">⚠️ {issue}</div>', unsafe_allow_html=True)
            if ats_tips:
                for tip in ats_tips:
                    st.markdown(f'<div class="status-card status-success">💡 {tip}</div>', unsafe_allow_html=True)

        with ner_col2:
            st.markdown("#### Extracted Resume Entities")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            
            if entities.get("names"): st.markdown(f"**👤 Name:** `{entities['names'][0]}`")
            if entities.get("emails"): st.markdown(f"**📧 Email:** `{entities['emails'][0]}`")
            if entities.get("phones"): st.markdown(f"**📞 Phone:** `{entities['phones'][0]}`")
            if entities.get("experience_years"): st.markdown(f"**⏱️ Experience:** `{entities['experience_years']} years`")
            if entities.get("companies"): st.markdown(f"**🏢 Companies:** " + " · ".join(entities["companies"]))
            if entities.get("education"): st.markdown(f"**🎓 Degrees:** " + " · ".join(entities["education"][:3]))
            if entities.get("job_titles"):
                pills = "".join([f'<span class="entity-pill">{t}</span>' for t in entities["job_titles"][:4]])
                st.markdown(f"**👔 Roles detected:** <br>{pills}", unsafe_allow_html=True)
            if entities.get("certifications"):
                st.markdown("**📜 Certifications:**")
                for cert in entities["certifications"][:3]: st.markdown(f"- `{cert}`")
            
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_suggestions:
        col_sugg, col_act = st.columns([1.2, 1])
        
        with col_sugg:
            st.markdown("#### Actionable Suggestions")
            if missing_skills or missing_kws:
                suggestions = generate_specific_suggestions(missing_skills, missing_kws)
                for suggestion in suggestions:
                    st.markdown(f'<div class="status-card status-info">🎯 {suggestion}</div>', unsafe_allow_html=True)
                st.markdown('<div class="status-card status-info">📊 **Quantify Achievements:** Add metrics to your bullet points (e.g., increased sales by 20%, managed team of 5).</div>', unsafe_allow_html=True)
                st.markdown('<div class="status-card status-info">📝 **Tailor Your Summary:** Ensure your professional summary reflects the core requirements of the JD.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-card status-success">🎉 Outstanding match! Just ensure formatting is clean and bullet points are impact-driven.</div>', unsafe_allow_html=True)

        with col_act:
            st.markdown("#### Next Steps & Tools")
            
            st.markdown('<div class="section-card" style="border: 1px solid #4f46e5;">', unsafe_allow_html=True)
            st.markdown("##### ✨ AI Resume Rewriter")
            st.markdown("<p style='font-size:13px;color:#94a3b8;'>Use AI to automatically rewrite your professional summary and bullet points to include missing keywords naturally.</p>", unsafe_allow_html=True)
            if st.button("🚀 Rewrite with AI", use_container_width=True):
                with st.spinner("AI is analyzing and rewriting..."):
                    st.session_state.rewritten_resume = rewrite_resume_with_openrouter(resume_text, jd_text)
            
            if "rewritten_resume" in st.session_state:
                st.markdown("<br>###### 📝 Suggested Revisions", unsafe_allow_html=True)
                st.markdown('<div style="background:#0f172a; padding:16px; border-radius:8px; font-size:13px;">', unsafe_allow_html=True)
                st.markdown(st.session_state.rewritten_resume)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card" style="border: 1px solid #10b981;">', unsafe_allow_html=True)
            st.markdown("##### 📊 Download Analysis Report")
            st.markdown("<p style='font-size:13px;color:#94a3b8;'>Get a professional PDF report with your complete analysis, scores, and suggestions to share or review later.</p>", unsafe_allow_html=True)
            
            pdf_suggestions = generate_specific_suggestions(missing_skills, missing_kws) if (missing_skills or missing_kws) else []
            pdf_bytes = generate_pdf_report(
                composite_score, semantic_score, kw_coverage, skill_match_pct, ats_score,
                section_scores, matched_skills, missing_skills, matched_kws, missing_kws,
                explanations, strengths, weaknesses, pdf_suggestions, sentence_matches
            )
            
            st.download_button(
                label="📥 Export PDF Report",
                data=pdf_bytes,
                file_name="ResumeIQ_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("💡 ResumeIQ uses Sentence-BERT (all-MiniLM-L6-v2) for semantic analysis · TF-IDF for keyword weighting · Rule-based NER · ATS heuristics")