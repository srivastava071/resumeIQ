# utils.py
import pdfplumber
import re
import math
import streamlit as st
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from skills import get_skills, get_skills_taxonomy, get_skill_category, are_false_positive_pair, SKILL_ALIASES
import spacy
from spacy.cli import download
import os
from dotenv import load_dotenv
from openai import OpenAI
# Force reload environment variables so it picks up .env changes without restarting
load_dotenv(override=True)

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# ─────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        
    text = text.strip()
    if not text:
        # TODO: Advanced later - Implement OCR fallback here (e.g., pytesseract)
        text = "Could not extract text properly"
        
    return text


# ─────────────────────────────────────────────
# SKILL EXTRACTION
# ─────────────────────────────────────────────

def extract_skills(text):
    """Extract skills from text based on predefined taxonomy."""
    skills = get_skills()
    text_lower = text.lower()
    found_skills = set()
    for skill in skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return list(found_skills)


def extract_skills_by_category(text):
    """Return skills grouped by category."""
    taxonomy = get_skills_taxonomy()
    text_lower = text.lower()
    result = {}
    for category, skills in taxonomy.items():
        found = []
        for skill in skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill)
        result[category] = found
    return result


def get_missing_skills(resume_skills, jd_skills):
    """Identify skills in JD but missing from resume (legacy exact match)."""
    return list(set(jd_skills) - set(resume_skills))


def skill_similarity(skill1, skill2):
    """Calculate semantic similarity between two single skills."""
    emb = model.encode([skill1, skill2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]


def get_semantic_skill_matches(resume_text, resume_skills, jd_skills, threshold=0.82):
    """
    Find matched and missing skills using exact/alias matching first,
    then semantic embeddings as a fallback.

    Key fixes vs. original:
    - Removed unreliable abbreviation harvesting (random CAPS from resume text
      caused false-positive semantic matches, e.g. 'AWS' matching 'Angular').
    - Raised semantic threshold to 0.82 (was 0.75) to cut false positives.
    - Added substring containment check before semantic comparison so
      "node.js" in resume_text matches the JD skill "node" reliably.
    - Alias normalisation is applied on both sides before any comparison.
    """
    if not jd_skills:
        return [], []

    def normalize(skill):
        return SKILL_ALIASES.get(skill.lower(), skill.lower())

    resume_normalized = set(normalize(s) for s in resume_skills)
    resume_lower = resume_text.lower()

    # Candidate pool: only actual extracted skills (no random abbreviations)
    candidates = list(resume_normalized)

    matched = []
    missing = []

    if candidates:
        cand_embeddings = model.encode(candidates)
        jd_embeddings = model.encode(jd_skills)
    else:
        # No resume skills at all — everything is missing
        return [], list(jd_skills)

    for i, jd_skill in enumerate(jd_skills):
        jd_norm = normalize(jd_skill)

        # 1. Exact / alias match
        if jd_norm in resume_normalized or jd_skill.lower() in resume_normalized:
            matched.append(jd_skill)
            continue

        # 2. Substring containment: JD skill appears verbatim in resume text
        #    (catches multi-word skills like "machine learning", "ruby on rails")
        if re.search(r'\b' + re.escape(jd_skill.lower()) + r'\b', resume_lower):
            matched.append(jd_skill)
            continue
        if re.search(r'\b' + re.escape(jd_norm) + r'\b', resume_lower):
            matched.append(jd_skill)
            continue

        # 3. Semantic match — only between actual extracted skills
        jd_emb = jd_embeddings[i]
        similarities = cosine_similarity([jd_emb], cand_embeddings)[0]
        best_idx = int(similarities.argmax())
        max_sim = float(similarities[best_idx])
        best_candidate = candidates[best_idx]

        if max_sim >= threshold and not are_false_positive_pair(jd_skill, best_candidate):
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return matched, missing


# ─────────────────────────────────────────────
# SEMANTIC SIMILARITY (Sentence-BERT)
# ─────────────────────────────────────────────

def clean_text(text):
    """Light cleaning — only collapses whitespace. Preserves natural language for SBERT."""
    # Only normalize whitespace; keep punctuation and casing so SBERT gets proper context.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _chunk_text(text, max_words=400):
    """
    Produce a representative sample of the text for SBERT encoding.
    Takes beginning, middle, and end thirds so that work-experience
    content (typically in the middle) is not silently dropped.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    third = max_words // 3
    mid_start = (len(words) - third) // 2
    return (
        ' '.join(words[:third])
        + ' ... '
        + ' '.join(words[mid_start: mid_start + third])
        + ' ... '
        + ' '.join(words[-third:])
    )


def calculate_similarity(text1, text2):
    """
    Calculate cosine similarity using Sentence-BERT embeddings.
    Feeds natural (lightly cleaned) text so SBERT can use full linguistic context.
    """
    if not text1 or not text2:
        return 0.0

    # Light clean only — preserve natural language for better SBERT encoding
    t1 = _chunk_text(clean_text(text1))
    t2 = _chunk_text(clean_text(text2))

    embeddings = model.encode([t1, t2])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(max(0, min(100, score * 100))), 2)


# ─────────────────────────────────────────────
# TF-IDF KEYWORD ANALYSIS
# ─────────────────────────────────────────────

def extract_tfidf_keywords(text, top_n=15, reference_text=None):
    """
    Extract top TF-IDF keywords from `text`.
    If `reference_text` is provided, IDF is computed across both documents
    for proper cross-document keyword weighting (JD vs Resume context).
    """
    sentences = re.split(r'[.\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) < 2:
        sentences = [text, text]

    try:
        if reference_text:
            # Build a joint corpus: target doc sentences + reference doc sentences
            # This gives proper IDF weights relative to both documents
            ref_sentences = re.split(r'[.\n]', reference_text)
            ref_sentences = [s.strip() for s in ref_sentences if len(s.strip()) > 10]
            corpus = sentences + (ref_sentences if ref_sentences else [reference_text])
        else:
            corpus = sentences

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=300,
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9.+#]{1,}\b'  # min 2 chars, must start with letter
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Score only the target document's sentences (first len(sentences) rows)
        target_scores = tfidf_matrix[:len(sentences)].sum(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()
        keyword_scores = sorted(zip(feature_names, target_scores), key=lambda x: x[1], reverse=True)
        return [(kw, round(float(sc), 3)) for kw, sc in keyword_scores[:top_n]]
    except Exception:
        return []


def keyword_coverage_score(resume_text, jd_text, top_n=20, threshold=0.82):
    """
    Calculate what % of JD's top TF-IDF keywords appear in the resume.

    Key fixes vs. original:
    - Threshold raised from 0.72 → 0.82 to reduce spurious semantic matches
      (e.g. "data" semantically matching "database" at 0.72).
    - Exact match now uses word-boundary regex instead of plain substring,
      so "node" in resume text no longer matches the JD keyword "nodejs".
    - Single-character keywords are skipped (they are almost always noise).
    """
    jd_keywords = [kw for kw, _ in extract_tfidf_keywords(jd_text, top_n, reference_text=resume_text)]
    if not jd_keywords:
        return 0, [], []

    resume_keywords = [kw for kw, _ in extract_tfidf_keywords(resume_text, top_n=40, reference_text=jd_text)]
    resume_lower = resume_text.lower()

    if resume_keywords:
        res_emb = model.encode(resume_keywords)
        jd_emb = model.encode(jd_keywords)
    else:
        res_emb = []
        jd_emb = model.encode(jd_keywords)

    matched = []
    missing = []

    for i, jd_kw in enumerate(jd_keywords):
        # Skip single-char noise tokens
        if len(jd_kw.strip()) <= 1:
            missing.append(jd_kw)
            continue

        # 1. Word-boundary exact match (more precise than plain substring)
        if re.search(r'\b' + re.escape(jd_kw.lower()) + r'\b', resume_lower):
            matched.append(jd_kw)
            continue

        # 2. Semantic match against resume keywords at tighter threshold
        if len(resume_keywords) > 0:
            similarities = cosine_similarity([jd_emb[i]], res_emb)[0]
            max_sim = float(similarities.max()) if len(similarities) > 0 else 0.0
            if max_sim >= threshold:
                matched.append(jd_kw)
                continue

        missing.append(jd_kw)

    coverage = round(len(matched) / len(jd_keywords) * 100, 1)
    return coverage, matched, missing


# ─────────────────────────────────────────────
# NAMED ENTITY RECOGNITION (Rule-based)
# ─────────────────────────────────────────────

def extract_resume_entities(text):
    """Extraction of key resume entities using spaCy and Regex."""
    entities = {
        "names": [],
        "emails": [],
        "phones": [],
        "education": [],
        "experience_years": None,
        "companies": [],
        "job_titles": [],
        "certifications": []
    }

    # Process with spaCy
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["companies"].append(ent.text.strip())
        elif ent.label_ == "PERSON":
            entities["names"].append(ent.text.strip())
            
    # Filter and deduplicate
    entities["companies"] = list(set([c for c in entities["companies"] if len(c) > 2]))[:5]
    entities["names"] = list(set([n for n in entities["names"] if len(n) > 2]))[:2]

    # Email
    entities["emails"] = re.findall(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', text, re.I)

    # Phone
    entities["phones"] = re.findall(
        r'(\+?\d[\d\s\-().]{7,}\d)', text
    )[:2]

    # Education degrees
    degree_patterns = [
        r'\b(B\.?Tech|B\.?E\.?|B\.?Sc\.?|Bachelor[s]?(?:\s+of\s+\w+)?)\b',
        r'\b(M\.?Tech|M\.?E\.?|M\.?Sc\.?|Master[s]?(?:\s+of\s+\w+)?)\b',
        r'\b(Ph\.?D|Doctorate)\b',
        r'\b(MBA|MCA|BCA|B\.?Com|M\.?Com)\b',
        r'\b(Associate[s]? Degree|Diploma)\b'
    ]
    for pat in degree_patterns:
        found = re.findall(pat, text, re.I)
        entities["education"].extend(found)
    entities["education"] = list(set(entities["education"]))

    # Years of experience
    exp_match = re.search(
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', text, re.I
    )
    if exp_match:
        entities["experience_years"] = int(exp_match.group(1))

    # Common job titles
    title_keywords = [
        "software engineer", "data scientist", "data analyst", "ml engineer",
        "backend developer", "frontend developer", "full stack developer",
        "devops engineer", "cloud architect", "product manager", "tech lead",
        "senior engineer", "junior developer", "research scientist", "ai engineer"
    ]
    text_lower = text.lower()
    for title in title_keywords:
        if title in text_lower:
            entities["job_titles"].append(title.title())

    # Certifications
    cert_patterns = [
        r'\b(AWS Certified[\w\s]+?(?=\n|,|\.))',
        r'\b(Google Certified[\w\s]+?(?=\n|,|\.))',
        r'\b(Microsoft Certified[\w\s]+?(?=\n|,|\.))',
        r'\b(CPA|CFA|PMP|CISSP|CEH|CISM|OSCP)\b',
        r'\b(Certified\s+\w+(?:\s+\w+)?(?:\s+Professional|Engineer|Developer|Associate)?)\b'
    ]
    for pat in cert_patterns:
        found = re.findall(pat, text, re.I)
        entities["certifications"].extend([f.strip() for f in found])
    entities["certifications"] = list(set(entities["certifications"]))[:5]

    return entities


# ─────────────────────────────────────────────
# ATS FRIENDLINESS SCORE
# ─────────────────────────────────────────────

def calculate_ats_score(resume_text):
    """
    Realistic ATS friendliness scorer built on an ADDITIVE (earn-your-points)
    model instead of the old start-at-100-and-barely-deduct model.

    Why the old scorer gave ~98 while real tools give 60-70:
      - Old: started at 100, checked only ~5 things, max deduction was ~35pts.
        A resume with email + phone + section headers + a few verbs = 95+.
      - Real ATS tools (Jobscan, Resume Worded, etc.) check 15+ dimensions
        and weight them heavily. This rewrite mirrors that approach.

    Scoring dimensions (total = 100 pts):
      Contact info        10 pts
      Section structure   20 pts
      Content length      10 pts
      Action verbs        15 pts
      Quantified impact   15 pts
      Formatting safety   10 pts  (no tables, columns, images, symbols)
      Readability signals 10 pts  (dates, job titles, consistent tense)
      Professional links   5 pts  (LinkedIn / GitHub)
      File/text quality    5 pts  (parseable, no garbled chars)
    """
    if not resume_text or resume_text == "Could not extract text properly":
        return 0, ["No readable text found"], ["Ensure your PDF is text-based, not an image."]

    issues = []
    tips = []
    score = 0
    text = resume_text
    text_lower = text.lower()
    word_count = len(text.split())

    # ── 1. Contact Information (10 pts) ─────────────────────────────────────
    contact_pts = 0

    if re.search(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', text, re.I):
        contact_pts += 4
    else:
        issues.append("No email address detected")
        tips.append("Include a professional email address — it's mandatory for ATS parsing.")

    if re.search(r'(\+?\d[\d\s\-().]{7,}\d)', text):
        contact_pts += 3
    else:
        issues.append("No phone number detected")
        tips.append("Add a phone number so recruiters can reach you.")

    if re.search(r'linkedin\.com/in/', text, re.I):
        contact_pts += 2
    else:
        tips.append("Add your LinkedIn profile URL (linkedin.com/in/yourname) — most ATS parsers extract it.")

    if re.search(r'github\.com/', text, re.I):
        contact_pts += 1
    else:
        tips.append("Consider adding your GitHub profile URL if you have public project work.")

    score += contact_pts

    # ── 2. Section Structure (20 pts) ───────────────────────────────────────
    # ATS parsers rely on clearly labelled section headers to bucket content.
    section_pts = 0

    critical_sections = {
        'experience': (r'\b(work\s+experience|experience|employment|work\s+history)\b', 7),
        'education':  (r'\b(education|academic|qualifications)\b', 5),
        'skills':     (r'\b(skills|technical\s+skills|core\s+competencies|expertise)\b', 5),
    }
    optional_sections = {
        'summary':    (r'\b(summary|objective|profile|about\s+me)\b', 2),
        'projects':   (r'\b(projects|portfolio|personal\s+projects)\b', 1),
    }

    missing_critical = []
    for sec, (pattern, pts) in critical_sections.items():
        if re.search(pattern, text, re.I):
            section_pts += pts
        else:
            missing_critical.append(sec.title())

    for sec, (pattern, pts) in optional_sections.items():
        if re.search(pattern, text, re.I):
            section_pts += pts

    if missing_critical:
        issues.append(f"Missing critical sections: {', '.join(missing_critical)}")
        tips.append(f"Add clearly labeled sections: {', '.join(missing_critical)}. ATS systems use these headers to categorise your content.")

    score += section_pts

    # ── 3. Content Length (10 pts) ──────────────────────────────────────────
    # Real ATS systems penalise both too-short and too-long resumes.
    if word_count < 80:
        issues.append(f"Resume is critically short ({word_count} words) — ATS cannot parse meaningful content")
        tips.append("Expand to at least 300 words covering experience, skills, and education.")
        # length_pts = 0
    elif word_count < 200:
        score += 2
        issues.append(f"Resume is too short ({word_count} words) — ATS prefers 300–700 words")
        tips.append("Add detail to your experience bullet points and skills section.")
    elif word_count < 300:
        score += 5
        issues.append(f"Resume is somewhat short ({word_count} words)")
        tips.append("Aim for 400–600 words for optimal ATS parsing depth.")
    elif word_count <= 800:
        score += 10  # Sweet spot
    elif word_count <= 1200:
        score += 7
        tips.append("Your resume is on the longer side. Consider trimming older or less-relevant roles.")
    else:
        score += 3
        issues.append(f"Resume is very long ({word_count} words) — ATS may truncate or misparse it")
        tips.append("Condense to 1–2 pages (600–900 words). Remove roles older than 10-15 years.")

    # ── 4. Action Verbs (15 pts) ────────────────────────────────────────────
    action_verbs = [
        'developed', 'designed', 'implemented', 'led', 'managed', 'built',
        'created', 'improved', 'optimized', 'analyzed', 'deployed', 'architected',
        'collaborated', 'delivered', 'increased', 'reduced', 'automated',
        'spearheaded', 'orchestrated', 'engineered', 'launched', 'streamlined',
        'integrated', 'migrated', 'refactored', 'mentored', 'trained',
        'researched', 'published', 'presented', 'negotiated', 'coordinated',
        'established', 'achieved', 'accelerated', 'administered', 'advised',
        'authored', 'championed', 'conceptualized', 'consolidated', 'directed',
        'drove', 'enabled', 'executed', 'facilitated', 'generated', 'guided',
        'handled', 'identified', 'influenced', 'initiated', 'maintained',
        'modernized', 'monitored', 'operated', 'oversaw', 'partnered',
        'planned', 'prioritized', 'produced', 'revamped', 'scaled', 'secured',
        'simplified', 'solved', 'standardized', 'supported', 'transformed'
    ]
    found_verbs = set(v for v in action_verbs if re.search(r'\b' + v + r'\b', text, re.I))
    n_verbs = len(found_verbs)

    if n_verbs == 0:
        issues.append("No strong action verbs found — bullet points appear passive")
        tips.append("Start each bullet point with an action verb: Developed, Led, Optimized, Built, etc.")
    elif n_verbs < 3:
        score += 4
        issues.append(f"Very few action verbs ({n_verbs}) — most bullet points may be passive")
        tips.append("Use varied action verbs at the start of each bullet: avoid repeating the same verb.")
    elif n_verbs < 6:
        score += 8
        tips.append("Good action verb usage. Aim for 8+ distinct verbs for stronger ATS signals.")
    elif n_verbs < 10:
        score += 12
    else:
        score += 15  # Full marks: rich, varied action vocabulary

    # ── 5. Quantified Achievements (15 pts) ─────────────────────────────────
    # ATS and recruiters both prioritise measurable impact.
    quant_patterns = [
        r'\d+\s*%',                   # percentages: 40%, 3.5%
        r'\d+x\b',                    # multipliers: 3x, 10x
        r'\$\s*[\d,]+',               # dollar amounts
        r'\b\d+\s*(users|customers|clients|engineers|employees|members|team)',
        r'\b(million|billion|k)\b',   # scale words
        r'\b\d+\s*(projects|products|features|services|systems)',
        r'(increased|decreased|reduced|improved|grew|saved|cut)\s+by\s+\d+',
        r'\b\d+\s*(hours|days|weeks|months)\b',  # time savings
    ]
    quant_hits = sum(1 for p in quant_patterns if re.search(p, text, re.I))

    if quant_hits == 0:
        issues.append("No quantified achievements detected (no numbers, percentages, or metrics)")
        tips.append("Add impact metrics: 'Reduced load time by 35%', 'Managed team of 6', 'Saved $20K annually'.")
    elif quant_hits == 1:
        score += 5
        issues.append("Very few metrics found — only 1 quantified achievement detected")
        tips.append("Add at least 3–5 bullet points with measurable results across your roles.")
    elif quant_hits <= 3:
        score += 9
        tips.append("Some metrics found. Add more numbers across different roles for stronger impact.")
    elif quant_hits <= 6:
        score += 12
    else:
        score += 15  # Excellent: data-driven resume

    # ── 6. Formatting Safety (10 pts) ───────────────────────────────────────
    # Tables, text boxes, columns, and decorative symbols break most ATS parsers.
    fmt_pts = 10

    # Decorative / non-ASCII symbols common in designed resumes
    special_chars = re.findall(r'[★✓✔●■◆▶→©®™•‣⁃◦]', text)
    if len(special_chars) > 5:
        fmt_pts -= 4
        issues.append(f"Many decorative symbols found ({len(special_chars)}) — these can corrupt ATS parsing")
        tips.append("Replace symbols (★, ✓, ●, ■) with plain text or standard hyphens.")
    elif special_chars:
        fmt_pts -= 2
        tips.append("A few decorative symbols detected. Keep formatting simple for ATS safety.")

    # Garbled / non-printable characters (sign of PDF extraction failure)
    garbled = re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', text)
    if len(garbled) > 10:
        fmt_pts -= 4
        issues.append("Garbled/unprintable characters detected — your PDF may not parse cleanly")
        tips.append("Save your resume as a standard PDF (not image-based or heavily formatted).")

    # Very long lines with no newline breaks (possible table row or text box)
    lines = text.split('\n')
    long_lines = [l for l in lines if len(l.split()) > 40]
    if len(long_lines) > 3:
        fmt_pts -= 2
        issues.append("Unusually long text lines detected — may indicate tables or columns that confuse ATS")
        tips.append("Avoid multi-column layouts and tables. Use a single-column format.")

    score += max(0, fmt_pts)

    # ── 7. Readability Signals (10 pts) ─────────────────────────────────────
    readability_pts = 0

    # Date patterns — ATS needs dates to parse work history timeline
    date_patterns = [
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b',
        r'\b\d{1,2}/\d{4}\b',
        r'\b(20\d{2}|19\d{2})\b',
        r'\b(present|current|till\s+date|to\s+date)\b',
    ]
    date_hits = sum(1 for p in date_patterns if re.search(p, text, re.I))
    if date_hits >= 2:
        readability_pts += 5
    elif date_hits == 1:
        readability_pts += 2
        tips.append("Add more date ranges to your experience entries (e.g., 'Jan 2021 – Mar 2023').")
    else:
        issues.append("No employment dates detected — ATS cannot build a timeline of your experience")
        tips.append("Add start and end dates to every role: 'Software Engineer, Jan 2021 – Present'.")

    # Job title presence — ATS extracts titles to classify the candidate
    generic_title_pattern = r'\b(engineer|developer|analyst|manager|designer|scientist|consultant|architect|lead|intern|specialist|coordinator)\b'
    if re.search(generic_title_pattern, text, re.I):
        readability_pts += 3
    else:
        tips.append("Ensure job titles are clearly written (e.g., 'Software Engineer', 'Data Analyst').")

    # Consistent past tense for past roles (readability signal)
    past_tense_verbs = len(re.findall(r'\b\w+ed\b', text))
    if past_tense_verbs >= 5:
        readability_pts += 2

    score += readability_pts

    # ── 8. Professional Links (5 pts) ───────────────────────────────────────
    # Already counted LinkedIn (2) and GitHub (1) in contact section.
    # Bonus here for having a portfolio or personal site.
    if re.search(r'https?://(?!linkedin|github)\S+', text, re.I):
        score += 2  # Portfolio / personal site

    # ── 9. Text / File Quality (5 pts) ──────────────────────────────────────
    quality_pts = 5
    # Check ratio of alphabetic chars (low ratio = garbled / table-heavy PDF)
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = max(len(text), 1)
    alpha_ratio = alpha_chars / total_chars
    if alpha_ratio < 0.5:
        quality_pts -= 3
        issues.append("Low readable text ratio — resume may contain images, tables, or be poorly extracted")
        tips.append("Use a plain, ATS-safe resume template. Avoid heavy graphics or image-based text.")
    elif alpha_ratio < 0.65:
        quality_pts -= 1

    # Repeated whitespace / newlines (sign of layout artefacts)
    excess_whitespace = len(re.findall(r'\n{4,}', text))
    if excess_whitespace > 5:
        quality_pts -= 1
        tips.append("Large blank gaps detected. Clean up whitespace for cleaner ATS parsing.")

    score += max(0, quality_pts)

    # ── Final clamp ─────────────────────────────────────────────────────────
    final_score = max(0, min(100, round(score)))
    return final_score, issues, tips


# ─────────────────────────────────────────────
# RADAR CHART DATA
# ─────────────────────────────────────────────

def get_category_coverage(matched_skills, jd_skills_by_cat):
    """
    For each skill category that the JD actually requires, calculate what %
    of those required skills are covered by the resume's matched skills.

    Fix vs. original: categories with zero JD skills are now excluded entirely
    (returned as None) rather than given an artificial 100%, which was
    inflating the radar chart for irrelevant categories.
    """
    categories = list(jd_skills_by_cat.keys())
    coverage = {}
    for cat in categories:
        jd_cat_skills = set(jd_skills_by_cat.get(cat, []))
        if not jd_cat_skills:
            # Exclude this category from the radar — JD doesn't mention it
            continue
        matched_in_cat = jd_cat_skills.intersection(set(matched_skills))
        pct = round(len(matched_in_cat) / len(jd_cat_skills) * 100)
        coverage[cat] = pct
    return coverage


# ─────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────

def calculate_composite_score(semantic_score, keyword_coverage, skill_match_pct, ats_score, num_jd_skills=0):
    """
    Weighted composite score combining all NLP signals.
    Weights are dynamically adjusted based on the job type:
    - Tech jobs (>3 skills): Semantic 35%, Skills 30%, Keywords 25%, ATS 10%
    - Non-tech (<=3 skills): Semantic 45%, Keywords 40%, Skills 5%, ATS 10%

    Guard: if no JD skills detected, skill_match_pct is treated as 100 (non-technical role)
    to avoid unfairly penalising candidates for a non-technical JD.
    """
    # Guard against zero-skill JDs collapsing the score
    effective_skill_pct = skill_match_pct if num_jd_skills > 0 else 100.0

    if num_jd_skills > 3:
        w_sem, w_kw, w_skill, w_ats = 0.35, 0.25, 0.30, 0.10
    else:
        # Non-technical role: rely more on semantic and keyword signals
        w_sem, w_kw, w_skill, w_ats = 0.45, 0.40, 0.05, 0.10

    composite = (
        float(semantic_score) * w_sem +
        float(keyword_coverage) * w_kw +
        float(effective_skill_pct) * w_skill +
        float(ats_score) * w_ats
    )
    return round(composite, 1)


# ─────────────────────────────────────────────
# SUGGESTIONS GENERATOR
# ─────────────────────────────────────────────

def generate_specific_suggestions(missing_skills, missing_keywords):
    """
    Generate contextual resume improvement suggestions based on skill category.
    """
    suggestions = []
    
    for skill in missing_skills[:4]:
        cat = get_skill_category(skill)
        skill_name = skill.title()
        
        if cat == "Programming Languages":
            suggestions.append(f"Consider adding **{skill_name}** to your projects or skills section if you have experience writing code in it.")
        elif cat == "Web & Frontend":
            suggestions.append(f"Mention how you built user interfaces or web components using **{skill_name}**.")
        elif cat == "Backend & Frameworks":
            suggestions.append(f"Detail any API development, server logic, or backend architecture you've built with **{skill_name}**.")
        elif cat == "Data & Databases":
            suggestions.append(f"Highlight your experience querying, modeling, or managing data using **{skill_name}**.")
        elif cat == "AI & Machine Learning":
            suggestions.append(f"Add any machine learning models, data pipelines, or AI projects you've developed involving **{skill_name}**.")
        elif cat == "Cloud & DevOps":
            suggestions.append(f"Mention any cloud deployments, CI/CD pipelines, or infrastructure you managed with **{skill_name}**.")
        elif cat == "Soft Skills":
            suggestions.append(f"Provide a concrete example of when you demonstrated **{skill_name}** in a past role or project.")
        else:
            suggestions.append(f"Explicitly mention your hands-on experience with **{skill_name}** in your work history.")

    # Only add top keywords that aren't already covered by missing skills
    kw_count = 0
    for kw in missing_keywords:
        if kw.lower() not in [s.lower() for s in missing_skills]:
            suggestions.append(f"The keyword **'{kw}'** appears frequently in the JD. Try weaving it into your professional summary or bullet points.")
            kw_count += 1
        if kw_count >= 2:
            break
            
    return suggestions


# ─────────────────────────────────────────────
# SECTION-WISE SCORING
# ─────────────────────────────────────────────

def extract_jd_requirements(text):
    """Extract required years of experience and education from JD."""
    reqs = {
        "experience_years": 0,
        "education": []
    }
    
    # Extract years of experience
    exp_match = re.search(r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', text, re.I)
    if exp_match:
        reqs["experience_years"] = int(exp_match.group(1))
        
    # Extract required education
    degree_patterns = [
        r'\b(B\.?Tech|B\.?E\.?|B\.?Sc\.?|Bachelor[s]?(?:\s+of\s+\w+)?)\b',
        r'\b(M\.?Tech|M\.?E\.?|M\.?Sc\.?|Master[s]?(?:\s+of\s+\w+)?)\b',
        r'\b(Ph\.?D|Doctorate)\b',
        r'\b(MBA|MCA|BCA|B\.?Com|M\.?Com)\b',
        r'\b(Associate[s]? Degree|Diploma)\b'
    ]
    for pat in degree_patterns:
        found = re.findall(pat, text, re.I)
        reqs["education"].extend(found)
    reqs["education"] = list(set(reqs["education"]))
    
    return reqs

def calculate_section_scores(resume_entities, jd_entities, skill_match_pct):
    """
    Calculate granular scores for Skills, Experience, and Education.
    Returns scores out of 100.
    """
    scores = {
        "skills": float(skill_match_pct),
        "experience": 100.0,
        "education": 100.0
    }
    
    # Experience Scoring
    jd_exp = jd_entities.get("experience_years", 0)
    res_exp = resume_entities.get("experience_years") or 0
    
    if jd_exp > 0:
        if res_exp >= jd_exp:
            scores["experience"] = 100.0
        else:
            scores["experience"] = round((res_exp / jd_exp) * 100, 1)
            
    # Education Scoring
    jd_edu = jd_entities.get("education", [])
    res_edu = resume_entities.get("education", [])
    
    if jd_edu:
        # If JD requires education, check if resume has any listed education
        # We use a simple heuristic: if they have any degree, it's a 100% (assuming it meets basic reqs)
        # If they lack any degree, it's 0%
        if res_edu:
            scores["education"] = 100.0
        else:
            scores["education"] = 0.0
            
    return scores


# ─────────────────────────────────────────────
# MATCH EXPLANATION
# ─────────────────────────────────────────────

def get_sentence_matches(resume_text, jd_text, top_n=5):
    """
    Find the top-N best matching sentence pairs between the resume and JD
    using Sentence-BERT cosine similarity.
    Returns a list of (resume_sentence, jd_sentence, similarity_score) tuples.
    """
    # Split into meaningful sentences
    resume_sents = [s.strip() for s in re.split(r'[.\n]', resume_text) if len(s.strip()) > 20]
    jd_sents = [s.strip() for s in re.split(r'[.\n]', jd_text) if len(s.strip()) > 20]

    if not resume_sents or not jd_sents:
        return []

    # Encode all sentences
    res_embeddings = model.encode(resume_sents)
    jd_embeddings = model.encode(jd_sents)

    # Compute pairwise similarity
    sim_matrix = cosine_similarity(res_embeddings, jd_embeddings)

    # Collect all pairs with their scores
    pairs = []
    for i in range(len(resume_sents)):
        for j in range(len(jd_sents)):
            pairs.append((resume_sents[i], jd_sents[j], float(sim_matrix[i][j])))

    # Sort by similarity descending and deduplicate resume sentences
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    seen_resume = set()
    seen_jd = set()
    top_matches = []
    for res_s, jd_s, score in pairs:
        if res_s not in seen_resume and jd_s not in seen_jd:
            top_matches.append((res_s, jd_s, round(score * 100, 1)))
            seen_resume.add(res_s)
            seen_jd.add(jd_s)
        if len(top_matches) >= top_n:
            break

    return top_matches


def generate_match_explanation(composite_score, semantic_score, kw_coverage, skill_match_pct, ats_score, 
                                matched_skills, missing_skills, matched_kws, missing_kws, num_jd_skills):
    """
    Generate a human-readable explanation of why the match score is high or low.
    Breaks down each scoring dimension with clear reasoning.
    """
    explanations = []
    strengths = []
    weaknesses = []

    # Overall verdict
    if composite_score >= 80:
        explanations.append("🟢 **Excellent Match** — Your resume is strongly aligned with this job description across all key dimensions.")
    elif composite_score >= 60:
        explanations.append("🟡 **Good Match** — Your resume covers most requirements, but there are specific gaps worth addressing.")
    elif composite_score >= 40:
        explanations.append("🟠 **Moderate Match** — There are significant gaps between your resume and this JD. Targeted improvements can help.")
    else:
        explanations.append("🔴 **Low Match** — Your resume needs substantial revision to align with this role. Focus on the weak areas below.")

    # Semantic similarity breakdown
    if semantic_score >= 70:
        strengths.append(f"**Semantic Similarity ({semantic_score}%):** Your resume's overall language and context closely mirrors the JD. The writing style and domain terminology are well-aligned.")
    elif semantic_score >= 45:
        weaknesses.append(f"**Semantic Similarity ({semantic_score}%):** Partial overlap in language. Consider rephrasing your bullet points to use similar terminology and framing as the JD.")
    else:
        weaknesses.append(f"**Semantic Similarity ({semantic_score}%):** The language in your resume is quite different from the JD. This suggests a domain or role mismatch — try tailoring your summary and descriptions.")

    # Keyword coverage breakdown
    if kw_coverage >= 70:
        strengths.append(f"**Keyword Coverage ({kw_coverage}%):** Most of the JD's important keywords appear in your resume. ATS systems will likely parse this well.")
    elif kw_coverage >= 40:
        weaknesses.append(f"**Keyword Coverage ({kw_coverage}%):** About half of the JD's critical keywords are missing. Add these to your skills or experience sections: {', '.join(missing_kws[:5])}.")
    else:
        weaknesses.append(f"**Keyword Coverage ({kw_coverage}%):** Very few JD keywords appear in your resume. This will hurt ATS ranking. Priority keywords to add: {', '.join(missing_kws[:5])}.")

    # Skill match breakdown
    if num_jd_skills > 0:
        if skill_match_pct >= 75:
            strengths.append(f"**Skill Match ({skill_match_pct}%):** You cover {len(matched_skills)}/{len(matched_skills)+len(missing_skills)} required technical skills. Strong alignment.")
        elif skill_match_pct >= 40:
            weaknesses.append(f"**Skill Match ({skill_match_pct}%):** You're missing {len(missing_skills)} required skills: {', '.join([s.title() for s in missing_skills[:5]])}.")
        else:
            weaknesses.append(f"**Skill Match ({skill_match_pct}%):** Major skill gaps detected. You're missing {len(missing_skills)} of {len(matched_skills)+len(missing_skills)} required skills: {', '.join([s.title() for s in missing_skills[:5]])}.")
    else:
        strengths.append("**Skill Match:** No specific technical skills detected in this JD — this appears to be a non-technical or generalist role.")

    # ATS score breakdown
    if ats_score >= 75:
        strengths.append(f"**ATS Friendliness ({ats_score}/100):** Your resume is well-structured for ATS parsers with proper sections, contact info, and action verbs.")
    elif ats_score >= 50:
        weaknesses.append(f"**ATS Friendliness ({ats_score}/100):** Some formatting or structural issues may trip up ATS parsers. Check the ATS tab for specific fixes.")
    else:
        weaknesses.append(f"**ATS Friendliness ({ats_score}/100):** Significant ATS compatibility issues. Missing sections, contact info, or action verbs will hurt your ranking.")

    return explanations, strengths, weaknesses


# ─────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────

def generate_pdf_report(composite_score, semantic_score, kw_coverage, skill_match_pct, ats_score,
                         section_scores, matched_skills, missing_skills, matched_kws, missing_kws,
                         explanations, strengths, weaknesses, suggestions, sentence_matches):
    """
    Generate a professional PDF analysis report using fpdf2.
    Returns the PDF as bytes.
    """
    from fpdf import FPDF
    from datetime import datetime

    class ReportPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(99, 102, 241)  # Indigo
            self.cell(0, 12, "ResumeIQ Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", new_x="LMARGIN", new_y="NEXT", align="C")
            self.ln(4)
            # Divider line
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(6)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"ResumeIQ - Page {self.page_no()}/{{nb}}", align="C")

        def section_title(self, title):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(55, 55, 80)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(99, 102, 241)
            self.line(10, self.get_y(), 80, self.get_y())
            self.ln(4)

        def sub_title(self, title):
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(80, 80, 100)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def body_text(self, text):
            self.set_font("Helvetica", "", 10)
            self.set_text_color(50, 50, 50)
            # Clean text for Latin-1 compatibility
            safe_text = text.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 6, safe_text)
            self.ln(2)

        def bullet(self, text):
            self.set_font("Helvetica", "", 10)
            self.set_text_color(50, 50, 50)
            safe_text = text.encode('latin-1', 'replace').decode('latin-1')
            self.set_x(self.l_margin)
            self.multi_cell(0, 6, f"  {chr(149)}  {safe_text}")
            self.ln(1)

        def score_row(self, label, value, color=(50, 50, 50)):
            self.set_font("Helvetica", "", 10)
            self.set_text_color(80, 80, 100)
            self.cell(70, 7, label)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*color)
            self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Composite Score ──
    pdf.section_title("Overall Match Score")
    verdict = "Excellent" if composite_score >= 75 else "Good" if composite_score >= 50 else "Low"
    score_color = (34, 197, 94) if composite_score >= 75 else (234, 179, 8) if composite_score >= 50 else (239, 68, 68)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(*score_color)
    pdf.cell(0, 20, f"{composite_score}%  ({verdict} Match)", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # ── Individual Metrics ──
    pdf.section_title("Score Breakdown")
    pdf.score_row("Semantic Similarity (SBERT):", f"{semantic_score}%", (139, 92, 246))
    pdf.score_row("Keyword Coverage (TF-IDF):", f"{kw_coverage}%", (52, 211, 153))
    pdf.score_row("Skill Match:", f"{skill_match_pct}%", (245, 158, 11))
    pdf.score_row("ATS Friendliness:", f"{ats_score}/100", (236, 72, 153))
    pdf.ln(2)
    pdf.score_row("Skills Section Score:", f"{section_scores['skills']}%")
    pdf.score_row("Experience Section Score:", f"{section_scores['experience']}%")
    pdf.score_row("Education Section Score:", f"{section_scores['education']}%")
    pdf.ln(4)

    # ── Match Explanation ──
    pdf.section_title("Match Explanation")
    for exp in explanations:
        clean_exp = re.sub(r'[*]', '', exp)  # Strip markdown bold
        clean_exp = re.sub(r'[^\x00-\xff]', '', clean_exp)  # Strip emojis
        pdf.body_text(clean_exp)

    if strengths:
        pdf.sub_title("Strengths")
        for s in strengths:
            clean_s = re.sub(r'[*]', '', s)
            clean_s = re.sub(r'[^\x00-\xff]', '', clean_s)
            pdf.bullet(clean_s)

    if weaknesses:
        pdf.sub_title("Weaknesses")
        for w in weaknesses:
            clean_w = re.sub(r'[*]', '', w)
            clean_w = re.sub(r'[^\x00-\xff]', '', clean_w)
            pdf.bullet(clean_w)

    pdf.ln(4)

    # ── Skills Analysis ──
    pdf.section_title("Skills Analysis")

    if matched_skills:
        pdf.sub_title(f"Matched Skills ({len(matched_skills)})")
        skills_text = ", ".join([s.title() for s in sorted(matched_skills)])
        pdf.body_text(skills_text)

    if missing_skills:
        pdf.sub_title(f"Missing Skills ({len(missing_skills)})")
        skills_text = ", ".join([s.title() for s in sorted(missing_skills)])
        pdf.body_text(skills_text)

    pdf.ln(2)

    # ── Keywords ──
    pdf.section_title("Keyword Analysis")

    if matched_kws:
        pdf.sub_title(f"Matched Keywords ({len(matched_kws)})")
        pdf.body_text(", ".join(matched_kws[:15]))

    if missing_kws:
        pdf.sub_title(f"Missing Keywords ({len(missing_kws)})")
        pdf.body_text(", ".join(missing_kws[:15]))

    pdf.ln(2)

    # ── Top Matching Sentences ──
    if sentence_matches:
        pdf.add_page()
        pdf.section_title("Top Matching Lines (Resume vs JD)")
        for idx, (res_s, jd_s, sim_score) in enumerate(sentence_matches, 1):
            pdf.sub_title(f"Match #{idx} - {sim_score}% similarity")
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(139, 92, 246)
            safe_res = res_s.encode('latin-1', 'replace').decode('latin-1')
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"Resume: {safe_res}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(96, 165, 250)
            safe_jd = jd_s.encode('latin-1', 'replace').decode('latin-1')
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"JD: {safe_jd}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

    # ── Suggestions ──
    pdf.add_page()
    pdf.section_title("Improvement Suggestions")
    if suggestions:
        for s in suggestions:
            clean_s = re.sub(r'[*]', '', s)
            clean_s = re.sub(r'[^\x00-\xff]', '', clean_s)
            pdf.bullet(clean_s)
    else:
        pdf.body_text("No major gaps detected. Your resume is well-aligned with this JD.")

    pdf.bullet("Quantify Achievements: Add metrics to your bullet points - percentages, numbers, team sizes.")
    pdf.bullet("Tailor Your Summary: Update your professional summary to mirror the language used in the JD.")

    return bytes(pdf.output())


# ─────────────────────────────────────────────
# AI RESUME REWRITER (OpenRouter)
# ─────────────────────────────────────────────

def rewrite_resume_with_openrouter(resume_text, jd_text):
    """
    Use OpenRouter API to rewrite the resume summary and bullet points.
    Tries multiple free models with automatic fallback if one is rate-limited.
    """
    import time

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        return "⚠️ Please set your valid OPENROUTER_API_KEY in the .env file."

    # List of free models to try in order of preference
    free_models = [
        "qwen/qwen3-coder:free",
        "google/gemma-4-31b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""
    You are an expert Executive Resume Writer and ATS Optimization Specialist. 
    I am providing you with a candidate's current resume and a target job description (JD).
    
    Your task:
    Rewrite the candidate's Professional Summary and Work Experience bullet points so they better align with the JD's keywords, tone, and requirements.
    
    CRITICAL RULES:
    1. DO NOT hallucinate or invent experience the candidate does not have.
    2. Keep the output highly professional, action-oriented, and quantified.
    3. Format the output in clean Markdown.
    
    --- TARGET JOB DESCRIPTION ---
    {jd_text}
    
    --- CANDIDATE RESUME ---
    {resume_text}
    
    Please provide the improved Professional Summary and rewritten Bullet Points below:
    """

    errors = []
    for model_id in free_models:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return completion.choices[0].message.content

        except Exception as e:
            errors.append(f"{model_id}: {str(e)}")
            time.sleep(2)  # Brief pause before trying next model
            continue

    return "❌ All free models are currently rate-limited. Please wait a minute and try again.\n\nDetails:\n" + "\n".join(errors)