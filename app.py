import streamlit as st
import requests
import hashlib
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

# =========================
# LOAD ENV VARIABLES (.env)
# =========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'cache' not in st.session_state:
    st.session_state.cache = {}


# =========================
# HELPER FUNCTIONS
# =========================

def generate_claim_hash(claim: str) -> str:
    return hashlib.sha256(claim.lower().strip().encode()).hexdigest()


def search_google(query: str) -> list:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'q': query,
        'dateRestrict': 'm3',
        'num': 5
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get('items', []):
            results.append({
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', '')
            })

        return results

    except Exception as e:
        st.error(f"Google Search API error: {str(e)}")
        return []


def analyze_with_openai(claim: str, search_results: list) -> dict:

    context = "\n\n".join([
        f"Source {i+1}: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
        for i, r in enumerate(search_results)
    ])

    prompt = f"""
You are a fact-checking AI. Analyze the following claim.

CLAIM: {claim}

SEARCH RESULTS:
{context}

Provide response in EXACT format:

VERDICT: [REAL/FAKE/UNVERIFIED]
CONFIDENCE: [0-100]
EXPLANATION: [Exactly 3 sentences explaining your verdict]
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional fact-checking assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        verdict = "UNVERIFIED"
        confidence = 50.0
        explanation = "Unable to determine verdict."

        for line in content.split("\n"):
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
            elif line.startswith("CONFIDENCE:"):
                confidence = float(line.split(":", 1)[1].strip())
            elif line.startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()

        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation
        }

    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0,
            "explanation": "Error during analysis."
        }


def verify_claim(claim: str):

    if not claim.strip():
        st.warning("⚠️ Please enter a claim")
        return None

    claim_hash = generate_claim_hash(claim)

    if claim_hash in st.session_state.cache:
        result = st.session_state.cache[claim_hash]
        result["cached"] = True
        return result

    with st.spinner("🔍 Searching..."):
        search_results = search_google(claim)

    if not search_results:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0,
            "explanation": "No sources found.",
            "sources": [],
            "cached": False
        }

    with st.spinner("🤖 Analyzing..."):
        analysis = analyze_with_openai(claim, search_results)

    result = {
        "verdict": analysis["verdict"],
        "confidence": analysis["confidence"],
        "explanation": analysis["explanation"],
        "sources": search_results,
        "cached": False,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    st.session_state.cache[claim_hash] = result
    return result


# =========================
# UI
# =========================

st.title("🔍VeriSearch 2026:")
st.subheader(" A Real-Time Fake News Verification System using RAG and LLMs")

claim = st.text_area(
    label="Enter claim",
    placeholder="Paste claim here...",
    height=170
)

if st.button("Verify"):
    result = verify_claim(claim)
    if result:
        st.session_state.result = result


if st.session_state.result:

    result = st.session_state.result

    st.markdown("---")
    st.markdown(f"### Verdict: {result['verdict']}")
    st.markdown(f"**Confidence:** {int(result['confidence'])}%")
    st.markdown(f"**Explanation:** {result['explanation']}")

    st.markdown("### Sources")
    for s in result["sources"]:
        st.markdown(f"- [{s['title']}]({s['url']})")