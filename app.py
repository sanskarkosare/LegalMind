import streamlit as st
import tempfile
import os

# --- SPACY SETUP ---
@st.cache_resource
def download_spacy_model():
    os.system("python -m spacy download en_core_web_sm")

download_spacy_model()

# --- IMPORT YOUR BACKEND LOGIC ---
from backend.ml.clause_classifier import analyze_contract
from backend.ml.risk_scorer import extract_text_from_pdf, summarize_contract, answer_legal_question

st.set_page_config(page_title="LegalMind", page_icon="⚖️", layout="centered")

# Custom CSS to mimic the React Tailwind styling
st.markdown("""
<style>
    .risk-HIGH { border-left: 5px solid #ef4444; background-color: #450a0a; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .risk-MEDIUM { border-left: 5px solid #eab308; background-color: #422006; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .risk-LOW { border-left: 5px solid #22c55e; background-color: #052e16; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .warning-box { background-color: #422006; border: 1px solid #713f12; color: #facc15; padding: 10px; border-radius: 8px; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #818cf8;'>⚖️ LegalMind</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af;'>AI-powered contract analysis and legal assistant</p>", unsafe_allow_html=True)
st.divider()

# Initialize session state for results and chat
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I'm LegalMind AI. Ask me anything about contracts, legal clauses, or legal documents."}
    ]

tab1, tab2 = st.tabs(["📄 Contract Analyzer", "💬 Legal AI Chat"])

# ================= TAB 1: ANALYZER =================
with tab1:
    st.subheader("Upload Contract")
    st.caption("Upload any contract PDF — AI will detect risky clauses and score the overall risk")
    
    uploaded_file = st.file_uploader("", type=['pdf'], label_visibility="collapsed")
    
    if st.button("⚖️ Analyze Contract", use_container_width=True, type="primary"):
        if not uploaded_file:
            st.error("Please upload a contract PDF.")
        else:
            with st.spinner("Analyzing contract... (this may take 30-60 seconds)"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    text = extract_text_from_pdf(tmp_path)
                    
                    # Clean the text to remove invisible whitespace
                    clean_text = text.strip() if text else ""
                    
                    # Check if we actually extracted readable words
                    if len(clean_text) < 50:
                        st.error(f"⚠️ PDF Extraction Failed: Only found {len(clean_text)} readable characters. If this is a scanned document or image, the AI cannot read it.")
                        st.text_area("What the AI saw:", clean_text, height=100)
                    else:
                        # DEBUG VISUALIZER
                        with st.expander("👀 Debug: View Raw Extracted Text (Check if this is gibberish)"):
                            st.write(clean_text[:1500] + "...\n\n(Text truncated for preview)")
                        
                        # Proceed with analysis
                        clause_analysis = analyze_contract(clean_text)
                        summary = summarize_contract(clean_text)
                        
                        st.session_state.analysis_result = {
                            "clause_analysis": clause_analysis,
                            "summary": summary
                        }
                except Exception as e:
                    error_msg = str(e)
                    if "shape=(0, 5000)" in error_msg:
                        st.error("❌ Machine Learning Error: The backend extracted the text, but found 0 valid clauses to analyze.")
                        st.warning("Please check the 'Debug: View Raw Extracted Text' dropdown above. If the text looks like random symbols or lacks punctuation, your backend's sentence splitter is failing to process it.")
                    else:
                        st.error(f"Analysis failed: {error_msg}")
                finally:
                    os.remove(tmp_path)

    # Display Results
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.divider()
        
        # Risk Score Board
        score_data = res.get('clause_analysis', {})
        level = score_data.get('risk_score', {}).get('level', 'UNKNOWN')
        score = score_data.get('risk_score', {}).get('score', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Contract Risk", level)
        with col2:
            st.metric("Risk Score", f"{score}/100")
            
        st.write(f"🔴 High: {score_data.get('high_risk_count', 0)} | 🟡 Medium: {score_data.get('medium_risk_count', 0)} | 🟢 Low: {score_data.get('low_risk_count', 0)}")
        
        # Summary Section
        if 'summary' in res:
            st.markdown("### 📋 Contract Summary")
            sum_data = res['summary']
            st.markdown(f"**Overview:** {sum_data.get('summary', '')}")
            st.markdown(f"**Parties Involved:** {sum_data.get('parties', '')}")
            
            if sum_data.get('key_terms'):
                st.markdown("**Key Terms:**")
                for term in sum_data['key_terms']:
                    st.markdown(f"- {term}")
                    
            if sum_data.get('red_flags'):
                st.markdown("**Red Flags:**")
                for flag in sum_data['red_flags']:
                    st.markdown(f"- ⚠️ {flag}")
                    
            if sum_data.get('recommendations'):
                st.markdown("**Recommendations:**")
                for rec in sum_data['recommendations']:
                    st.markdown(f"- ✅ {rec}")

        # Detected Clauses
        if score_data.get('detected_clauses'):
            st.markdown(f"### 🔍 Detected Clauses ({score_data.get('total_clauses_found', 0)})")
            for clause in score_data['detected_clauses']:
                risk = clause.get('risk_level', 'LOW')
                st.markdown(f"""
                <div class="risk-{risk}">
                    <strong>{clause.get('clause', 'Unknown')}</strong> ({clause.get('confidence', 0)}% confidence) - <em>{risk} RISK</em><br>
                    <span style="font-size: 14px; opacity: 0.8;">{clause.get('description', '')}</span>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown('<div class="warning-box">⚠️ AI analysis only. Always consult a qualified lawyer before signing any contract.</div>', unsafe_allow_html=True)

# ================= TAB 2: CHAT =================
with tab2:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else None):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about contracts, legal clauses, NDA, IP rights..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        context = ""
        if st.session_state.analysis_result and 'summary' in st.session_state.analysis_result:
            context = st.session_state.analysis_result['summary'].get('summary', '')

        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Thinking..."):
                response_text = answer_legal_question(prompt, context)
            st.write(response_text)
            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
            
    st.markdown('<br><div class="warning-box">⚠️ AI legal information only. Not a substitute for professional legal advice.</div>', unsafe_allow_html=True)
