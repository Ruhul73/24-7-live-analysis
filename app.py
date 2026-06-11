import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

# 1. Automated Dependency Setup
@st.cache_resource
def initialize_nlp():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = initialize_nlp()

# 2. Automated Application Layer (NLP Engine)
def extract_instant_insights(raw_text):
    if not raw_text.strip():
        return None
        
    # Split text into individual reviews/lines
    reviews = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 10]
    if not reviews:
        return "Insufficient Data"

    pos_count, neg_count = 0, 0
    all_words = []
    stop_words = set(['the', 'and', 'a', 'to', 'of', 'is', 'in', 'it', 'that', 'was', 'for', 'on', 'with', 'as', 'this', 'but', 'my', 'you', 'with', 'i', 'we'])
    
    # Process reviews automatically
    for review in reviews:
        score = sia.polarity_scores(review)['compound']
        if score >= 0.05:
            pos_count += 1
        elif score <= -0.05:
            neg_count += 1
            
        words = re.findall(r'\b\w+\b', review.lower())
        all_words.extend([w for w in words if w not in stop_words and len(w) > 3])

    # Compile Metrics
    total = len(reviews)
    positive_pct = int((pos_count / total) * 100) if total > 0 else 0
    negative_pct = int((neg_count / total) * 100) if total > 0 else 0
    common_keywords = Counter(all_words).most_common(3)
    
    return {
        "total": total,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "keywords": common_keywords,
        "raw_count": len(reviews)
    }

# 3. UI and Output Layer
st.set_page_config(page_title="24/7 Solo-Founder Notepad", layout="wide", initial_sidebar_state="collapsed")

st.title("📝 24/7 Live Analysis Notepad")
st.caption("Designed exclusively for Solo-Entrepreneurs • 100% Automated Customer Insights")
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Paste Raw Reviews or Feedback")
    placeholder_text = (
        "Great tool, saved me 3 hours of manual work yesterday!\n"
        "The customer service response time was a bit slow, took 2 days.\n"
        "Love the new automated feature. Extremely easy to use.\n"
        "The pricing setup is clean, but the app crashed once on my phone."
    )
    user_input = st.text_area(
        "The notepad continuously parses your text in real-time:",
        value=placeholder_text,
        height=400,
        help="Paste Google reviews, emails, or raw customer feedback here."
    )

with col_right:
    st.subheader("⚡ Live Automated Insights")
    
    insights = extract_instant_insights(user_input)
    
    if insights == "Insufficient Data" or insights is None:
        st.info("Start typing or paste customer reviews in the notepad to trigger live automation.")
    else:
        # Metrics Cards
        m_col1, m_col2 = st.columns(2)
        
        # Smart Sentiment Indicator
        pulse_color = "🟢" if insights["positive_pct"] >= 70 else "🟡" if insights["positive_pct"] >= 40 else "🔴"
        m_col1.metric("Current Brand Pulse", f"{pulse_color} {insights['positive_pct']}% Positive")
        m_col2.metric("Total Line Items Parsed", insights["total"])
        
        st.markdown("---")
        
        # Automated Actionable Checklist Generation
        st.write("### 💡 Automated Solo-Founder Action Items")
        
        # Dynamic keywords extraction for copy ideas
        if insights["keywords"]:
            top_kw = insights["keywords"][0][0].upper()
            st.success(f"🎯 **Marketing Asset:** Use the keyword **'{top_kw}'** in your next landing page update. It's trending highly in your positive feedback.")
        
        if insights["negative_pct"] > 15:
            st.warning(f"⚠️ **Operational Risk:** Negative sentiment is at {insights['negative_pct']}%. Review recent negative lines in the notepad to pinpoint infrastructure or support bottlenecks.")
        else:
            st.info("✅ **Retention Stability:** Negative sentiment is minimal. Keep pushing current product features.")
            
        # Top Topics Table Output
        st.write("### 🔑 Top Trending Topics")
        if insights["keywords"]:
            kw_data = pd.DataFrame(insights["keywords"], columns=["Topic / Keyword", "Mentions"])
            st.dataframe(kw_data, use_container_width=True, hide_index=True)
        else:
            st.write("Processing topics...")

        # One-Click Automated Export Pipeline
        st.markdown("---")
        export_data = pd.DataFrame({
            "Metric": ["Total Analyzed", "Positive %", "Negative %"],
            "Value": [insights["total"], f"{insights['positive_pct']}%", f"{insights['negative_pct']}%"]
        })
        st.download_button(
            label="📥 Download Automated Insight Report (CSV)",
            data=export_data.to_csv(index=False).encode('utf-8'),
            file_name="solo_entrepreneur_insights.csv",
            mime="text/csv",
            use_container_width=True
        )
