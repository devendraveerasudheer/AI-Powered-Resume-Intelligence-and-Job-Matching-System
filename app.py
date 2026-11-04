
import streamlit as st
import tempfile
import os
import plotly.express as px

import backend


# Page configuration
st.set_page_config(
    page_title="AI-Powered Resume Intelligence and Job Matching System",
    page_icon="📄",
    layout="wide"
)


# Application title
st.title("📄 AI-Powered Resume Intelligence and Job Matching System")

st.write(
    "Analyze your resume against a job description "
    "using Machine Learning, NLP, and Generative AI."
)

st.divider()


# Resume upload
st.subheader("1. Upload Your Resume")

uploaded_resume = st.file_uploader(
    "Upload PDF or DOCX",
    type=["pdf", "docx"]
)


# Job description input
st.subheader("2. Enter Job Description")

jd_text = st.text_area(
    "Paste the complete job description below",
    height=250,
    placeholder="Paste job description here..."
)


# Analyze button
if st.button("Analyze Resume", type="primary"):

    if uploaded_resume is None:
        st.error("Please upload your resume.")

    elif not jd_text.strip():
        st.error("Please enter a job description.")

    else:

        file_extension = os.path.splitext(
            uploaded_resume.name
        )[1].lower()

        temp_path = None

        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(uploaded_resume.getvalue())
                temp_path = temp_file.name

            # Validate inputs
            backend.validate_resume_inputs(
                temp_path,
                jd_text
            )

            # Run analysis
            with st.spinner(
                "Analyzing resume... Please wait."
            ):

                results = backend.analyze_resume_pipeline(
                    temp_path,
                    jd_text
                )

            # Store results
            st.session_state["analysis_results"] = results

            st.success("Resume analysis completed!")

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

        finally:
            # Remove temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# Display results
if "analysis_results" in st.session_state:

    results = st.session_state["analysis_results"]

    st.divider()

    st.header("3. Resume Analysis Results")

    # Overall score
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Overall Match Score",
            f"{results['overall_match_score']}%"
        )

    with col2:
        st.metric(
            "Match Category",
            results["match_category"]
        )

    # Individual scores
    st.subheader("Similarity Analysis")

    score_data = {
        "Metric": [
            "Skill Match",
            "Semantic Similarity",
            "TF-IDF Similarity"
        ],
        "Score": [
            results["skill_match"],
            results["semantic_similarity"],
            results["tfidf_similarity"]
        ]
    }

    fig = px.bar(
        score_data,
        x="Metric",
        y="Score",
        range_y=[0, 100],
        text="Score",
        title="Resume Similarity Scores"
    )

    fig.update_traces(texttemplate="%{text:.2f}%")
    st.plotly_chart(fig, use_container_width=True)

    # Skills analysis
    st.subheader("Skill Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Matched Skills**")

        for skill in results["matched_skills"]:
            st.write(f"✅ {skill}")

        if not results["matched_skills"]:
            st.write("No matching skills found.")

    with col2:
        st.markdown("**Missing Skills**")

        for skill in results["missing_skills"]:
            st.write(f"❌ {skill}")

        if not results["missing_skills"]:
            st.write("No missing skills found.")

    st.subheader("Additional Skills")

    st.write(
        ", ".join(results["additional_skills"])
        if results["additional_skills"]
        else "No additional skills identified."
    )

    # Gemini feedback
    st.divider()

    st.header("4. AI-Generated Resume Feedback")

    st.markdown(results["llm_analysis"])

    # Word counts
    st.divider()

    st.subheader("Document Statistics")

    col1, col2 = st.columns(2)

    col1.metric(
        "Resume Word Count",
        results["resume_word_count"]
    )

    col2.metric(
        "Job Description Word Count",
        results["jd_word_count"]
    )


st.divider()

st.caption(
    "AI Resume Analyzer | Built using Python, NLP, "
    "Machine Learning, Gemini API, and Streamlit."
)

