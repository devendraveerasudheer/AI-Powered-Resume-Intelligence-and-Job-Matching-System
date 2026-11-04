# AI-Powered Resume Intelligence and Job Matching System

An intelligent resume analysis application that evaluates resumes against job descriptions using Natural Language Processing (NLP), Machine Learning, and semantic similarity techniques. It helps candidates identify skill gaps, understand their job compatibility, and improve their resumes.

## 🚀 Live Application

**Try the application here:**

[AI-Powered Resume Intelligence and Job Matching System](https://ai-powered-resume-intelligence-and-job-matching-system-syr4x8w.streamlit.app/)

## 📌 Project Overview

The AI-Powered Resume Intelligence and Job Matching System analyzes uploaded resumes and compares them with job descriptions to evaluate candidate suitability.

The application extracts relevant information from PDF and DOCX resumes, identifies technical skills, calculates similarity scores, and provides an overall resume match score.

It combines traditional NLP techniques with semantic similarity to provide a more comprehensive resume evaluation.

## ✨ Key Features

- **Resume Upload:** Supports PDF and DOCX resume formats.
- **Text Extraction:** Extracts textual information from uploaded resumes.
- **Text Preprocessing:** Performs text normalization, tokenization, stopword removal, and lemmatization.
- **Skill Extraction:** Identifies technical skills from resumes and job descriptions.
- **Skill Matching:** Finds matched, missing, and additional skills.
- **Semantic Similarity:** Uses Sentence Transformers to understand contextual similarity between resumes and job descriptions.
- **TF-IDF Similarity:** Calculates textual similarity using TF-IDF and cosine similarity.
- **Resume Match Score:** Combines multiple evaluation techniques into an overall compatibility score.
- **AI-Powered Feedback:** Supports Gemini-based resume analysis when API access is configured.
- **Interactive Dashboard:** Displays analysis results through a Streamlit interface.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| NLP | NLTK, Regex |
| Machine Learning | Scikit-learn |
| Semantic Analysis | Sentence Transformers |
| Text Similarity | TF-IDF, Cosine Similarity |
| Deep Learning Framework | PyTorch |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Resume Parsing | PyMuPDF, python-docx |
| Generative AI | Google Gemini API |

## ⚙️ Project Workflow

1. Upload a resume in PDF or DOCX format.
2. Enter the target job description.
3. Extract text from the uploaded resume.
4. Preprocess the resume and job description.
5. Extract technical skills from both documents.
6. Identify matched, missing, and additional skills.
7. Calculate TF-IDF similarity.
8. Generate semantic embeddings using Sentence Transformers.
9. Calculate semantic similarity.
10. Combine evaluation scores to calculate the overall resume match score.
11. Generate additional AI-based feedback when Gemini API access is configured.
12. Display the analysis results in the Streamlit dashboard.

## 📊 Resume Scoring Methodology

The overall resume match score combines three evaluation components:

| Evaluation Component | Weight |
|---|---:|
| Skill Matching | 40% |
| Semantic Similarity | 35% |
| TF-IDF Similarity | 25% |
| **Total** | **100%** |

The weighted scoring approach evaluates both skill compatibility and textual/contextual similarity between the resume and job description.

## 📂 Project Structure

```text
AI-Resume-Intelligence/
│
├── app.py
├── backend.py
├── requirements.txt
├── main.ipynb
├── README.md
├── .gitignore
│
└── data/
    └── Sample files (optional)
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

If you want to enable Gemini-powered feedback locally, create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep your `.env` file private. Never upload API keys to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔐 Security

- API keys are managed through environment variables.
- Sensitive configuration files are excluded using `.gitignore`.
- Personal resume documents should not be committed to a public repository.

## 🎯 Project Objective

The primary objective of this project is to simplify resume evaluation by combining NLP, Machine Learning, and semantic analysis techniques. It helps job seekers understand how well their resumes align with job requirements and identify areas for improvement.

## 👨‍💻 Author

**AI-Powered Resume Intelligence and Job Matching System**

Developed as a practical AI/ML project using Python, NLP, Machine Learning, and Generative AI.

---

⭐ If you find this project useful, consider giving the repository a star!
