# 💼 Financial Document Q&A Assistant

A local Streamlit application that processes **financial documents (PDF & Excel)** and provides an **interactive Q&A chatbot** to query financial data using natural language.  
Built with **Streamlit**, **Ollama SLMs**, and **pandas/pdfplumber** for data extraction.  

---

## Features

- Upload **PDF** and **Excel** financial documents  
- Extract **text** and **tables** (Income Statements, Balance Sheets, Cash Flow Statements)  
- Interactive **chat interface** for natural language questions  
- **Structured metric extraction** (Revenue, Net Income, Expenses, Assets, Liabilities, Equity)  
  - Detects specific years in the question (e.g., *“Revenue in 2024”*)  
  - If no year specified → automatically **sums across all years**  
- **Vector-based retrieval** + **local LLM (Mistral)** for context-aware answers  
- Works **offline / locally** using Ollama (no cloud dependency)  
- Transparent results with **table previews** and **retrieved context**  

---

## Tech Stack

- [Streamlit](https://streamlit.io/) -> UI & chat interface  
- [pdfplumber](https://github.com/jsvine/pdfplumber) -> PDF text/table extraction  
- [pandas](https://pandas.pydata.org/) -> Excel parsing & table handling  
- [Ollama](https://ollama.ai/) -> Local SLMs for embeddings + Q&A  
  - `mistral:7b-instruct` (chat model)  
  - `nomic-embed-text` (embedding model)  

---

## Installation

### 1. Install [Ollama](https://ollama.ai/)
Ensure Ollama is installed and running locally (`http://localhost:11434`).

### 2. Pull Required Models
```bash
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text
```

### 3. Clone the Repo
```bash
git clone https://github.com/<your-username>/financial-qa-bot.git
cd financial-qa-bot
```

### 4. Setup Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Run the App
```bash
streamlit run app.py
```
The app will open in your browser (default: [http://localhost:8501](http://localhost:8501)).

---

## Project Structure

```
financial-qa-bot/
│── app.py                # Main Streamlit app
│── requirements.txt      # Python dependencies
│── README.md             # Documentation
└── sample_data/
    └── sample_income_statement.xlsx
```

---

## Usage

1. **Upload PDFs/Excels** in the sidebar  
2. Click **Build Index** to process the files  
3. Ask questions like:
   - *“What is the total revenue?”* → sums across years  
   - *“What is the revenue in 2024?”* → picks the column for 2024  
   - *“What is the net income?”*  
   - *“What are total assets?”*  
4. Expand the **structured metric** and **retrieved chunks** panels to see the exact sources  

---

## Example

Sample Excel (`sample_income_statement.xlsx`):

| Metric             | 2023       | 2024       |
|--------------------|------------|------------|
| Total Revenue      | $1,200,000 | $1,450,000 |
| Operating Expenses | $200,000   | $240,000   |
| Net Income         | $180,000   | $220,000   |

**Query:** *“What is the total revenue?”*  
**Answer:** $2,650,000 (sum of 2023 + 2024)  

**Query:** *“What is the revenue in 2024?”*  
**Answer:** $1,450,000  
