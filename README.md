# CRM Experiment Lab

A comprehensive web application for running **Frequentist** and **Bayesian** analyses on CRM A/B tests. Perfect for CRM experimentation — subject lines, creative tests, segments, and campaign holdouts. Uses inputs (sends, opens, clicks, orders, revenue) and delivers clear verdicts, revenue scenarios, and copy-paste-ready summaries for stakeholders.

## Features

### Core Functionality

- **📋 Shared inputs** — Enter experiment context and campaign data once. Control vs Variant: total sends, unique opens, unique clicks, orders, and revenue. Both tabs use the same data.
- **📊 Frequentist tab** — Open rate, CTR, conversion rate, and AOV with two-proportion z-tests and Welch’s t-test. Clear verdict: Variant wins, Control wins, or Not enough evidence. Confidence intervals and a table you can copy into Google Sheets.
- **📈 Bayesian tab** — Same rate metrics plus **Revenue Per User (RPU)**. Monte Carlo simulation (10,000 runs) with Beta priors for rates; optional historical priors to anchor small tests. Probability Variant is better, expected lift, credible intervals, risk, and worst / most likely / best case revenue impact. Step-by-step breakdown of how those numbers are calculated.
- **🎯 In-app guidance** — Expander explains when to use Frequentist vs Bayesian (e.g. Bayesian to monitor mid-test, Frequentist to conclude and report).

### Analysis & Reporting

- **📉 Interactive charts** — Plotly distribution curves (Beta for rates, Normal for RPU), verdict cards, and summary tables with plain-language labels for CRM teams.
- **💰 Revenue impact** — Bayesian tab shows revenue impact if rolled out to full audience (worst case, most likely, best case) and a worked example so you can double-check the maths.
- **📝 AI-generated summaries** — Optional “Summarise Findings” in both tabs. Generates copy-paste-ready text for Slack or email based on experiment context and results (via OpenRouter API). Summary is clearly labelled as AI-generated.

### User Experience

- **✅ Input validation** — Real-time checks (e.g. opens/clicks/orders cannot exceed sends).
- **📖 Which tab to use** — Quick reference table and TL;DR so CRM managers know which tab to use when.
- **🔒 Safe secrets** — API key via environment variable or `.env`; never committed to the repo.

## Installation

Clone this repository:

```bash
git clone <repository-url>
cd crm_experimentation
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Local Development

Start the Streamlit app:

```bash
streamlit run app.py
```

Open your browser and navigate to the URL shown (typically http://localhost:8501).

**Workflow:**

1. **Experiment context** — Describe what you’re testing (e.g. subject line test, segment, send date).
2. **Campaign data** — Enter Control and Variant: Total Sends, Unique Opens, Unique Clicks, Orders, Total Revenue (£).
3. **Frequentist tab** — Review verdicts, confidence intervals, and the copy-to-Sheets table. Use “Summarise Findings” for an AI summary (optional).
4. **Bayesian tab** — Review probability Variant is better, expected lift, revenue scenarios, and Step 6 “How We Calculated the Revenue Scenarios” if you want the full workings.

### Streamlit Cloud Deployment

1. Push your code to GitHub (do **not** commit `.env` — it’s in `.gitignore`).
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Click **New app** and connect your GitHub repository.
4. Set the main file path to: **app.py**.
5. **Configure API key** (for “Generate Summary” feature):
   - In Streamlit Cloud, go to your app **Settings**.
   - Open the **Secrets** tab.
   - Add:
     ```bash
     OPENROUTER_API_KEY = "your-api-key-here"
     ```
6. Deploy.

### Local Development — API Key Setup

For “Summarise Findings” (AI-generated summaries), you have three options:

**Option 1: .env file (easiest)**

- Create a `.env` file in the project root.
- Add:
  ```bash
  OPENROUTER_API_KEY=your-api-key-here
  ```
- The app loads this via `python-dotenv`.

**Option 2: Streamlit secrets**

- Create `.streamlit/secrets.toml`.
- Add:
  ```toml
  OPENROUTER_API_KEY = "your-api-key-here"
  ```

**Option 3: Environment variable**

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

**Note:** The app runs fully without the API key. Only the “Generate Summary” buttons in the Frequentist and Bayesian tabs require it. All other features work normally.

## Key Concepts

| Concept | Description |
|--------|-------------|
| **Frequentist** | Gives a clear pass/fail verdict (significant or not). Best for final reporting and trackers. |
| **Bayesian** | Gives a probability and revenue scenarios. Useful for mid-test reads and smaller samples. |
| **Revenue Per User (RPU)** | Total revenue ÷ sends. Captures both conversion and order value; can favour a variant with fewer orders but higher revenue. |
| **Historical priors** | Optional Bayesian setting: anchor the analysis with your typical rates and “avg sends per campaign” so small tests don’t swing wildly. |

## Dependencies

- **streamlit** — Web framework
- **numpy** — Numerical computing and Monte Carlo sampling
- **pandas** — Data handling and tables
- **scipy** — Frequentist tests (z-tests, Welch’s t-test) and distributions
- **plotly** — Interactive charts
- **requests** — HTTP requests for OpenRouter API
- **python-dotenv** — Load `.env` for local API key (optional)

## Additional Files

- **app.py** — Main Streamlit application (use this for deployment).
- **requirements.txt** — Python dependencies.
- **.env.example** — Template for local API key (copy to `.env`; do not commit `.env`).
- **.gitignore** — Excludes `.env`, `.streamlit/secrets.toml`, and common Python/IDE files.
