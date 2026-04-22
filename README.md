# Testimonial Sentiment Analysis App

Streamlit app for uploading a testimonials CSV, running Gemini-based sentiment analysis, extracting reusable quotes, and exporting a scored CSV. The app uses a master cache CSV so only new or changed testimonials need to be analyzed on later runs.

## What It Does

- Upload a testimonials CSV in the browser
- Reuse prior results from a master cache CSV
- Analyze only new or changed rows
- Generate sentiment, theme, quote, and review fields
- Download:
  - scored output for the current upload
  - updated master cache CSV for the next run

## Main Files

- `app.py`: Streamlit UI and cache workflow
- `testimonial_pipeline.py`: sentiment, quote extraction, scoring, and cache logic
- `test_pipeline_regression.py`: regression tests for deterministic pipeline behavior
- `requirements.txt`: Python dependencies

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key or enter it in the app UI:

```bash
export GEMINI_API_KEY=your_api_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

## How Incremental Reuse Works

- Each testimonial row gets a stable `row_key`
- Each row also gets a `content_hash`
- Cached results are reused only when:
  - the `row_key` matches
  - the `content_hash` matches
  - the `pipeline_version` matches

If only 5 rows are new or changed, only those 5 should be sent for analysis.

## Testing

Run the regression tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m unittest test_pipeline_regression.py
```

Check syntax:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile app.py testimonial_pipeline.py test_pipeline_regression.py
```

## Git Notes

Local CSVs, notebooks, caches, and other working artifacts are intentionally ignored by `.gitignore` so the repository stays focused on the app code. If you want to include a sanitized sample dataset later, add it explicitly and adjust `.gitignore`.
