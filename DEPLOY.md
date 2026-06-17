# Deploying the AZT1D Dashboard (live link)

The app runs on **synthetic demo data** — real patient CGM data is never deployed.

## Streamlit Community Cloud (free, ~3 minutes)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app**.
3. Pick this repo, branch `main`, and set the entry file to **`streamlit_app.py`**
   (it generates synthetic data on first run, then launches `app.py`).
4. Deploy. You'll get a public URL like `https://<you>-azt1d.streamlit.app`.
5. Add the URL to your README badge and portfolio.

## Local run
```bash
pip install -r requirements.txt
python ml/generate_synthetic_data.py --subjects 6 --days 14 --out "data/raw/CGM Records"
streamlit run streamlit_app.py
```

## Notes
- `data/raw/` is gitignored; the entrypoint regenerates synthetic data on the server.
- To demo on real data locally, `dvc pull` first (requires dataset access).
