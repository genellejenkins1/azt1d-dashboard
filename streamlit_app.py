"""
streamlit_app.py — deploy entrypoint for Streamlit Community Cloud.

Copies the committed synthetic demo data from sample_data/ into the
data/raw/ path that SubjectDataLoader expects, renaming each file to
the Subject N.csv convention. No subprocess or runtime code generation
needed — the sample CSVs are already in the repository.
"""
import shutil, runpy
from pathlib import Path

SRC_ROOT = Path("sample_data/CGM Records")
DST_ROOT = Path("data/raw/CGM Records")

def _setup_demo_data() -> None:
    """Mirror sample_data/ → data/raw/ with correct filenames."""
    for src_sub in sorted(SRC_ROOT.glob("Subject *")):
        if not src_sub.is_dir():
            continue
        dst_sub = DST_ROOT / src_sub.name
        dst_sub.mkdir(parents=True, exist_ok=True)

        # Accept synthetic.csv or any existing csv as the source
        src_csvs = list(src_sub.glob("*.csv"))
        if not src_csvs:
            continue

        dst_csv = dst_sub / f"{src_sub.name}.csv"
        if not dst_csv.exists():
            shutil.copy(src_csvs[0], dst_csv)

_setup_demo_data()

runpy.run_path("app.py", run_name="__main__")
