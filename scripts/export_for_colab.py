"""Bundle code + the data the joint model needs into colab/lade_colab.zip (upload it to Google Drive).
Data are written as Parquet so they load on Colab's pandas regardless of the local pandas version.
  python scripts/export_for_colab.py [--datasets pickup delivery]"""
import argparse, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from lade import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
ap.add_argument("--code-only", action="store_true", help="small colab/lade_code_<tag>.zip (code only) to unzip over the data")
ap.add_argument("--tag", default="dev", help="version name in the code zip's file name and CODE_VERSION.txt, e.g. run3")
args = ap.parse_args()

out_dir = C.ROOT / "colab"; tmp = out_dir / "parquet"; tmp.mkdir(parents=True, exist_ok=True)
files = []
for name in ([] if args.code_only else args.datasets):
    for part in ["snaps", "nodes"]:
        stem = f"{name}_{part}"
        df = pd.read_pickle(C.PROC / f"{stem}.pkl")
        p = tmp / f"{stem}.parquet"
        df.to_parquet(p, compression="zstd", index=True)
        back = pd.read_parquet(p)                     # round-trip check
        assert back.shape == df.shape and list(back.columns) == list(df.columns), stem
        for c in df.columns[(back.dtypes != df.dtypes).values]:   # string storage may differ; values must not
            assert pd.api.types.is_string_dtype(df[c]) and (back[c].astype(object).values == df[c].astype(object).values).all(), (stem, c)
        print(f"{stem}: {df.shape}, {p.stat().st_size / 1e6:.0f} MB")
        files.append(p)

import datetime
version = f"{args.tag} (built {datetime.datetime.now():%Y-%m-%d %H:%M})"
zp = out_dir / (f"lade_code_{args.tag}.zip" if args.code_only else "lade_colab.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:            # parquet is already compressed
    for p in files:
        z.write(p, f"lade-route-eta/data/processed/{p.name}")
    for sub in ["src", "scripts", "tests"]:
        for f in (C.ROOT / sub).rglob("*.py"):
            z.write(f, f"lade-route-eta/{f.relative_to(C.ROOT).as_posix()}")
    for f in ["CLAUDE.md", "colab/requirements_colab.txt"]:
        z.write(C.ROOT / f, f"lade-route-eta/{f}")
    z.writestr("lade-route-eta/CODE_VERSION.txt", version + "\n")
print(version)
print(f"wrote {zp} ({zp.stat().st_size / 1e6:.0f} MB)")
