import json
from pathlib import Path

p = Path(__file__).resolve().parent / "examine_mlp_training.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
old = 'megmem_tp100_df = pd.read_csv("../sweep_meg_out/20260331_140842/tp100/sweep_results.csv")'
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and any(old in line for line in cell.get("source", [])):
        print("found")
        break
else:
    print("not found")
