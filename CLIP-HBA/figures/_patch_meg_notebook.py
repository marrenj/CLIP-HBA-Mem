"""One-off patch for examine_mlp_training.ipynb; delete after run."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent / "examine_mlp_training.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

repls = [
    (
        'megmem_tp100_df = pd.read_csv("../sweep_meg_out/20260331_140842/tp100/sweep_results.csv")\n'
        'top5_mse = megmem_tp100_df.nsmallest(5, "mean_val_mse")\n'
        "print(top5_mse)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_140842/tp100/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_mse, sep="\\n")',
    ),
    (
        'megmem_tp100_df = pd.read_csv("../sweep_meg_out/20260331_140842/tp100/sweep_results.csv")\n'
        'top5_rho = megmem_tp100_df.nlargest(5, "mean_val_rho")\n'
        "print(top5_rho)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_140842/tp100/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_rho, sep="\\n")',
    ),
    (
        'megmem_tp200_df = pd.read_csv("../sweep_meg_out/20260331_092603/tp200/sweep_results.csv")\n'
        'top5_mse = megmem_tp200_df.nsmallest(5, "mean_val_mse")\n'
        "print(top5_mse)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_092603/tp200/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_mse, sep="\\n")',
    ),
    (
        'megmem_tp200_df = pd.read_csv("../sweep_meg_out/20260331_092603/tp200/sweep_results.csv")\n'
        'top5_rho = megmem_tp200_df.nlargest(5, "mean_val_rho")\n'
        "print(top5_rho)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_092603/tp200/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_rho, sep="\\n")',
    ),
    (
        'megmem_tp300_df = pd.read_csv("../sweep_meg_out/20260331_143706/tp300/sweep_results.csv")\n'
        'top5_mse = megmem_tp300_df.nsmallest(5, "mean_val_mse")\n'
        "print(top5_mse)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_143706/tp300/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_mse, sep="\\n")',
    ),
    (
        'megmem_tp300_df = pd.read_csv("../sweep_meg_out/20260331_143706/tp300/sweep_results.csv")\n'
        'top5_rho = megmem_tp300_df.nlargest(5, "mean_val_rho")\n'
        "print(top5_rho)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_143706/tp300/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_rho, sep="\\n")',
    ),
    (
        'megmem_tp400_df = pd.read_csv("../sweep_meg_out/20260331_094644/tp400/sweep_results.csv")\n'
        'top5_mse = megmem_tp400_df.nsmallest(5, "mean_val_mse")\n'
        "print(top5_mse)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_094644/tp400/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_mse, sep="\\n")',
    ),
    (
        'megmem_tp400_df = pd.read_csv("../sweep_meg_out/20260331_094644/tp400/sweep_results.csv")\n'
        'top5_rho = megmem_tp400_df.nlargest(5, "mean_val_rho")\n'
        "print(top5_rho)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_094644/tp400/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_rho, sep="\\n")',
    ),
    (
        'megmem_tp600_df = pd.read_csv("../sweep_meg_out/20260331_010925/tp600/sweep_results.csv")\n'
        'top5_mse = megmem_tp600_df.nsmallest(5, "mean_val_mse")\n'
        "print(top5_mse)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_010925/tp600/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_mse, sep="\\n")',
    ),
    (
        'megmem_tp600_df = pd.read_csv("../sweep_meg_out/20260331_010925/tp600/sweep_results.csv")\n'
        'top5_rho = megmem_tp600_df.nlargest(5, "mean_val_rho")\n'
        "print(top5_rho)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_010925/tp600/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_rho, sep="\\n")',
    ),
    (
        'megmem_tp900_df = pd.read_csv("../sweep_meg_out/20260331_093940/tp900/sweep_results.csv")\n'
        'top5_mse = megmem_tp900_df.nsmallest(5, "mean_val_mse")\n'
        "print(top5_mse)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_093940/tp900/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_mse, sep="\\n")',
    ),
    (
        'megmem_tp1300_df = pd.read_csv("../sweep_meg_out/20260331_011128/tp1300/sweep_results.csv")\n'
        'top5_mse = megmem_tp1300_df.nsmallest(5, "mean_val_mse")\n'
        "print(top5_mse)",
        'meg = load_meg_sweep_results("../sweep_meg_out/20260331_011128/tp1300/sweep_results.csv")\n'
        'print(meg.time_point, meg.top5_mse, sep="\\n")',
    ),
]

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    for old, new in repls:
        if old in src:
            src = src.replace(old, new)
    lines = src.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        cell["source"].append(lines[-1])

p.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("patched", p)
