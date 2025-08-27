import sys, subprocess, os

# Small, fast sanity run
cmd = [
    sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "train_criticF_v2.py"),
    "--precision", "float32",
    "--children", "binomial",
    "--max_epochs", "5",
    "--N", "20000",
    "--save_dir", "saved_F_model_quick"
]

print("Launching:\n ", " ".join(cmd))
ret = subprocess.run(cmd, check=False)
print("\nReturn code:", ret.returncode)
