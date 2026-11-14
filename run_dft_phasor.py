import tkinter as tk
from tkinter import filedialog
import subprocess

print("-----running dft phasor script-----")

root = tk.Tk()
root.withdraw()

# Open the file explorer and capture the selected file path
irf_path = filedialog.askopenfilename(
    initialdir="/",
    title="Select an IRF file",
    filetypes=(("TIFF files", "*.tif"), ("All files", "*.*"))
)

data_path = filedialog.askdirectory(
    initialdir="/",
    title="Select a data directory"
)

out_path = filedialog.askdirectory(
    initialdir="/",
    title="Select an output directory"
)

print(f"Reading data: {data_path}")
print(f"IRF file: {irf_path}")
print(f"Output directory: {out_path}")

cmd = ["C:\\Users\\leorb\\.conda\\envs\\image\\python.exe",
       "dft_bowman_method.py", data_path, "--irf", irf_path, "--out", out_path, "--hide-plots"]

cmd = " ".join(cmd)
print("Command:", cmd)

subprocess.run(cmd)
