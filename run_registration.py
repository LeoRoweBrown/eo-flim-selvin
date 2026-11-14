import tkinter as tk
from tkinter import filedialog
import subprocess

root = tk.Tk()
root.withdraw()

# Open the file explorer and capture the selected file path
reg_path = filedialog.askopenfilename(
    initialdir="/",
    title="Select a registration file",
    filetypes=(("Text files", "*.json"), ("All files", "*.*"))
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
print(f"Registration file: {reg_path}")
print(f"Output directory: {out_path}")


cmd = ["C:\\Users\\leorb\\.conda\\envs\\image\\python.exe",
       "register_all.py", data_path, "--reg", reg_path, "--out", out_path]

cmd = " ".join(cmd)
print(cmd)

subprocess.run(cmd)
