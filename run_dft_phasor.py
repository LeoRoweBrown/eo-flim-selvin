
import subprocess


code_root = "E:\\code\\eo_flim\\code_20261020\\eo-flim-selvin\\"
script_path = f"{code_root}dft_bowman_method.py"
irf_fn = "irf_tirf5048_10mW_10ms_20261111_112507_registered.tif"
root = "E:\\20261111\\20261111\\"
data = "E:\\20261111\\20261111\\registered\\"
out = f"{root}output"
irf = f"{data}{irf_fn}"

cmd = ["C:\\Users\\leorb\\.conda\\envs\\image\\python.exe",
       script_path, data, "--irf", irf, "--out", out]


cmd = " ".join(cmd)
print(cmd)

subprocess.run(cmd)
