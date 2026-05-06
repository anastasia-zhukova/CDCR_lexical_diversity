import subprocess
import sys

script_dir = "/Workspace/Users/sergio_gonzalez3@epam.com/ms-coref"

cmd = [
    sys.executable,
    "-m", "torch.distributed.run",
    "--nproc_per_node=8",
    "--nnodes=1",
    "--master_addr=127.0.0.1",
    "--master_port=29500",
    "train.py",
    "--output-dir", "/Volumes/dmz/action_recognition/cdcr/ms_coref",
    "--data-dir", "data",
    "--num-epochs", "25",
    "--batch-size", "8192",
    "--micro-batch-size", "128",
    "--fp16",
]

print(f"Using Python: {sys.executable}")
print(f"CMD: {' '.join(cmd)}\n")

# Stream output live so progress is visible during the run
proc = subprocess.Popen(
    cmd,
    cwd=script_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()

print(f"\n=== Exit code: {proc.returncode} ===")
