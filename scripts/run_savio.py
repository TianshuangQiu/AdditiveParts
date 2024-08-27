from glob import glob
import os
import subprocess
import time
import pdb

paths = glob("*.sh")
paths.sort()
for path in paths:
    subprocess.run(
        [
            "sbatch",
            f"{path}",
        ]
    )
