from glob import glob
import os
import subprocess
import time
import pdb

for path in glob("*.sh"):
    subprocess.run(
        [
            "sbatch",
            f"{path}",
        ]
    )
