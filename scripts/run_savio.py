from glob import glob
import os
import subprocess
import pdb

for path in glob("*.sh"):
    subprocess.run(
        [
            "sbatch",
            "path",
        ]
    )
