import os
import time

# station, start date (inclusive), end date (inclusive)
ARGS = [
    ("KTYX", "20160726", "20160729")
    # ("KTYX", "20170801", "20170831"),
]
# directory for system outputs
DATA_ROOT = "/mnt/nfs/scratch1/wenlongzhao/roosts_data"

for args in ARGS:
    station = args[0]
    start = args[1]
    end = args[2]
    
    slurm_logs = f"slurm_logs/{station}"
    slurm_output = os.path.join(slurm_logs, f"{station}_{start}_{end}.out")
    slurm_error = os.path.join(slurm_logs, f"{station}_{start}_{end}.err")
    os.makedirs(slurm_logs, exist_ok=True)
    
    cmd = f'''sbatch \
    --job-name="{station}_{start}_{end}" \
    --output="{slurm_output}" \
    --error="{slurm_error}" \
    --nodes=1 \
    --ntasks=1 \
    --mem=1000 \
    --partition=longq \
    --time=1-00:00 \
    demo.sbatch --station {station} --start {start} --end {end} --data_root {DATA_ROOT}'''
    
    os.system(cmd)
    time.sleep(1)