import os
import time


NUM_CPUS = 7
# station, start date (inclusive), end date (inclusive)
# ARGS = [("KTYX", "20160803", "20160804")]
# greatlakes_test
# ARGS = [
#     ("KBUF", "20100201", "20100331"),
#     ("KBUF", "20100801", "20100930"),
#     ("KBUF", "20170201", "20170331"),
#     ("KBUF", "20170801", "20170930"),
#     ("KCLE", "20100201", "20100331"),
#     ("KCLE", "20100801", "20100930"),
#     ("KCLE", "20170201", "20170331"),
#     ("KCLE", "20170801", "20170930"),
#     ("KTYX", "20100201", "20100331"),
#     ("KTYX", "20100801", "20100930"),
#     ("KTYX", "20170201", "20170331"),
#     ("KTYX", "20170801", "20170930"),
# ]
# deployment
STATIONS = ["KCLE", "KBUF", "KTYX", "KGRB", "KMQT", "KMKX",
            "KLOT", "KIWX", "KGRR", "KAPX", "KDTX", "KDLH"]
TIMES = [("20100101", "20100331"), ("20100401", "20100630"),
         ("20100701", "20100930"), ("20101001", "20101231"),
         ("20170101", "20170331"), ("20170401", "20170630"),
         ("20170701", "20170930"), ("20171001", "20171231"),]
ARGS = [(s, t[0], t[1]) for s in STATIONS for t in TIMES]
# directory for system outputs
EXPERIMENT_NAME = "c4"
DATA_ROOT = f"/mnt/nfs/scratch1/wenlongzhao/roosts_data/{EXPERIMENT_NAME}"

for args in ARGS:
    station = args[0]
    start = args[1]
    end = args[2]
    
    slurm_logs = f"slurm_logs/{station}"
    slurm_output = os.path.join(slurm_logs, f"{station}_{start}_{end}.out")
    slurm_error = os.path.join(slurm_logs, f"{station}_{start}_{end}.err")
    os.makedirs(slurm_logs, exist_ok=True)

    os.system(f"export MKL_NUM_THREADS={NUM_CPUS}")
    os.system(f"export OPENBLAS_NUM_THREADS={NUM_CPUS}")
    os.system(f"export OMP_NUM_THREADS={NUM_CPUS}")

    cmd = f'''sbatch \
    --job-name="{station}{start}_{end}" \
    --output="{slurm_output}" \
    --error="{slurm_error}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task={NUM_CPUS} \
    --mem-per-cpu=2000 \
    --partition=defq \
    demo.sbatch --station {station} --start {start} --end {end} --data_root {DATA_ROOT}'''
    
    os.system(cmd)
    time.sleep(1)