import os
import time

NUM_CPUS = 7
# station, start date (inclusive), end date (inclusive)
# ARGS = [
#     ("KTYX", "20160803", "20160804"),
#     ("KBUF", "20180109", "20180109"),
#     ("KBUF", "20180123", "20180123"),
#     ("KDOX", "20081002", "20081002"),
#     ("KDOX", "20081010", "20081010")
# ]
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
         ("20200101", "20200331"), ("20200401", "20200630"),
         ("20200701", "20200930"), ("20201001", "20201231"),]
SUN_ACTIVITY = "sunrise"
MIN_BEFORE = 30
MIN_AFTER = 90
# directory for system outputs
EXPERIMENT_NAME = "all_stations_v1" # "c4"
DATA_ROOT = f"/mnt/nfs/scratch1/wenlongzhao/roosts_data/{EXPERIMENT_NAME}"


ARGS = [(s, t[0], t[1]) for s in STATIONS for t in TIMES]
for args in ARGS:
    station = args[0]
    start = args[1]
    end = args[2]
    
    slurm_logs = f"slurm_logs/{EXPERIMENT_NAME}/{station}"
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
    --time=12:00:00 \
    demo.sbatch \
    --station {station} --start {start} --end {end} \
    --sun_activity {SUN_ACTIVITY} --min_before {MIN_BEFORE} --min_after {MIN_AFTER} \
    --data_root {DATA_ROOT}'''
    
    os.system(cmd)
    time.sleep(1)