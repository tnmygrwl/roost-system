"""
Append local_date to web ui files
"""
import os
import pytz
from datetime import datetime
from roosts.utils.nexrad_util import NEXRAD_LOCATIONS


DIR = "/mnt/nfs/scratch1/wenlongzhao/roosts_data/bat/ui/scans_and_tracks/"


def key_to_local_time(scan):
    utc_time = datetime(
        int(scan[4:8]), # year
        int(scan[8:10]), # month
        int(scan[10:12]), # date
        int(scan[13:15]), # hour
        int(scan[15:17]), # min
        int(scan[17:19]), # sec
        tzinfo=pytz.utc
    )
    tz = pytz.timezone(NEXRAD_LOCATIONS[scan[:4]]['tz'])
    local_time = utc_time.astimezone(tz)
    return local_time.strftime('%Y%m%d_%H%M%S')

for file in os.listdir(DIR):
    lines = [line.strip() for line in open(os.path.join(DIR, file), "r").readlines()]

    if file.startswith("scans"):
        open(os.path.join(DIR, file), "w").writelines(
            [f"{line},{key_to_local_time(line)}\n" for line in lines]
        )

    elif file.startswith("tracks"):
        with open(os.path.join(DIR, file), "w") as f:
            f.writelines([f"{lines[0]},local_time\n"])
            f.writelines([f"{line},{key_to_local_time(line.split(',')[1])}\n" for line in lines[1:]])
