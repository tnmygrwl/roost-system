from datetime import datetime, timedelta
import pytz
from roosts.utils.nexrad_util import NEXRAD_LOCATIONS

def format_time(date_string):
    # INPUT: yyyymmdd
    # OUTPUT: datetime objects indicating noon of the days without time zone info
    year = int(date_string[:4])
    month = int(date_string[4:6])
    day = int(date_string[6:])
    formatted_date = datetime(year, month, day, 12, 0)
    return formatted_date


def get_days_list(start_date_str, end_date_str):
    start_date = format_time(start_date_str)
    end_date = format_time(end_date_str) + timedelta(days=1) # right exclusive
    days = []
    current_date = start_date
    while current_date < end_date:
        days.append(current_date)
        current_date += timedelta(days=1)
    return days


def utc_to_local_time(scan):
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