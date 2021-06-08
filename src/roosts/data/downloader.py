import os
import pytz
import logging
import time
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from roosts.utils.sunrise_util import get_sunrise_time
from roosts.utils.s3_util import get_station_day_scan_keys, download_scans
import roosts.utils.file_util as fileUtil
from tqdm import tqdm


def format_time(start_date_string, end_date_string):
    # INPUT: yyyymmdd, yyyymmdd
    # OUTPUT: datetime objects indicating noon of the day without time zone info
    year1 = int(start_date_string[:4])
    month1 = int(start_date_string[4:6])
    day1 = int(start_date_string[6:])
    year2 = int(end_date_string[:4])
    month2 = int(end_date_string[4:6])
    day2 = int(end_date_string[6:])
    start_date = datetime(year1, month1, day1, 12, 0)
    end_date = datetime(year2, month2, day2, 12, 0)
    return start_date, end_date


def get_days_list(start_date_str, end_date_str):
    start_date, end_date = format_time(start_date_str, end_date_str)
    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date)
        current_date += timedelta(days=1)
    return days


class Downloader:

    """ 
        This class iteratively downloads the radar scans in a certain date range from a certain radar station
        in a daily basis. Station-day is the minimum unit of tracking roosts.
    """

    def __init__(self, min_before_sunrise, min_after_sunrise, log_dir):
        self.min_before_sunrise = min_before_sunrise
        self.min_after_sunrise = min_after_sunrise
        self.log_dir = log_dir


    def set_request(self, request, outdir):
        """ 
            index: the current index 
            request: {"station": "KDOX", "date": (start_date, end_date)}
        """
        assert type(request["station"]) is str
        self.station = request["station"]
        self.days = get_days_list(request["date"][0], request["date"][1])
        self.index = 0
        self.num_days = len(self.days)

        # scan path
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok = True)


    def __iter__(self):
        return self 


    def __len__(self):
        return self.num_days


    def __next__(self):
        start_time = time.time()
        if self.index == self.num_days:
            return StopIteration
        scan_paths, key_prefix, log_path, logger = self.download_scans(self.days[self.index])
        self.index = self.index + 1
        return scan_paths, start_time, key_prefix, log_path, logger


    def download_scans(self, current_date):
        """ Download radar scans from AWS """
        scan_paths = [] # list of the file path of downloaded scans

        key_prefix = os.path.join(current_date.strftime('%Y/%m/%d'), self.station)
        log_dir = os.path.join(self.log_dir, self.station)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.station}_{current_date.strftime('%Y%m%d')}.log")
        logger = logging.getLogger(key_prefix)
        filelog = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        formatter.converter = time.gmtime
        filelog.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(filelog)

        sunrise = get_sunrise_time(self.station, current_date)
        start_time = sunrise - timedelta(minutes=self.min_before_sunrise)
        end_time = sunrise + timedelta(minutes=self.min_after_sunrise)

        keys = get_station_day_scan_keys(start_time, end_time, self.station)
        keys = sorted(list(set(keys))) # aws keys
        for key in tqdm(keys, desc="Downloading"):
            try:
                filepath = download_scans([key], self.outdir)
                scan_paths.append(filepath)
                logger.info('[Download Success] scan %s' % key)
            except Exception as ex:
                logger.error('[Download Failure] scan %s - %s' % (key, str(ex)))

        return scan_paths, key_prefix, log_path, logger


if __name__ == "__main__":
    
    request = {"station": "KDOX", "date": ("20210527", "20210527")}
    outdir = './scans'
    downloader = Downloader(request, outdir)
    print(len(downloader))
    for scan_paths in downloader:
        if type(scan_paths) in (list,):
            print(len(scan_paths))
        else:
            break


