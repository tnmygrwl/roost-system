import os
import pytz
import logging
import time
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from roosts.utils.sun_activity_util import get_sun_activity_time
from roosts.utils.s3_util import get_station_day_scan_keys, download_scans
import roosts.utils.file_util as fileUtil
from tqdm import tqdm


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


class Downloader:

    """ 
        This class iteratively downloads the radar scans in a certain date range from a certain radar station
        in a daily basis. Station-day is the minimum unit of tracking roosts.
    """

    def __init__(
            self, sun_activity, min_before, min_after, log_dir,
    ):
        assert sun_activity in ["sunrise", "sunset"]
        self.sun_activity = sun_activity
        self.min_before = min_before
        self.min_after = min_after
        self.log_dir = log_dir


    def set_request(self, request, output_dir):
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
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok = True)


    def __iter__(self):
        return self 


    def __len__(self):
        return self.num_days


    def __next__(self):
        start_time = time.time()
        if self.index == self.num_days:
            return StopIteration
        date_string, date_station_predix, logger, scan_paths = self.download_scans(self.days[self.index])
        self.index = self.index + 1
        return date_string, date_station_predix, logger, scan_paths, start_time


    def download_scans(self, current_date):
        """ Download radar scans from AWS """

        date_string = current_date.strftime('%Y%m%d') # yyyymmdd
        date_station_predix = os.path.join(
            current_date.strftime('%Y'),
            current_date.strftime('%m'),
            current_date.strftime('%d'),
            self.station
        ) # yyyy/mm/dd/ssss
        log_dir = os.path.join(self.log_dir, self.station, current_date.strftime('%Y'))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.station}_{date_string}.log")
        logger = logging.getLogger(date_station_predix)
        filelog = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        formatter.converter = time.gmtime
        filelog.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(filelog)

        sun_activity_time = get_sun_activity_time(self.station, current_date, sun_activity=self.sun_activity)
        start_time = sun_activity_time - timedelta(minutes=self.min_before)
        end_time = sun_activity_time + timedelta(minutes=self.min_after)
        keys = get_station_day_scan_keys(start_time, end_time, self.station)
        keys = sorted(list(set(keys))) # aws keys

        scan_paths = [] # list of the file path of downloaded scans
        for key in tqdm(keys, desc="Downloading"):
            try:
                filepath = download_scans([key], self.output_dir)
                scan_paths.append(filepath)
                logger.info('[Download Success] scan %s' % key.split("/")[-1])
            except Exception as ex:
                logger.error('[Download Failure] scan %s - %s' % (key.split("/")[-1], str(ex)))

        return date_string, date_station_predix, logger, scan_paths


if __name__ == "__main__":
    
    request = {"station": "KDOX", "date": ("20210527", "20210527")}
    output_dir = './scans'
    downloader = Downloader(request, output_dir)
    print(len(downloader))
    for scan_paths in downloader:
        if type(scan_paths) in (list,):
            print(len(scan_paths))
        else:
            break


