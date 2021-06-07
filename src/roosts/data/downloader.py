import os
import pytz
import logging
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

    def __init__(self, min_before_sunrise=30, min_after_sunrise=90):
        self.min_before_sunrise = min_before_sunrise
        self.min_after_sunrise = min_after_sunrise


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
        self.log_dir = os.path.join(outdir, 'logs')
        self.not_s3_log_dir = os.path.join(outdir, 'not_s3_logs')
        self.error_scans_log_dir = os.path.join(outdir, 'error_scans_logs')
        os.makedirs(self.outdir, exist_ok = True)
        os.makedirs(self.log_dir, exist_ok = True)
        os.makedirs(self.not_s3_log_dir, exist_ok = True)
        os.makedirs(self.error_scans_log_dir, exist_ok = True)


    def __iter__(self):
        return self 


    def __len__(self):
        return self.num_days


    def __next__(self):
        if self.index == self.num_days:
            return StopIteration
        scan_paths = self.download_scans(self.days[self.index])
        self.index = self.index + 1
        return scan_paths


    def download_scans(self, current_date):
        """ Download radar scans from AWS """
        current_date_str = current_date.strftime('%Y-%m-%d')
        scan_paths = [] # list of the file path of downloaded scans

        def log_success(scan):
            with open(os.path.join(self.log_dir, f"{self.station}_{current_date_str}.log"), 'a+') as f:
                f.write(scan + '\n')

        def log_not_s3(scan):
            with open(os.path.join(self.not_s3_log_dir, f"{self.station}_{current_date_str}.log"), 'a+') as f:
                f.write(scan + '\n')

        def log_error_scans(scan):
            with open(os.path.join(self.error_scans_log_dir, f"{self.station}_{current_date_str}.log"), 'a+') as f:
                f.write(scan + '\n')

        sunrise = get_sunrise_time(self.station, current_date)
        start_time = sunrise - timedelta(minutes=self.min_before_sunrise)
        end_time = sunrise + timedelta(minutes=self.min_after_sunrise)

        keys = get_station_day_scan_keys(start_time, end_time, self.station)
        keys = set(keys) # aws keys
        for key in tqdm(keys, desc="Downloading"):
            try:
                filepath = download_scans([key], self.outdir)
                scan_paths.append(filepath)
                log_success(key)
            except ClientError as err:
                error_code = int(err.response['Error']['Code'])
                if error_code == 404:
                    log_not_s3(key)
                else:
                    log_error_scans(key)
            except Exception as ex:
                log_error_scans(key)

        return scan_paths


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


