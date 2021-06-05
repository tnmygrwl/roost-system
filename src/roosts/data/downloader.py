import sys
import os
import calendar
import pytz
import re
import time
import logging
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from roosts.utils.sunset_util import get_sunset_sunrise_time
from roosts.utils.s3_util import get_scans, download_one_scan
import roosts.utils.nexrad_util as nexrad
import roosts.utils.geo_util as geoUtil
import roosts.utils.file_util as fileUtil
from tqdm import tqdm

def format_time(start_date_string, end_date_string):
    # INPUT: yyyymmdd, yyyymmdd
    # OUTPUT: datetime object
    year1 = int(start_date_string[:4])
    month1 = int(start_date_string[4:6])
    day1 = int(start_date_string[6:])
    year2 = int(end_date_string[:4])
    month2 = int(end_date_string[4:6])
    day2 = int(end_date_string[6:])
    first_day = day1
    last_day = day2
    start_date = datetime(year1, month1, first_day, 0, 0, 0, 0, tzinfo=pytz.utc)
    end_date = datetime(year2, month2, last_day, 0, 0, 0, 0, tzinfo=pytz.utc)
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
        This class iteratively downloads the randar scans in a certain date range from a certain radar station
        in a daily basis. Station plus one-day scans is the minimum unit of tracking roosts. 
        After downloading the scans, convert the files into numpy array, render ref1/cv1 and delete the scans
    """

    def __init__(self, min_before_sunrise=30, min_after_sunrise=90):

        
        self.min_before_sunrise = min_before_sunrise
        self.min_after_sunrise = min_after_sunrise


    def set_request(self, request, outdir):
        """ 
            index: the current index 
            request: {"station": "KDOX", "date": (start_date, end_date)}
        """

        self.station = request["station"]
        self.days = get_days_list(request["date"][0], request["date"][1])
        self.index = 0
        self.num_days = len(self.days)

        # scan path 
        self.outdir = outdir
        self.not_s3_dir = os.path.join(outdir, 'not_s3')
        self.error_scans_dir = os.path.join(outdir, 'error_scans')
        fileUtil.mkdir(self.outdir)
        fileUtil.mkdir(self.not_s3_dir)
        fileUtil.mkdir(self.error_scans_dir)
 

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

        not_s3 = []
        error_scans = []
        scan_paths = [] # list of the file path of downloaded scans 

        date_str = current_date.strftime('%Y-%m-%d')
        sunset, sunrise = get_sunset_sunrise_time(self.station, date_str)
        start_time = sunrise - timedelta(minutes=self.min_before_sunrise)
        end_time = sunrise + timedelta(minutes=self.min_after_sunrise)

        scans = get_scans(start_time, end_time, [self.station], with_station=False)
        scans = set(scans)
        for scan in tqdm(scans, desc="Downloading"):
            try:
                filepath = download_one_scan(scan, self.outdir)
                scan_paths.append(filepath)
            except ClientError as err :
                error_code = int(err.response['Error']['Code'])
                if error_code == 404:
                    not_s3.append(scan)
                else:
                    error_scans.append(scan)
            except Exception as ex:
                error_scans.append(scan)

        if len(not_s3) > 0:
            not_s3_file = '%s/%s'%(self.not_s3_dir, current_date)
            with open(not_s3_file, 'w') as f:
                f.write('\n'.join(not_s3))

        if len(error_scans) > 0:
            error_scans_file = '%s/%s'%(self.error_scans_dir, current_date)
            with open(error_scans_file, 'w') as f:
                f.write('\n'.join(error_scans))

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



