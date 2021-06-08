import ephem
from datetime import datetime
import pytz
from roosts.utils.nexrad_util import NEXRAD_LOCATIONS

def get_sunrise_time(station, input_daytime, silent=True):
    # Make an observer
    obs = ephem.Observer()

    # Provide lat, lon and elevation of radar
    sinfo = NEXRAD_LOCATIONS[station]
    obs.lat = str(sinfo['lat'])
    obs.lon = str(sinfo['lon'])
    obs.elev = sinfo['elev']

    tz = pytz.timezone(sinfo['tz'])
    localized_daytime = tz.localize(input_daytime) # add time zone info
    obs.date = localized_daytime.astimezone(pytz.utc) # convert to utc

    if not silent:
        print(obs.lat, obs.lon, obs.elev)

    # Taken from the refernce link
    # To get U.S. Naval Astronomical Almanac values, use these settings
    obs.pressure = 0
    obs.horizon = '-0:34'

    sun = ephem.Sun()
    sunrise = obs.previous_rising(sun).datetime()

    return sunrise
