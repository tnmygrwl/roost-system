#Ref - https://stackoverflow.com/questions/2637293/calculating-dawn-and-sunset-times-using-pyephem/18622944#18622944
import ephem
from roosts.utils.nexrad_util import NEXRAD_LOCATIONS

#Make an observer
radar = ephem.Observer()

def get_sunset_sunrise_time(station, date):
    #DateTime in UTC (6 pm is to ensure its day time in US, so that we get the right sunset time)
    radar.date="%s 18:00:00"%(date)
    
    #Provide lat, lon and elevation of radar
    sinfo = NEXRAD_LOCATIONS[station]
    radar.lat = str(sinfo['lat'])
    radar.lon = str(sinfo['lon'])
    radar.elev = sinfo['elev']
    
    # print radar.lat, radar.lon, radar.elev
    
    #Taken from the refernce link
    #To get U.S. Naval Astronomical Almanac values, use these settings
    radar.pressure = 0
    radar.horizon = '-0:34'
    
    sun = ephem.Sun()
    
    sunset  = radar.next_setting(sun).datetime()
    sunrise = radar.next_rising(sun).datetime()
    
    return sunset, sunrise


def get_sunrise_time(station, date):
    #DateTime in UTC (6 pm is to ensure its day time in US, so that we get the right sunset time)
    radar.date="%s 18:00:00"%(date)
    
    #Provide lat, lon and elevation of radar
    sinfo = NEXRAD_LOCATIONS[station]
    radar.lat = str(sinfo['lat'])
    radar.lon = str(sinfo['lon'])
    radar.elev = sinfo['elev']
    
    #Taken from the refernce link
    #To get U.S. Naval Astronomical Almanac values, use these settings
    radar.pressure = 0
    radar.horizon = '-0:34'
    
    sun = ephem.Sun()
    
    sunrise = radar.previous_rising(sun).datetime()

    return sunrise.minute + sunrise.hour * 60.
    

if __name__ == '__main__':
    print(get_sunset_sunrise_time('KDOX', '2010-03-01'))
    print(get_sunrise_time('KDOX', '2010-03-02'))
