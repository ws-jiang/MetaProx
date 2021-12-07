
from datetime import datetime
import time

def sleep_minute(n_minute):
    time.sleep(60 * n_minute)

def sleep_sec(n_sec):
    time.sleep(n_sec)

class TimeUtils:

    def __init__(self):
        pass

    DDYYYYMM = "%d%m%Y"
    YYYY_MM_DD = "%Y-%m-%d"
    YYYYMMDD = "%Y%m%d"
    YYYYMMDDHHMMSS = "%Y-%m-%d %H:%M:%S"
    YYYYMMDDHHMMSS_COMPACT = "%Y%m%d_%H%M%S"
    YYYYMMDDHHMM_COMPACT = "%Y%m%d_%H%M"

    @staticmethod
    def get_now_str(fmt=YYYYMMDDHHMMSS_COMPACT):
        return datetime.today().strftime(fmt)


class TimeCounter(object):
    def __init__(self):
        self.start = time.time()

    def count(self):
        return int(time.time() - self.start)

    def count_ms(self):
        ts = time.time() - self.start
        return int(ts * 1000)

