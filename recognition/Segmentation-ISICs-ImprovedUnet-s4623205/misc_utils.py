import sys
from math import ceil


def progressbar(current, max_size):
    """
    Function for displaying the progress in the console

    Parameters
    ----------
    current : integer
      The current progress count
    max_size : integer
      The maximum progress count
    """
    sys.stdout.write('\r')
    progress = ceil((100 / int(max_size)) * current)
    sys.stdout.write("[%-100s] %d%%" % ('=' * progress, progress))
    sys.stdout.flush()