from __future__ import print_function
import re


CONFIG = 'run.conf'


if __name__ == '__main__':
    f = open(CONFIG)

    for l in f:
        if 'RES' in l:
            print(l.split()[2][:-1])
            
