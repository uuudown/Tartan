from __future__ import print_function
import re


CONFIG = 'run.conf'


if __name__ == '__main__':
    f = open(CONFIG)
    p = re.compile('[0-9]+')
    for l in f:
        if 'NSS' in l:
            print(p.search(l).group())
            
