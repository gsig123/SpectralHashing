#!/usr/bin/python3

# Run setup.sh to prepare the sift dataset
# Only run this file manually in case something went wrong during setup.

import struct


def update_progress(progress):
    """Update progress bar in terminal.
    Progress should be a float between 0 and 1"""
    print('\r[{0:50s}] {1:.1f}%'
          .format('#'*int(50*progress),100*progress),
          end="",flush=True)


f = open('./sift/sift_base.fvecs', 'rb')
output = open('sift.dat', 'w')
counter = 1
sift_linecount = 1000000
print('Parsing dataset, please do not interrupt')
while(True):
    try:
        data = f.read(4)
        d = struct.unpack('<i', data)[0]
        vector = [struct.unpack('<f', f.read(4))[0] for x in range(d)]
        mystring = ' {}'*d
        output.write(str(counter)+(' {}'*len(vector)).format(*vector)+'\n')
        counter += 1
        if counter % 1000 == 0:
            update_progress(counter/sift_linecount)
    except:
        f.close
        output.close
        print('\nEnd of file reached.\nParsed file available in sift_base.dat')
        break
