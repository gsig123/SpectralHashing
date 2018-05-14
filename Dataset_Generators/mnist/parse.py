#!/usr/bin/python3

import struct

def update_progress(progress):
    """Update progress bar in terminal.
    Progress should be a float between 0 and 1"""
    print('\r[{0:50s}] {1:.1f}%'
          .format('#'*int(50*progress), 100*progress),
          end='', flush=True)


print("Parsing MNIST dataset")    
with open('train-images-idx3-ubyte', 'rb') as input:
    with open('mnist.dat', 'w') as output:
        (k,) = struct.unpack('>i', input.read(4))
        if k != 2051:
            raise Exception('invalid magic number')
        (n,) = struct.unpack('>i', input.read(4))
        (r,) = struct.unpack('>i', input.read(4))
        (c,) = struct.unpack('>i', input.read(4))
        for i in range(n):
            row = list(struct.unpack('%dB' % (r * c), input.read(r * c)))
            output.write(str(i)+ (' {}'*len(row)).format(*row)+'\n')
            if (i + 1) % 1000 == 0:
                update_progress(i/60000)
        output.close
    input.close

print("\n")
