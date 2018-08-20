import tables
import numpy as np

if __name__ == '__main__':
    NX = 256
    NY = NX
    NZ = NX + 2

    Usx = np.fromfile('256/Usx.bin',dtype=np.float32)
    Usx = Usx.reshape((NX,NY,NZ))
    f = tables.openFile('256/Usx.h5','w')
    h = f.createArray('/','u',Usx)
    f.close()

    Usy = np.fromfile('256/Usy.bin',dtype=np.float32)
    Usy = Usy.reshape((NX,NY,NZ))
    f = tables.openFile('256/Usy.h5','w')
    h = f.createArray('/','u',Usy)
    f.close()

    Usz = np.fromfile('256/Usz.bin',dtype=np.float32)
    Usz = Usz.reshape((NX,NY,NZ))
    f = tables.openFile('256/Usz.h5','w')
    h = f.createArray('/','u',Usz)
    f.close()
