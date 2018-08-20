from __future__ import division, print_function
import numpy as np
import tables

def ddz(hat):
    nx = hat.shape[0]
    kz = np.arange(nx//2+1).astype(np.complex64)
    kz = kz.reshape((1,1,nx//2+1))

    return 1j*kz*hat

def ddy(hat):
    nx = hat.shape[0]
    kz = np.arange(nx//2+1).astype(np.complex64)
    ky = np.zeros((nx,), dtype=np.complex64)
    ky[:nx//2] = kz[:nx//2]
    ky[nx//2:] = -kz[:0:-1]

    ky = ky.reshape((1,nx,1))
    return 1j*ky*hat

def ddx(hat):
    nx = hat.shape[0]
    kz = np.arange(nx//2+1).astype(np.complex64)
    kx = np.zeros((nx,), dtype=np.complex64)
    kx[:nx//2] = kz[:nx//2]
    kx[nx//2:] = -kz[:0:-1]

    kx = kx.reshape((nx,1,1))

    return 1j*kx*hat

def readfield(filename):
    uf = tables.openFile(filename)
    (nx,ny,nz) = uf.root.u.shape
    uhat = np.empty((nx,nx,nx//2+1), dtype=np.complex64)
    uhat.real = uf.root.u[:,:,::2]
    uhat.imag = uf.root.u[:,:,1::2]
    uf.close()

    uhat[0,0,0] = 0+0*1j
    return uhat

def fou2phys(uhat):
    return np.fft.irfftn(uhat)

def enstrophy(uhat,vhat,what):
    omega_x_hat = ddy(what) - ddz(vhat)
    omega_y_hat = ddz(uhat) - ddx(what)
    omega_z_hat = ddx(vhat) - ddy(uhat)

    return np.fft.irfftn(omega_x_hat)**2 + \
        np.fft.irfftn(omega_y_hat)**2 + \
        np.fft.irfftn(omega_z_hat)**2

def omag(uhat,vhat,what):
    omega_x_hat = ddy(what) - ddz(vhat)
    omega_y_hat = ddz(uhat) - ddx(what)
    omega_z_hat = ddx(vhat) - ddy(uhat)

    return np.sqrt(np.fft.irfftn(omega_x_hat)**2 + \
                   np.fft.irfftn(omega_y_hat)**2 + \
                   np.fft.irfftn(omega_z_hat)**2)
