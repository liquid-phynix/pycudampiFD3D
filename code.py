#!/usr/bin/python2

def debug(msg):
    print ('rank %d: ' + msg) % mpi.rank

def debug0(msg):
    if master:
        print msg

import argparse
parser = argparse.ArgumentParser(description = '3D Cuda finite difference solver')
args = parser.parse_args()

import numpy as np
from mpi4py import MPI

from parutils import pprint
#def pprint(*args):
#    if mpi.rank == 0:
#        print(*args)


# globals
mpi = MPI.COMM_WORLD
if mpi.size <= 1:
    print('mpi.size needs to be greater than 1')
    exit(1)
master = mpi.size == mpi.rank + 1
slave = not master
slaves = mpi.size - 1
master_rank = slaves
rank_left = (mpi.rank - 1) % slaves
rank_right = (mpi.rank + 1) % slaves

pprint('/'*74)
pprint('Running on %d cards' % slaves)
pprint('\\'*74)

mpi.Barrier()

#lx, ly, lz = 10.0*2.0*np.pi, 1.0*2.0*np.pi, 10.0*2.0*np.pi
lx, ly, lz = 20.0*2.0*np.pi, 20.0*2.0*np.pi, 20.0*2.0*np.pi
bx, by, bz = 8, 8, 8
gx, gy, gz = 20, 20, 20
nx, ny, nz = bx * gx, by * gy, bz * gz
dimx, dimy, dimz = nx + 6, ny + 6, nz + 6
pprint('domain is %d x %d x %d' % (dimx, dimy, dimz))
dx, dy, dz = float(lx) / float(nx), float(ly) / float(ny), float(lz) / float(nz)
ddm2 = dx ** (-2.0)
elements = dimx * dimy * dimz

def debug(msg):
    print(('rank %d: ' + msg) % mpi.rank)

# assigning GPU ids to ranks
if master:
    from collections import defaultdict
    mapping = defaultdict(list)
    for k, v in [mpi.recv(source = i, tag = 0x1334) for i in range(mpi.size - 1)]:
        mapping[k].append(v)

    for k, v in mapping.iteritems():
        for gpu_id, rank in enumerate(v):
            mpi.send(gpu_id, dest = rank, tag = 0x400b)
    gpu_id = -1
else:
    mpi.send((MPI.Get_processor_name(), mpi.rank), dest = master_rank, tag = 0x1334)
    gpu_id = mpi.recv(source = master_rank, tag = 0x400b)
print('rank %d has gpu_id %d' % (mpi.rank, gpu_id))
mpi.Barrier()

# compiling cubin and loading it up
if master:
    import subprocess
    subprocess.check_call(['nvcc', '--cubin', '-arch', 'sm_20', 'kernel_semi.cu'])
    for i in range(mpi.size - 1):
        mpi.send(None, dest = i, tag = 0xdeadbee)
#    debug('cubin compiled, msg sent')
else:
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(gpu_id)
    device.make_context()
    mpi.recv(source = master_rank, tag = 0xdeadbee)
    mod = cuda.module_from_file('kernel_semi.cubin')
    debug('msg received, cubin loaded')
    debug('# of devices: %d' % cuda.Device.count())
    debug('gpu_id: %d' % gpu_id)
mpi.Barrier()



def init_nucleus(out_ar):
    cos = np.cos
    tanh = np.tanh
    pi = np.pi
    sq3 = np.sqrt(3.0)
    def sc_rod(x, y, z):
        return 0.166666666666666666667 * (cos(x) + cos(x + z) + cos(z) + cos(x) + cos(x - z) + cos(z))
    def bcc(x, y, z):
        return 0.166666666666666666667 * (cos(x + y) + cos(x + z) + cos(y + z) + cos(x - y) + cos(x - z) + cos(y - z))
    def hex_rod(x, y, z):
        return 0.166666666666666666667 * (cos(x) + cos(x/2.0 + sq3*z/2.0) + cos(-x/2.0 + sq3*z/2.0))
    def hex_rod_rot(x, y, z):
        return 0.166666666666666666667 * (cos(z) + cos(z/2.0 + sq3*x/2.0) + cos(-z/2.0 + sq3*x/2.0))
    # xvec = np.linspace(0, lx, nx, endpoint = False)
    # yvec = np.linspace(0, ly, ny, endpoint = False)
    # zvec = np.linspace(mpi.rank * lz, (mpi.rank + 1) * lz, nz, endpoint = False)
    # X, Y, Z = np.meshgrid(xvec, yvec, zvec)
    [Z, Y, X] = np.mgrid[(mpi.rank * lz) : (mpi.rank + 1) * lz : (float(lz)/float(nz-0.5)), 0 : ly : dy, 0 : lx : dx]
#    [Z, Y, X] = np.mgrid[0 : lz : dz, 0 : ly : dy, 0 : lx : dx]
    #    out_ar[3:-3, 3:-3, 3:-3] = (1+tanh((8.0*pi)**2.0 - (X-lx/2.0)**2.0 - (Z-lz/2.0)**2.0))
    out_ar[3:-3, 3:-3, 3:-3] = -0.255 \
        + 0.5 * (1+tanh((8.0*pi)**2.0 - ((X-lx/2.0)**2.0 + (Z-lz/2.0)**2.0))) * (hex_rod(X, Y, Z) + (-0.25) - (-0.255)) \
        + 0.5 * (1+tanh((8.0*pi)**2.0 - ((X-lx/2.0)**2.0 + (Z-3.0*lz/2.0)**2.0))) * (hex_rod_rot(X, Y, Z) + (-0.25) - (-0.255))

    #    out_ar[3:-3, 3:-3, 3:-3] = -0.55 + (1+tanh((8.0*pi)**2.0 - (X-lx/2)**2.0 - (Z-lz/2)**2.0)) * 0.05 * bcc(X, Y, Z)

# setting up arrays on GPUs
if slave:
    h_domain_save = None
    #    h_domain = -0.2 + (np.random.rand(dimx, dimy, dimz).astype(np.float32)-0.5) / 50.0
    h_domain = np.zeros((dimz, dimy, dimx), dtype = np.float32)
    init_nucleus(h_domain)

    d_domain   = cuda.mem_alloc(h_domain.nbytes)
    d_domain_p = cuda.mem_alloc(h_domain.nbytes)

    d_left     = cuda.mem_alloc(dimx * dimy * 3 * 4)
    d_right    = cuda.mem_alloc(dimx * dimy * 3 * 4)

    h_give_left  = np.zeros(dimx * dimy * 3, dtype = np.float32)
    h_give_right = np.zeros(dimx * dimy * 3, dtype = np.float32)
    h_recv_left  = np.zeros(dimx * dimy * 3, dtype = np.float32)
    h_recv_right = np.zeros(dimx * dimy * 3, dtype = np.float32)

else: # master
    h_domains = [np.zeros((dimz - 6, dimx - 6), dtype = np.float32) for _ in range(mpi.size - 1)]
    #    h_domains = [np.zeros((dimx, dimz), dtype = np.float32) for _ in range(mpi.size - 1)]

mpi.Barrier()

debug0('ddm2 : %f' % ddm2)

iterations = 10000000
savefreq = 100

if slave:
    kernel_source = mod.get_function('kernel_source')
    kernel_timestep = mod.get_function('kernel_timestep')
    kernel_pbc_noz = mod.get_function('kernel_pbc_noz')
    kernel_ghost_copy = mod.get_function('kernel_ghost_copy')
    kernel_ghost_copy_inv = mod.get_function('kernel_ghost_copy_inv')
    # copy host array to device
    cuda.memcpy_htod(d_domain, h_domain)
    cuda.memcpy_htod(d_domain_p, h_domain)

    iters = 0

    # initial synchronization
    kernel_pbc_noz(d_domain_p, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (16, 16, 1), grid = (1, 1, 1))
    kernel_ghost_copy(d_domain_p, d_left, d_right, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (256, 1, 1), grid = (1, 1, 1))
    cuda.memcpy_dtoh(h_give_left, d_left)
    cuda.memcpy_dtoh(h_give_right, d_right)
    r_left = mpi.Irecv(h_recv_left, source = rank_left, tag = ((rank_left << 16) | 0x51))
    r_right = mpi.Irecv(h_recv_right, source = rank_right, tag = ((rank_right << 16) | 0x1e))
    mpi.Isend(h_give_left, dest = rank_left, tag = ((mpi.rank << 16) | 0x1e))
    mpi.Isend(h_give_right, dest = rank_right, tag = ((mpi.rank << 16) | 0x51))
    r_left.Wait()
    r_right.Wait()
    cuda.memcpy_htod(d_left, h_recv_left)
    cuda.memcpy_htod(d_right, h_recv_right)
    kernel_ghost_copy_inv(d_domain_p, d_left, d_right, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (256, 1, 1), grid = (1, 1, 1))
    cuda.memcpy_dtod(d_domain, d_domain_p, h_domain.nbytes)

    mpi.Barrier()
    while iters < iterations:
        mpi.Barrier()
        if iters % 100 == 0:
            debug(str(iters))
        if iters % savefreq == 0:
            # copy domain to host
            cuda.memcpy_dtoh(h_domain, d_domain)
#            h_domain_save[:] = h_domain[3:-3,3,3:-3].copy()[:]
#            h_domain_save = h_domain[3:-3,3,3:-3].copy()
            #            h_domain_save = h_domain[:,3,:].copy()
            mpi.Send(h_domain[3:-3,3,3:-3].copy(), dest = master_rank, tag = 0x5a43)
            mpi.Barrier()

        iters += 1

        # source term
        # if mpi.rank == 0:
        #     kernel_source(d_domain, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (1,1,1), grid = (1,1))
        # timestep: compute d_domain_p from d_domain
        kernel_timestep(d_domain, d_domain_p, np.int32(dimx), np.int32(dimy), np.int32(dimz), np.float32(ddm2), block = (bx, by, bz), grid = (gx, gy, gz))
        # writing out source term before applying PBC
        # if mpi.rank == 0:
        #     kernel_source(d_domain_p, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (1,1,1), grid = (1,1))

        # ensure PBC on d_domain_p except in the z direction
        kernel_pbc_noz(d_domain_p, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (16, 16, 1), grid = (1, 1, 1))

        # copy ghost z direction ghost zones to linear memory on device
        kernel_ghost_copy(d_domain_p, d_left, d_right, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (256, 1, 1), grid = (1, 1, 1))
        # copy ghost z direction ghost zones from device to host
        cuda.memcpy_dtoh(h_give_left, d_left)
        cuda.memcpy_dtoh(h_give_right, d_right)
        # send & recv these
        r_left = mpi.Irecv(h_recv_left, source = rank_left, tag = ((rank_left << 16) | 0x51))
        r_right = mpi.Irecv(h_recv_right, source = rank_right, tag = ((rank_right << 16) | 0x1e))
        mpi.Isend(h_give_left, dest = rank_left, tag = ((mpi.rank << 16) | 0x1e))
        mpi.Isend(h_give_right, dest = rank_right, tag = ((mpi.rank << 16) | 0x51))
        r_left.Wait()
        r_right.Wait()
        # copy ghost z direction ghost zones from host to device
        cuda.memcpy_htod(d_left, h_recv_left)
        cuda.memcpy_htod(d_right, h_recv_right)
        # copy ghost z direction ghost zones from linear memory on device
        kernel_ghost_copy_inv(d_domain_p, d_left, d_right, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (256, 1, 1), grid = (1, 1, 1))

        # finalize: copy d_domain_p over to d_domain
        cuda.memcpy_dtod(d_domain, d_domain_p, h_domain.nbytes)

    cuda.Context.pop()
else: # master
    iters = 0
    mpi.Barrier()
    while iters < iterations:
        mpi.Barrier()
        if iters % savefreq == 0:
            [mpi.Recv(h_domains[i], source = i, tag = 0x5a43) for i in range(mpi.size - 1)]
            save = np.vstack(h_domains).transpose()
            save.tofile('out_%.8d_%dx%d.bin' % (iters, save.shape[1], save.shape[0]))
            mpi.Barrier()
        iters += 1

mpi.Barrier()

debug0('finished %d iterations' % iterations)
