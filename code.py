#!/usr/bin/python2

import argparse
parser = argparse.ArgumentParser(description='3D Cuda finite difference solver')
args = parser.parse_args()

import numpy as np
from mpi4py import MPI

from parutils import pprint

# globals
mpi = MPI.COMM_WORLD
if mpi.size <= 1:
    print 'mpi.size needs to be greater than 0'
    exit(0)
master = mpi.size == mpi.rank + 1
master_rank = mpi.size - 1
slave = not master
rank_left = (mpi.rank - 1) % (mpi.size - 1)
rank_right = (mpi.rank + 1) % (mpi.size - 1)

pprint('|'*74)
pprint(' Running on %d cores' % mpi.size)
pprint('|'*74)

mpi.Barrier()

bx, by, bz = 8, 8, 8
gx, gy, gz = 10, 10, 10
dimx, dimy, dimz = bx * gx + 6, by * gy + 6, bz * gz + 6
elements = dimx * dimy * dimz

def debug(msg):
    print ('rank %d: ' + msg) % mpi.rank

# assigning GPU ids to ranks
if master:
    from collections import defaultdict
    mapping = defaultdict(list)
    for k, v in [mpi.recv(source = i, tag = 0x1334) for i in range(mpi.size - 1)]:
        mapping[k].append(v)

    for k, v in mapping.iteritems():
        for gpu_id, rank in zip(range(4), v):
            mpi.send(gpu_id, dest = rank, tag = 0x400b)
else:
    debug(' before send')
    mpi.send((MPI.Get_processor_name(), mpi.rank), dest = master_rank, tag = 0x1334)
    debug(' after send, before recv')
    gpu_id = mpi.recv(source = master_rank, tag = 0x400b)
    debug(' after recv')
mpi.Barrier()

# compiling cubin and loading it up
if master:
    import subprocess
    subprocess.check_call(['nvcc', '--cubin', '-arch', 'sm_20', 'kernel.cu'])
    for i in range(mpi.size - 1):
        mpi.send(None, dest = i, tag = 0xdeadbee)
    debug('cubin compiled, msg sent')
else:
    import pycuda.driver as cuda
    cuda.init()
    debug('init completed')
    device = cuda.Device(gpu_id)
    device.make_context()
    mpi.recv(source = master_rank, tag = 0xdeadbee)
    mod = cuda.module_from_file('kernel.cubin')
    debug('msg received, cubin loaded')
    debug('# of devices: %d' % cuda.Device.count())
    debug('gpu_id: %d' % gpu_id)
mpi.Barrier()
# setting up arrays on GPUs
if slave:
    h_domain_save = None
    h_domain = np.random.rand(dimx, dimy, dimz).astype(np.float32)
#    h_domain = np.empty((dimx, dimy, dimz), dtype = np.float32)
    d_domain = cuda.mem_alloc(h_domain.nbytes)
    d_domain_p = cuda.mem_alloc(h_domain.nbytes)

    d_left = cuda.mem_alloc(dimx * dimy * 3 * 4)
    d_right = cuda.mem_alloc(dimx * dimy * 3 * 4)

    h_give_left  = np.zeros(dimx * dimy * 3, dtype = np.float32)
    h_give_right = np.zeros(dimx * dimy * 3, dtype = np.float32)
    h_recv_left  = np.zeros(dimx * dimy * 3, dtype = np.float32)
    h_recv_right = np.zeros(dimx * dimy * 3, dtype = np.float32)

else: # master
    h_domains = [np.zeros((dimx, dimz), dtype = np.float32) for _ in range(mpi.size - 1)]
#    h_domains = [np.zeros((dimx, dimy, dimz), dtype = np.float32) for _ in range(mpi.size - 1)]
    

mpi.Barrier()

iterations = 1000
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
    mpi.Barrier()
    while iters < iterations:
        mpi.Barrier()
        if iters % 100 == 0:
            debug(str(iters))
        if iters % savefreq == 0:
            # copy domain to host
            cuda.memcpy_dtoh(h_domain, d_domain)
            h_domain_save = h_domain[:,3,:].copy()
            r = mpi.Isend(h_domain_save, dest = master_rank, tag = 0x5a43)
            r.Wait()
            mpi.Barrier()

        iters += 1
        # compute new values from d_domain to d_domain_p
        kernel_timestep(d_domain, d_domain_p, np.int32(dimx), np.int32(dimy), np.int32(dimz), np.float32(1), block = (bx, by, bz), grid = (gx, gy, gz))
#        debug('1')
        # ensure PBC on d_domain_p except in the z direction
        kernel_pbc_noz(d_domain_p, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (16, 16, 1), grid = (1, 1, 1))
#        debug('2')
        # copy ghost z direction ghost zones to linear memory on device
        kernel_ghost_copy(d_domain_p, d_left, d_right, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (256, 1, 1), grid = (1, 1, 1))
#        debug('3')
        # copy ghost z direction ghost zones from device to host
        cuda.memcpy_dtoh(h_give_left, d_left)
        cuda.memcpy_dtoh(h_give_right, d_right)
        # send & recv these
        r_left = mpi.Irecv(h_recv_left, source = rank_left, tag = rank_left)
        r_right = mpi.Irecv(h_recv_right, source = rank_right, tag = rank_right)
        mpi.Isend(h_give_left, dest = rank_left, tag = mpi.rank)
        mpi.Isend(h_give_right, dest = rank_right, tag = mpi.rank)
        r_left.Wait()
        r_right.Wait()
        # copy ghost z direction ghost zones from host to device
        cuda.memcpy_htod(d_left, h_recv_left)
        cuda.memcpy_htod(d_right, h_recv_right)
#        debug('6')
        # copy ghost z direction ghost zones from linear memory on device
        kernel_ghost_copy_inv(d_domain_p, d_left, d_right, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (256, 1, 1), grid = (1, 1, 1))
#        debug('7')
        # copy d_domain_p to d_domain
        cuda.memcpy_dtod(d_domain, d_domain_p, h_domain.nbytes)
#        debug('8')
        # kernel_source(d_domain, np.int32(dimx), np.int32(dimy), np.int32(dimz), block = (1,1,1), grid = (1,1))
    cuda.Context.pop()
else: # master
    iters = 0
    mpi.Barrier()
    while iters < iterations:
        mpi.Barrier()
        if iters % savefreq == 0:
            rs = [mpi.Irecv(h_domains[i], source = i, tag = 0x5a43) for i in range(mpi.size - 1)]
            [r.Wait() for r in rs]
            save = np.hstack(h_domains).transpose()
            save.tofile('out_%.8d_%dx%d.bin' % (iters, save.shape[0], save.shape[1]))
            mpi.Barrier()
        iters += 1

mpi.Barrier()

pprint('finished %d iterations' % iterations)
