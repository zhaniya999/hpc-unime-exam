import pyopencl as cl
import numpy as np
import time
import os
import mpi4py
import argparse
from mpi4py import MPI
import pandas as pd


def current_milli_time():
    return round(time.time() * 1000)

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#os.environ['PYOPENCL_CTX'] = '0'

(n, m, p) = (1100, 1100, 3000)
out = {"e":"","t":0,"platform":"","size":0,"rank":0}
debug = True

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
out["size"]=size
out["rank"]=rank

# a = np.random.randn(n, m).astype(np.float32)
# b = np.random.randn(m, p).astype(np.float32)
if debug:
    print(rank,"/",size," Init A: ",n,"x",m)
a = np.random.randint(2, size=(n*m))
if debug:
    print(rank,"/",size," Init B: ",m,"x",p)
b = np.random.randint(2, size=(m*p))
if debug:
    print(rank,"/",size," Init C: ",n,"x",p)
c = np.zeros((n*p), dtype=np.float64)

a = a.astype(np.float64)
b = b.astype(np.float64)

#ctx = cl.create_some_context()
platforms = cl.get_platforms()
out["platform"]=platforms[0].name
dev = platforms[0].get_devices(device_type=cl.device_type.GPU)
if(len(dev)==0):
    dev = platforms[0].get_devices(device_type=cl.device_type.CPU)
    if debug:
        print(rank,"/",size," the GPU is absent, the CPU is used")
    out["e"]="C"
else:
    if debug:
        print(rank,"/",size," Working with GPU")
    out["e"]="G"
ctx = cl.Context(devices=dev)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

prg = cl.Program(ctx, """
    __kernel void multiply(ushort n,
    ushort m, ushort p, __global float *a,
    __global float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = 0.0f;
      int rowC = gid/p;
      int colC = gid%p;
      __global float *pA = &a[rowC*m];
      __global float *pB = &b[colC];
      for(int k=0; k<m; k++)
      {
         pB = &b[colC+k*p];
         c[gid] += (*(pA++))*(*pB);
      }
    }
    """).build()
prg.multiply(queue, c.shape, None,
             np.uint16(n), np.uint16(m), np.uint16(p),
             a_buf, b_buf, c_buf)
a_mul_b = np.empty_like(c)
t0 = current_milli_time()
cl.enqueue_copy(queue, a_mul_b, c_buf)
out["t"]=current_milli_time()-t0
if debug:
    print("t:",str(out["t"]),"ms")
    print (rank,"/",size," matrix A:")
    print (a.reshape(n, m))
    print (rank,"/",size," matrix B:")
    print (b.reshape(m, p))
    print (rank,"/",size," multiplied A*B:")
    print (a_mul_b.reshape(n, p))
print(out)