import pyopencl as cl
import numpy as np
import time
import os
from mpi4py import MPI
import util as ut
import json
import argparse
from datatypes import DataMessage, ResponseMessage


parser = argparse.ArgumentParser(description='hpc exam calculator - MPI Test')
parser.add_argument('-d',help='Enable debug messages')
args = parser.parse_args()
debug = False

if args.d is None:
    debug = False
else:
    debug = True


info = MPI.INFO_NULL
service = "hpc"
ut.log("looking-up service '%s'", service)
port = MPI.Lookup_name(service)
ut.log("service located  at port '%s'", port)
root = 0
ut.log('waiting for data ')
comm = MPI.COMM_WORLD.Connect(port, info, root)

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
out = ResponseMessage(result=None,type=None,time=0,row=None)

while True:
    #comm.send("", dest=0, tag=0)
    message = comm.recv(source=0, tag=0).strip()
    ut.log("Received %s",message)
    messageObj = json.loads(message)
    out.setRow(messageObj["row"])

    a = np.array(messageObj["a"])
    b = np.array(messageObj["b"])

    #the first matrix is always a row vector of size 1xn
    #the second matrix have n rows, so the columns are calulate dividing the readed array lenght by the rows
    #the results is a row vector with the same number of columns of b matrix

    n = 1
    m = int(len(a))
    p = int(len(b)/m)

    c = np.zeros(p, dtype=np.float64)

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    platforms = cl.get_platforms()
    #out["platform"]=platforms[0].name
    dev = platforms[0].get_devices(device_type=cl.device_type.GPU)
    if(len(dev)==0):
        dev = platforms[0].get_devices(device_type=cl.device_type.CPU)
        if debug:
            ut.log("GPU is absent, using CPU")
        out.setType("C")
    else:
        if debug:
            ut.log("using GPU")
        out.setType("G")
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
    prg.multiply(queue, c.shape, None, np.uint16(n), np.uint16(m), np.uint16(p), a_buf, b_buf, c_buf)
    a_mul_b = np.empty_like(c)
    t0 = ut.current_milli_time()
    cl.enqueue_copy(queue, a_mul_b, c_buf)
    out.setTime(ut.current_milli_time()-t0)
    out.setResult(a_mul_b.tolist())
    print(out.getResult())
    comm.send(out.toJSON(), dest=0, tag=0)
    if debug:
        print("t:",str(out.getTime()),"ms")
        print ("matrix A:")
        print (a.reshape(n, m))
        print ("matrix B:")
        print (b.reshape(m, p))
        print ("multiplied A*B:")
        print (a_mul_b.reshape(n, p))
    print(out.toJSON())