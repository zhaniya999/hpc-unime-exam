import pyopencl as cl
import numpy as np
import time
import os

def current_milli_time():
    return round(time.time() * 1000)

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#os.environ['PYOPENCL_CTX'] = '0'

(n, m, p) = (1100, 1100, 3000)

# a = np.random.randn(n, m).astype(np.float32)
# b = np.random.randn(m, p).astype(np.float32)
print("Inizializzo A",n,"x",m)
a = np.random.randint(2, size=(n*m))
print("Inizializzo B",m,"x",p)
b = np.random.randint(2, size=(m*p))
print("Inizializzo C",n,"x",p)
c = np.zeros((n*p), dtype=np.float64)

a = a.astype(np.float64)
b = b.astype(np.float64)

#ctx = cl.create_some_context()
platforms = cl.get_platforms()
dev = platforms[0].get_devices(device_type=cl.device_type.GPU)
if(len(dev)==0):
    dev = dev = platforms[0].get_devices(device_type=cl.device_type.CPU)
    print("GPU non presente, elaboro su CPU")
else:
    print("Elaboro su GPU")

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
print("t:",str(current_milli_time()-t0),"ms")
print ("matrix A:")
print (a.reshape(n, m))
print ("matrix B:")
print (b.reshape(m, p))
print ("multiplied A*B:")
print (a_mul_b.reshape(n, p))
