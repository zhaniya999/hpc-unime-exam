from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = None
if rank == 0:
    sendbuf = np.empty([size, 100], dtype='i')
    sendbuf.T[:,:] = range(size)
    print("r:"+str(rank)+" s:"+str(sendbuf))
recvbuf = np.empty(100, dtype='i')
print("r:"+str(rank)+" r:"+str(recvbuf))
comm.Scatter(sendbuf, recvbuf, root=0)
assert np.allclose(recvbuf, rank)