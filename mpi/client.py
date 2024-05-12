#! /usr/bin/env python
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
print("rank:"+str(rank))

def log(msg, *args):
    if rank == 0:
        print (msg % args)

info = MPI.INFO_NULL
service = "hpc"
log("looking-up service '%s'", service)
port = MPI.Lookup_name(service)
log("service located  at port '%s'", port)

root = 0
log('waiting for server connection...')
comm = MPI.COMM_WORLD.Connect(port, info, root)
log('server connected...')

while True:
    done = False
    if rank == root:
        try:
            message = input('pyeval (quit to exit)>>> ')
            if message == 'quit':
                message = None
                
                done = True
        except EOFError:
            message = None
            done = True
        comm.send(message, dest=0, tag=0)
        message = comm.recv(source=0, tag=0)
        log('received:%s' %message)
    else:
        message = None
    done = MPI.COMM_WORLD.bcast(done, root)
    if done:
        break

log('disconnecting server...')
comm.Disconnect()