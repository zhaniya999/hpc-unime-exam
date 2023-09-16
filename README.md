# hpc-unime-exam

* pip install pyopencl
* pip install pyopencl[pocl]
* https://www.hackerearth.com/practice/notes/raspberry-pi-hacks-part-1-building-mpi-for-python-on-a-raspberry-pi-cluster/
* https://nyu-cds.github.io/python-mpi/setup/

WINDOWS
* remove, if present, any OpenMPI, Microsoft MPI
* run python.exe -m pip uninstall mpi4py
* clear pip cache with python -m pip cache remove *
* from https://learn.microsoft.com/it-it/message-passing-interface/microsoft-mpi#ms-mpi-downloads download latest version of Microsoft MPI and install it
* install mpi4py with python.exe -m pip install mpi4py