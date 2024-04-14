# hpc-unime-exam

Configure a venv with:
* python -m venv /hpc-mpi
Activate with:
* /hpc-mpi/Scripts/Activate.ps1 (Windows with powershell)
* /hpc-mpi/Scripts/activate.bat (Windows with cmd)
* /hpc-mpi/bin/activate (Linux)

* pip install pyopencl
* pip install pyopencl[pocl]
* pip install pandas
* pip install mpi4py (must be installed python3-devel, libopenmpi-devel to be compiled)
* https://www.hackerearth.com/practice/notes/raspberry-pi-hacks-part-1-building-mpi-for-python-on-a-raspberry-pi-cluster/
* https://nyu-cds.github.io/python-mpi/setup/

In some environment like LINUX could be necessary to substitute the pip command with pip3

WINDOWS

* remove, if present, any OpenMPI, Microsoft MPI
* run python.exe -m pip uninstall mpi4py
* clear pip cache with python -m pip cache remove *
* from https://learn.microsoft.com/it-it/message-passing-interface/microsoft-mpi#ms-mpi-downloads download latest version of Microsoft MPI and install it
* install mpi4py with python.exe -m pip install mpi4py

Test opencl with:

WINDOWS

* mpiexec.exe -n 1 python.exe .\test-mpi-matrix-product.py

LINUX

* mpiexec -n 1 python3 ./test-mpi-matrix-product.py
