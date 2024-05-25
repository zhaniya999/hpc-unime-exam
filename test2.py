import pyopencl as cl
import numpy as np

# Setup OpenCL context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Kernel code
kernel_code = """
__kernel void mat_vec_mult(__global const float* matrix,
                           __global const float* vector,
                           __global float* result,
                           const int rows,
                           const int cols) {
    int row = get_global_id(0);
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}
"""

# Compile the kernel
program = cl.Program(context, kernel_code).build()

# Create example data
rows, cols = 4, 4
matrix = np.random.rand(rows, cols).astype(np.float32)*10
vector = np.random.rand(cols).astype(np.float32)*10
result = np.zeros(rows, dtype=np.float32)

# Create buffers
matrix_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
vector_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector)
result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result.nbytes)

# Set kernel arguments and execute
kernel = program.mat_vec_mult
kernel.set_args(matrix_buf, vector_buf, result_buf, np.int32(rows), np.int32(cols))

# Execute the kernel
cl.enqueue_nd_range_kernel(queue, kernel, (rows,), None)

# Copy the result from the device to host
cl.enqueue_copy(queue, result, result_buf)

# Wait for all commands to complete
queue.finish()

# Print the results
print("Matrix:\n", matrix)
print("Vector:\n", vector)
print("Result:\n", result)
