import tensorflow as tf

tf.test.is_built_with_cuda()

tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)