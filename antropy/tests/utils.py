import numpy as np

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
NORMAL_TS = np.random.normal(size=3000)
RANDOM_TS_LONG = np.random.rand(6000)
PURE_SINE = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
PURE_COSINE = np.cos(2 * np.pi * 1 * np.arange(3000) / 100)
ARANGE = np.arange(3000)

# Data types for which to test input array compatibility
TEST_DTYPES = [
    # floats
    np.float16, np.float32, np.float64, np.int8,
    # ints
    np.int16, np.int32, np.int64,
    # unsigned ints
    np.uint8, np.uint16, np.uint32, np.uint64,
]
