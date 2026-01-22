
import ctypes
import os
import sys

# Load shared library
if sys.platform == "win32":
    lib_path = os.path.join(os.path.dirname(__file__), "../build/cogito.dll")
elif sys.platform == "darwin":
    lib_path = os.path.join(os.path.dirname(__file__), "../build/libcogito.dylib")
else:
    lib_path = os.path.join(os.path.dirname(__file__), "../build/libcogito.so")

# Helper to find library
if not os.path.exists(lib_path):
    # Try finding in current dir (if installed)
    lib_path = "libcogito.so"

try:
    _cg = ctypes.CDLL(lib_path)
except OSError:
    print(f"Warning: Could not load Cogito library from {lib_path}")
    _cg = None

# Types
class CgTensor(ctypes.Structure):
    pass

CgTensor_p = ctypes.POINTER(CgTensor)

if _cg:
    # Function IDs
    _cg.cg_tensor_new.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool]
    _cg.cg_tensor_new.restype = CgTensor_p
    
    _cg.cg_tensor_print.argtypes = [CgTensor_p, ctypes.c_char_p]
    _cg.cg_tensor_free.argtypes = [CgTensor_p]

class Tensor:
    def __init__(self, shape, requires_grad=False, _ptr=None):
        if _ptr:
            self._ptr = _ptr
        else:
            if not _cg: raise RuntimeError("Cogito lib not loaded")
            c_shape = (ctypes.c_int * len(shape))(*shape)
            self._ptr = _cg.cg_tensor_new(c_shape, len(shape), requires_grad)
            if not self._ptr:
                raise MemoryError("Failed to allocate tensor")
    
    def __del__(self):
        if self._ptr and _cg:
            _cg.cg_tensor_free(self._ptr)
            
    def print(self, name="tensor"):
        if _cg:
            _cg.cg_tensor_print(self._ptr, name.encode('utf-8'))
    
    @staticmethod
    def from_ptr(ptr):
        return Tensor([], _ptr=ptr)

# Shape Inference Helper
def infer_matmul_shape(shape_a, shape_b):
    if len(shape_a) < 2 or len(shape_b) < 2:
        raise ValueError("Matmul requires at least 2 dimensions")
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"Shape mismatch: {shape_a} vs {shape_b}")
    return shape_a[:-1] + shape_b[-1:]
