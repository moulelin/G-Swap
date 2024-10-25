from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='swap_kernel_linear_cpp',
    ext_modules=[
        CUDAExtension('swap_kernel', [
            'swap_kernel.cpp',
            'swap_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

