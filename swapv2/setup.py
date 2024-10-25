import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')
if "main.cpp" in sources:
    print("remove main.cpp in case of duplication operation")
    sources.remove("main.cpp")


setup(
    name='gswapv2',
    version='2.0',
    author='moule',
    author_email='moulecs@gmail.com',
    description='swap',
    long_description='swap',
    ext_modules=[
        CUDAExtension(
            name='gswapv2',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)