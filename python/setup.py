#!/usr/bin/python3
from setuptools import setup
import subprocess

try:
    from Cython.Build import cythonize
    from Cython.Distutils import Extension
    from Cython.Distutils import build_ext  
    import numpy
except:
    print('ERROR: Setup requires Cython and Numpy.')
    raise

# read the version of the package
with open('craam/version.py') as f:
    code = compile(f.read(), "raam/version.py", 'exec')
    exec(code, globals(), locals())

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

ext_modules = [
    Extension(
        "craam.crobust",
        ["craam/crobust.pyx"],
        extra_compile_args = ['-std=c++14','-fopenmp','-O3','-march=native'],
        extra_link_args=['-fopenmp'],
        include_dirs = [numpy.get_include(), '../']),
    ]

setup(
    name='craam',
    version=version,
    author='Marek Petrik',
    author_email='marekpetrik@gmail.com',
    packages=['craam'],
    scripts=[],
    url='http://www.github.com/marekpetrik/craam',
    license='LICENSE',
    description='Algorithms for solving robust and approximate and plain Markov decision processes',
    install_requires=[
    ],
    cmdclass = {'build_ext' : build_ext},
    ext_modules = cythonize(ext_modules),    
)

