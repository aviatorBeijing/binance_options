from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "gamma_scalping",
        ["gamma_scalping.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="gamma_scalping",
    ext_modules=ext_modules,
)

