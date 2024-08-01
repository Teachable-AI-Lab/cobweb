from glob import glob
from setuptools import setup

from pybind11.setup_helpers import intree_extensions
from pybind11.setup_helpers import build_ext

ext_modules = intree_extensions(glob('cobweb/*.cpp'))

# Specify the C++ standard for each extension module
for module in ext_modules:
    pass
#     module.cxx_std = '2a'
#     module.extra_link_args.append("-ltbb")
    #module.extra_compile_args.append("-g3")

setup(
    name="cobweb",
    author="Christopher J. MacLellan, Xin Lian, Nicki Barari, Erik Harpstead",
    author_email="maclellan.christopher@gmail.com, xinthelian@hotmail.com, nb895@drexel.edu, whitill29@gmail.com",
    url="https://github.com/Teachable-AI-Lab/cobweb",
    description="The library for revised Cobweb family,"
                "the incremental concept formation system and its variations.",
    long_description=open('README.md').read(),
    description_content_type="text/x-rst; charset=UTF-8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: PyPy"
    ],
    keywords="clustering,machine-learning",
    license="MIT",
    license_file="LICENSE.txt",
    packages=["cobweb"],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["pybind11", "torch", "torchvision", "numpy", "pandas", "scikit-learn", "tqdm", "matplotlib", "nltk", "spacy"],
)
