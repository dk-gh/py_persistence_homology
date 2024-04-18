#!/usr/bin/env python
# -*- coding: utf-8 -*-

# setup.py template copied from
# https://github.com/navdeep-G/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command, Extension

from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = False

# Package meta-data.
NAME = 'persistence_homology'
DESCRIPTION = 'Computes persistence homology of Vietoris-Rips complex.'
URL = 'https://github.com/dk-gh/persistence_homology'
EMAIL = 'kevin.dunne@mailbox.org'
AUTHOR = 'Kevin Dunne'
REQUIRES_PYTHON = '>=3.10.0'
VERSION = None # is read from __version__.py

REQUIRED = [
    'pandas',
    'numpy',
    'matplotlib',
    'scipy',
]

# What packages are optional?
EXTRAS = {
   'tests': ['scikit-learn'],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}

if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel'.format(sys.executable))

#        self.status('Uploading the package to PyPI via Twine…')
#        os.system('twine upload dist/*')

 #       self.status('Pushing git tags…')
 #       os.system('git tag v{0}'.format(about['__version__']))
 #       os.system('git push --tags')

        sys.exit()


extensions = [
    Extension(
        name='persistence_homology.sparse_matrix.cy_src_sparse_matrix',
        sources=['persistence_homology/sparse_matrix/cy_src_sparse_matrix.c'],
        include_dirs=['persistence_homology/sparse_matrix']
    )
]


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    ext_modules=cythonize(extensions,
        annotate=False,
        language_level=3
    ),
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={'persistence_homology.sparse_matrix':['cy_src_sparse_matrix.c']},
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    setup_requires=['cython'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
