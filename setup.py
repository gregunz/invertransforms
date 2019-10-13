#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='invertransforms',
    version='0.1.2',
    license='BSD 3-Clause',
    description='A library which turns torchvision transformations invertible and replayable.',
    packages=find_packages(),
    keywords=['invertible', 'transforms', 'transformations', 'torchvision', 'data', 'augmentation',
              'replay', 'replayable', 'invertransforms'],
    install_requires=[
        'torch>=1.2.0',
        'torchvision>=0.4.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite="tests",
)
