#!/usr/bin/env python

from setuptools import setup, find_packages

# this library version number = torchvision version number
version = '0.4.0'

setup(name='Invertible Transformations',
      version=version,
      description='Torchvision transformations can now be reversed',
      author='Gregoire Clement',
      author_email='mail@gregunz.io',
      url='github.com/gregunz',
      packages=find_packages(),
      requires=[
          'torch',
          f'torchvision=={version}',
      ])
