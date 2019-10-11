#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='Invertible Transformations',
      version=f'1.0.0',
      description='Torchvision transformations can now be reversed',
      author='Gregoire Clement',
      author_email='mail@gregunz.io',
      url='github.com/gregunz',
      packages=find_packages(),
      requires=[
          'torch',
          f'torchvision==0.4.0',
      ])
