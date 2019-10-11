#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='Invertible Transformations',
      version='1.0.0',
      description='A library which makes torchvision invertransforms invertible in a snap',
      author='Gregoire Clement',
      author_email='mail@gregunz.io',
      url='github.com/gregunz',
      packages=find_packages(),
      requires=[
          'torch',
          'torchvision>=0.4.0',
      ])
