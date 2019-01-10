# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='programmingalpha',
    version='1.0',
    description='Answering the QA posts questions via machine reading comprehension',
    long_description=readme,
    license=license,
    python_requires='>=3.6',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)
