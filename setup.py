"""
Copyright 2023 by Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '2023.12.11.0'

requirements = [
    "numpy>=1.23.4",
    "opencv-python>=4.8.1.78",
    "torch>=2.0.0",
    "torchvision>=0.16.1",
    "huggingface-hub>=0.17.3"
]

setup(
    # Metadata
    name='image_to_sticker',
    version=VERSION,
    author='Sergei Belousov aka BeS',
    author_email='sergei.o.belousov@gmail.com',
    description='ImageToSticker: convert your image into a sticker',
    long_description=readme,
    long_description_content_type='text/markdown',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
