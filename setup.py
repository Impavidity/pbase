from setuptools import find_packages, setup


setup(
    name='pbase',
    version='0.0.1',
    author='Peng Shi',
    author_email='peng_shi@outlook.com',
    description='framework for deep learning applications',
    url='https://github.com/Impavidity/pbase',
    license='MIT',
    install_requires=['tqdm', 'numpy', 'collections', 'torch', 'fuzzywuzzy', 'nltk'],
    packages=find_packages(),
)