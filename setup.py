from setuptools import setup, find_packages

setup(
    name='Golem',
    version='0.1',
    description='A fun playground for Bayesian inference.',
    url='https://github.com/leoentersthevoid/golem',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'networkx',
        'matplotlib'
    ]
)


