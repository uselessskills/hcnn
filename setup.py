from setuptools import setup, find_packages

setup(
    version='0.0.1',

    name='hcnn',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch"
    ],
)
