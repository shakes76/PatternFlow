from setuptools import setup

setup(
    name='StyleGAN',
    author='Zhien Zhang',
    author_email='zhien.zhang@uqconnect.edu.au',
    install_requires=[
        'tensorflow-gpu==2.6',
        'matplotlib',
        "neptune-client",
        "neptune-tensorflow-keras"
    ]
)
