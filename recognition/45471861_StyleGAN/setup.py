from setuptools import setup

setup(
    name='StyleGAN',
    author='Zhien Zhang',
    author_email='zhien.zhang@uqconnect.edu.au',
    install_requires=[
        'tensorflow',
        'matplotlib',
        "neptune-client",
        "neptune-tensorflow-keras"
    ]
)
