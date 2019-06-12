from distutils.core import setup

setup(
    name='textrl',
    author='Kolby Nottingham',
    packages=['textrl',],
    long_description=open('README.md').read(),
    install_requires=[
        "torch == 1.1.0",
        "textworld == 1.1.1"
    ],
)
