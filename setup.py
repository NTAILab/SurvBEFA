from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='survbeta',
    version='0.1.0',
    description='A package for survival analysis with Beran estimator',
    author='Semen Khomets',
    author_email='khomets.semen@yandex.ru',
    packages=find_packages(),
    install_requires=requirements,
)
