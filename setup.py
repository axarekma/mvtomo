from setuptools import setup
import setuptools

__version__ = "0.0.1"


setup(
    name="mvtomo",
    version=__version__,
    author="Axel Ekman",
    author_email="Axel.Ekman@iki.fi",
    url="https://github.com/axarekma/mvtomo",
    description="Wrapper for some algorithms for tomosipo",
    long_description="NA",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
