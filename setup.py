from setuptools import setup, find_packages

with open("README.md") as f:
    README = f.read()

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Database",
]


# Read requirements from requirements.txt
with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()


PROJECT_URLS = {
    #"Bug Tracker": "https://github.com/salvnetto/pyreglin",
    #"Documentation": "https://salvnetto.github.io/pyreglin",
    "Source Code": "https://github.com/salvnetto/pyreglin",
}

setup(
    name='pyreglin',
    version='0.1.0',
    description= "A Python package for generating response variables with linear regression structure.",
    packages= find_packages(),
    long_description= README,
    long_description_content_type= "text/markdown",
    url= "https://github.com/salvnetto/pyreglin",
    author= "Fábio N. Demarqui",
    author_email= "fndemarqui@est.ufmg.br",
    maintainer= "Salvador Netto, Tomás Bernardes",
    maintainer_email= "salvv.netto@gmail.com",
    license= "MIT",
    platforms="any",
    classifiers= CLASSIFIERS,
    install_requires= REQUIREMENTS,
    zip_safe=False,
    python_requires='>3.8',
    project_urls=PROJECT_URLS,
)