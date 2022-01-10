import setuptools
from os import path
import dampn

here = path.abspath(path.dirname(__file__))
AUTHORS = """
Evan Komp
"""


# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='dampn',
        version=dampn.__version__,
        author=AUTHORS,
        project_urls={
            'Source': 'https://github.com/EvanKomp/DAMPN',
        },
        description=
        'Distance aware message passing neural net',
        long_description=long_description,
        include_package_data=False, #no data yet, True if we want to include data
        keywords=[
            'Machine Learning', 'Ab initio',
            'Chemical Engineering','Chemistry', 
        ],
        license='MIT',
        packages=setuptools.find_packages(exclude="tests"),
        scripts = [], #if we want to include shell scripts we make in the install
        install_requires=[
            'numpy', 
            'pandas', 
            'ase',
        ],
        extras_require={
            'tests': [
                'pytest',
                'coverage',
                'flake8',
                'flake8-docstrings'
            ],
            'docs': [
                'sphinx',
                'sphinx_rtd_theme',

            ]
        },
        classifiers=[
            'Development Status :: 1 - Planning',
            'Environment :: Console',
            'Operating System :: OS Independant',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
        ],
        zip_safe=False,
    )
