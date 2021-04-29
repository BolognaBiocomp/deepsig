
from os import path
from setuptools import setup, find_packages
#import deepsig


# Get the long description from the README file
setup_dir = path.abspath(path.dirname(__file__))
with open(path.join(setup_dir, 'README.pypi.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='deepsig-biocomp',
    version='0.9',
    description='DeepSig - Predictor of signal peptides in proteins based on deep learning',
    keywords=['bioinformatics', 'annotation', 'bacteria', 'signal peptides'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    author='Castrense Savojardo',
    author_email='savojard@biocomp.unibo.it',
    url='https://github.com/BolognaBiocomp/deepsig',
    packages=find_packages(include=['deepsig']),
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'biopython >= 1.78',
        'Keras >= 2.4.3',
        'tensorflow'
    ],
    entry_points={
        'console_scripts': [
            'deepsig=deepsig.deepsig:main'
        ]
    },
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Development Status :: 4 - Beta ',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/BolognaBiocomp/deepsig/issues',
        'Source': 'https://github.com/BolognaBiocomp/deepsig'
    },
)
