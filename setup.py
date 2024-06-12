from setuptools import setup, find_packages

setup(
    name='BRAD',
    version='0.1.0',
    author='Joshua Pickard',
    author_email='jpic@umich.edu',
    description='A bioinformatics package for retrieval augmented data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Jpickard1/RAG-DEV',  # Replace with your package URL
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # List other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
