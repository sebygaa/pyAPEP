import setuptools

with open('README', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
        name='pyapep',
        version='0.1.5',
        author_email='sgasga@ulsan.ac.kr',
        description='Python package for adsorption process simulations',
        long_description_content_type='text/markdown',
        url='https://sebygaa.github.io/pyAPEP/',
        packages=setuptools.find_packages(),
        classifieres=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            ],
        python_requires='>=3.6',
        install_requires=[
            'numpy>=1.20.0',
            'scipy>=1.5.0',
            'matplotlib>=3.0.0',
            ],
        author='Seongbin Ga',
        )
print('Package path found here:')
print(setuptools.find_packages())