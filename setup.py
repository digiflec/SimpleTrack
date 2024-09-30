from setuptools import setup, find_packages

setup(
    name='mot_3d',
    version='0.1',
    description='Multi-object Tracking in 3D Space',
    author='Krish-Digiflec',
    author_email='krish@digiflec.com',
    url='https://github.com/digiflec/SimpleTrack.git',  # Replace with your repository URL
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
