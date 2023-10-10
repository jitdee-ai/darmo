from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('darmo/version.py').read())

setup(
    name='darmo',
    version=__version__,
    author='Chakkrit Termritthikun',
    author_email='chakkritt@nu.ac.th',
    packages=[  'darmo', 
                'darmo.models',
                'darmo.models.gcn_lib',
                'darmo.layers.tresnetv1',
            ],
    url='https://github.com/jitdee-ai/darmo',
    description='darts model pre-trained',
    install_requires=['torch >= 1.0', 'torchvision', 'ofa==0.0.4.post2007200808', 'timm>=0.9.1', 'filelock'],
    include_package_data=True,
    python_requires='>=3.7',
    package_data={
        'darmo': ['config/*.*'],
    },
)

