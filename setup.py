from setuptools import setup

exec(open('darmo/version.py').read())

setup(
    name='darmo',
    version=__version__,
    author='Chakkrit Termritthikun',
    author_email='chakkritt60@nu.ac.th',
    packages=[  'darmo', 
                'darmo.layers.tresnetv1',
            ],
    url='https://github.com/jitdee-ai/darmo',
    description='darts model pre-trained',
    install_requires=['torch >= 1.0', 'torchvision', 'ofa==0.0.4.post2007200808', 'timm==0.4.12'],
    python_requires='>=3.6',
    package_data={
        'darmo': ['config/*.*'],
    },
)

