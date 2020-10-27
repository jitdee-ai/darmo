from setuptools import setup

exec(open('darmo/version.py').read())

setup(
    name='darmo',
    version=__version__,
    author='Chakkrit Termritthikun',
    author_email='chakkritt60@nu.ac.th',
    packages=['darmo'],
    url='https://github.com/jitdee-ai/darts-models',
    description='darts model pre-trained',
    install_requires=['torch >= 1.0', 'torchvision'],
    python_requires='>=3.6',
)