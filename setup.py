from setuptools import setup

setup(
    name='openai_pygenerator',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    py_modules=['openai_pygenerator'],
    package_dir={'': 'src'},
)