
# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


import os.path

readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.rst')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')


setup(
    long_description=readme,
    name='dial-visualization',
    version='0.2a0',
    description='Visualization techniques for CNN.',
    python_requires='<=3.8.3,>=3.6.0',
    project_urls={"homepage": "https://github.com/JDM-ULL-93/Dial-Visualization", "repository": "https://github.com/JDM-ULL-93/Dial-Visualization"},
    author='Javier Duque Melguizo',
    author_email='javierduquemelguizo@gmail.com',
    license='GPL-3.0-only',
    keywords='deep-learning dial-app',
    packages=[
        'dial_visualization',
        'dial_visualization.conv_visualization',
        'dial_visualization.conv_visualization.delegate_tree',
        'dial_visualization.conv_visualization.extended_widgets',
        'dial_visualization.conv_visualization.model_table',
        'dial_visualization.conv_visualization.model_tree',
        'dial_visualization.conv_visualization.Utils',
        'dial_visualization.conv_visualization.Utils.Algorithms',
        'dial_visualization.preprocessor_loader'
        ],
    package_dir={"": "."},
    package_data={},
    install_requires=[
        'dial-core',
        'dial-gui',
        "PySide2==5.*,>=5.12.6",
        "tensorflow==2.*,>=2.4.0",
        "Pillow==7.*,>=7.2.0"
    ],
    #dependency_links=['/home/davafons/dial-core'],
    #dependency_links=["C:/Users/DuquePC/source/repos/Practicas-TFG/dial-core"],
    dependency_links=["../dial-core","../dial-gui"]
)
