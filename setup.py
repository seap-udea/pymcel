##################################################################
#                                                                #
# ██████  ██    ██ ███    ███  ██████ ███████ ██                 #
# ██   ██  ██  ██  ████  ████ ██      ██      ██                 #
# ██████    ████   ██ ████ ██ ██      █████   ██                 #
# ██         ██    ██  ██  ██ ██      ██      ██                 #
# ██         ██    ██      ██  ██████ ███████ ███████            #
#                                                                #
# Utilidades de Mecáncica Celeste                                #
#                                                                #
##################################################################
# Licencia http://github.com/seap-udea/pymcel                    #
##################################################################
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # ######################################################################
    # BASIC DESCRIPTION
    # ######################################################################
    name='pymcel',
    author="Jorge I. Zuluaga",
    author_email="jorge.zuluaga@udea.edu.co",
    description="Utilidades de Mecánica Celeste",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/pymcel",
    keywords='astronomy astrodynamics',
    license='MIT',

    # ######################################################################
    # CLASSIFIER
    # ######################################################################
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    version='0.6.1',

    # ######################################################################
    # FILES
    # ######################################################################
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    
    # ######################################################################
    # ENTRY POINTS
    # ######################################################################
    entry_points={
        'console_scripts': ['install=pymcel.install:main'],
    },

    # ######################################################################
    # TESTS
    # ######################################################################
    test_suite='nose.collector',
    tests_require=['nose'],

    # ######################################################################
    # DEPENDENCIES
    # ######################################################################
    install_requires=['spiceypy','astroquery','pandas',
	              'matplotlib','tqdm','pandas'],

    # ######################################################################
    # OPTIONS
    # ######################################################################
    include_package_data=True,
    package_data={"": ["data/*"]},
)
