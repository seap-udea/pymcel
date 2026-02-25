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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    version='0.9.8',

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
	              'matplotlib','tqdm','pandas','plotly','scipy'],

    # ######################################################################
    # OPTIONS
    # ######################################################################
    include_package_data=True,
    package_data={"": ["data/*"]},
)
