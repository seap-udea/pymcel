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
#!/bin/bash
PYTHON=python3
PACKNAME=pymcel

type=$1;shift
if [ "x$type" = "x" ]
then
    echo "You need to choose a type of release: 'test', 'release'."
    exit 1
elif [ "$type" = "test" ]
then
    qtype=0
elif [ "$type" = "release" ]
then
    qtype=1
else
    echo "Type '$type' not recognized (it should be 'test' or 'release'"
    exit 1
fi

version=$1
setversion=$(grep "version=" setup.py |awk -F"'" '{print $2}')

##################################################################
# Latest version
##################################################################
if [ "x$version" = "x" ]
then
    version=$(tail -n 1 .versions)
    echo "Latest version: $version"
    echo "Version in setup file: $setversion"
    exit 1
fi

if [ "$version" = "$setversion" ]
then
    echo "Version provided ($version) coincide with version in setup.py file ($setversion). It must be different."
    exit 1
fi

echo "Releasing new version $version (current version $setversion) of the package in mode '$type'..."

##################################################################
# Update setup.py file
##################################################################
sed -i.bak "s/version=\'[0-9\.]*\'/version='$version'/gi" setup.py 
sed -i.bak "s/version='[0-9\.]*'/version='$version'/gi" setup.py 

##################################################################
# Remove previous versions
##################################################################
echo "Removing previous version..."
rm -rf dist/*

##################################################################
# Report version
##################################################################
echo $version >> .versions
cp src/$PACKNAME/version.py tmp/version.py.bak
echo "version='$version'" > src/$PACKNAME/version.py

##################################################################
# Build package
##################################################################
echo "Building packages..."
$PYTHON -m build

##################################################################
# Uploading the package
##################################################################
echo
if [ $qtype -eq 0 ]
then
    echo "Uploading to Test PyPI (use __token__ as username and pypi-<token> as password)..."
    $PYTHON -m twine upload --repository testpypi dist/* 
else
    echo "Uploading to PyPI (use your username and password)..."
    $PYTHON -m twine upload dist/* 
fi
