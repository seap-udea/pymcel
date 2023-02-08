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

##################################################################
#VARIABLES
##################################################################
SHELL:=/bin/bash

#Package information
include .packrc

#Github specifics
BRANCH=$(shell bash .getbranch.sh)
VERSION=$(shell tail -n 1 .versions)

show:
	@echo "Versión: $(VERSION)"
	@echo "Rama de github: $(BRANCH)"

##################################################################
#BRANCHING
##################################################################
branch:
	git checkout -b dev

make rmbranch:
	git branch -d dev

devbranch:
	git checkout dev

master:
	git checkout master

##################################################################
#BASIC RULES
##################################################################
clean:cleancrap

cleanall:cleancrap cleanout cleandist cleandata

#=========================
#Clean
#=========================
cleancrap:
	@echo "Cleaning crap..."
	@-find . -name "*~" -delete
	@-find . -name "#*#" -delete
	@-find . -name "#*" -delete
	@-find . -name ".#*" -delete
	@-find . -name ".#*#" -delete
	@-find . -name ".DS_Store" -delete
	@-find . -name "Icon*" -delete
	@-find . -name "*.egg-info*" -type d | xargs rm -fr

cleanout:
	@echo "Cleaning all compiled objects..."
	@-find . -name "*.o" -delete
	@-find . -name "*.opp" -delete
	@-find . -name "*.gcno" -delete
	@-find . -name "*.gcda" -delete
	@-find . -name "*.gcov" -delete
	@-find . -name "*.info" -delete
	@-find . -name "*.out" -delete
	@-find . -name "*.tout" -delete
	@-find . -name "*.so" -delete
	@-find . -name '__pycache__' -type d | xargs rm -fr

cleandist:
	@-rm -rf dist/
	@-rm -rf build/

cleandata:
	@echo "Cleaning all downloaded kernels..."
	rm -rf src/$(PACKNAME)/data/[a-z]*.*

##################################################################
#GIT
##################################################################
addall:cleanall
	@echo "Adding..."
	@-git add -A .

commit:
	@echo "Commiting..."
	@-git commit -am "Commit"
	@-git push origin $(BRANCH)

pull:
	@echo "Pulling new files..."
	@-git reset --hard HEAD
	@-git pull origin $(BRANCH)

##################################################################
#PACKAGE RULES
##################################################################
#Example: make release RELMODE=release VERSION=0.2.0.2 
release:
	@echo "Releasing a new version..."
	@bash .release.sh $(RELMODE) $(VERSION)

install:
	@$(PIP) install -e .

import:
	@$(PYTHON) -c "from pymcel import *;print(version)"
