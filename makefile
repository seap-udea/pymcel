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
BRANCH=$(shell bash bin/getbranch.sh)
VERSION=$(shell tail -n 1 .versions)
COMMIT=[MAN] Maintainance
RELMODE=release
PYTHON=python3
PIP=pip3
PACKNAME=pymcel

show:
	@echo "Versión: $(VERSION)"
	@echo "Rama de github: $(BRANCH)"

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
	@-find . -name ".ipynb_checkpoints" -type d | xargs rm -fr
	@-find . -name "__pycache__" -type d | xargs rm -fr

cleandist:
	@-rm -rf dist/
	@-rm -rf build/
	@-rm -rf $(PACKNAME)-*/
	
cleandata:
	@echo "Cleaning all downloaded kernels..."
	@rm -rf src/$(PACKNAME)/data/[a-z]*.*

##################################################################
#GIT
##################################################################
addall:cleanall
	@echo "Adding..."
	@-git add -A .

commit:
	@echo "Commiting..."
	@git commit -am "$(COMMIT)"
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
	@bash bin/release.sh $(RELMODE) $(VERSION)

install:
	@sudo $(PIP) install -e .

import:
	@$(PYTHON) -c "from pymcel import *;print(version)"
