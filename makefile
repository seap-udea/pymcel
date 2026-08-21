##################################################################
#                                                                #
# ██████  ██    ██ ███    ███  ██████ ███████ ██                 #
# ██   ██  ██  ██  ████  ████ ██      ██      ██                 #
# ██████    ████   ██ ████ ██ ██      █████   ██                 #
# ██         ██    ██  ██  ██ ██      ██      ██                 #
# ██         ██    ██      ██  ██████ ███████ ███████            #
#                                                                #
# Utilidades de Mecánica Celeste                                 #
#                                                                #
##################################################################
# Licencia http://github.com/seap-udea/pymcel                    #
##################################################################

.PHONY: help show clean cleanall cleancrap cleanout cleandist cleandata \
	install install-dev build docs env addall commit pull push release import

##################################################################
# VARIABLES
##################################################################
SHELL := /bin/bash
BRANCH := $(shell bash bin/getbranch.sh)
VERSION := $(shell tail -n 1 .versions)
COMMIT_MSG ?= [MAN] Maintenance
RELMODE ?= release
PYTHON ?= python3
PIP ?= pip3
PACKNAME := pymcel

help:
	@echo "PyMCel Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  show        - Show current version and branch"
	@echo "  install     - Install the package"
	@echo "  install-dev - Install in development mode"
	@echo "  build       - Build distribution packages"
	@echo "  docs        - Build documentation (installs docs requirements)"
	@echo "  env         - Create local dev environment (.pymcel)"
	@echo "  clean       - Remove build artifacts and cache files"
	@echo "  cleanall    - Deep clean (build + caches + data)"
	@echo "  push        - Commit (all files) and push current branch"
	@echo "  release     - Release a new version (make release RELMODE=release VERSION=x.y.z)"

show:
	@echo "Versión: $(VERSION)"
	@echo "Rama de github: $(BRANCH)"

##################################################################
# BASIC RULES
##################################################################
clean: cleancrap

cleanall: _dev_cleanall cleancrap cleanout cleandist cleandata

#=========================
# Clean
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
# PACKAGE RULES
##################################################################
install:
	$(PYTHON) -m pip install .

install-dev:
	$(PYTHON) -m pip install -e .
	@if [ -f requirements.txt ]; then $(PYTHON) -m pip install -r requirements.txt; fi
	@if [ -f requirements-dev.txt ]; then $(PYTHON) -m pip install -r requirements-dev.txt; fi

env:
	@echo "Creating local development environment..."
	@test -d .pymcel || $(PYTHON) -m venv .pymcel
	@echo "Installing dependencies from setup.py..."
	@. .pymcel/bin/activate && pip install --upgrade pip
	@. .pymcel/bin/activate && pip install -e .
	@echo "______________________________________________________________________"
	@echo "Environment setup complete."
	@echo "To activate the environment, run:"
	@echo "source .pymcel/bin/activate"

build: clean
	$(PYTHON) -m build

docs:
	$(PYTHON) -m pip install -r docs/requirements.txt
	rm -rf docs/_build
	@chmod +x bin/prepare_docs.sh
	@./bin/prepare_docs.sh
	cd docs && $(PYTHON) -m sphinx.cmd.build -M html "." "_build"

##################################################################
# GIT
##################################################################
addall: cleanall
	@echo "Adding..."
	@-git add -A .

commit:
	@echo "Commiting..."
	@git commit -am "$(COMMIT_MSG)"
	@-git push origin $(BRANCH)

pull:
	@echo "Pulling new files..."
	@-git reset --hard HEAD
	@-git pull origin $(BRANCH)

push:
	@echo "Committing tracked changes (if any)..."
	@if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$$(git status --porcelain)" ]; then \
		git add . && \
		files="$$(git diff --cached --name-only | paste -sd', ' - || true)" && \
		msg="$(COMMIT_MSG)" && \
		if [ "$(origin COMMIT_MSG)" != "command line" ] && [ "$(origin COMMIT_MSG)" != "environment" ]; then \
			if [ -n "$$files" ]; then msg="$$msg [$$files]"; fi; \
		fi && \
		git commit -m "$$msg"; \
	else \
		echo "Working tree is clean (tracked files); nothing to commit."; \
	fi
	@echo "Pushing current branch..."
	@git push -u origin HEAD

##################################################################
# RELEASE
##################################################################
# Example: make release RELMODE=release VERSION=0.2.0.2
release:
	@echo "Releasing a new version..."
	@bash bin/release.sh $(RELMODE) $(VERSION)

import:
	@$(PYTHON) -c "from pymcel import *;print(version)"
# --- dev/cleanall (auto) ---
include .dev_common.mk
