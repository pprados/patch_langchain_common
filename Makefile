SHELL=/bin/bash
.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests
POETRY_EXTRA?=--all-extras
POETRY_WITH?=dev,lint,test,codespell
export PYTHONPATH=patch_partners/unstructured

# Default target executed when no arguments are given to make.
all: help

.PHONY: dump-*
# Tools to dump makefile variable (make dump-AWS_API_HOME)
dump-%:
	@if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* is not set"; \
		exit 1; \
	else \
		echo "$*=${${*}}"; \
	fi

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

integration_tests:
#	source .env && poetry run pytest \
#		tests/integration_tests
	source .env && poetry run pytest \
		patch_partners/unstructured/tests/integration_tests

test tests:
	poetry run pytest -v $(TEST_FILE)

test_watch:
	poetry run ptw --now . -- tests/unit_tests


# ---------------------------------------------------------------------------------------
.PHONY: dump-*
dump-%:
	@if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* is not set"; \
		exit 1; \
	else \
		echo "$*=${${*}}"; \
	fi \


#########################
# LINTING AND FORMATTING
#########################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/experimental --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
NB_FILES:=$(shell find docs/docs -name '*.ipynb')

lint lint_diff:
	poetry run mypy --exclude integration_tests $(PYTHON_FILES)
	poetry run black $(PYTHON_FILES) --check
	poetry run ruff .

lint_tests:
	poetry run mypy  tests
	poetry run black tests --check
	poetry run ruff tests

format format_diff:
	poetry run black $(PYTHON_FILES)
	poetry run ruff --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml --skip="compare_old_new,PPR*"

spell_fix:
	poetry run codespell --toml pyproject.toml -w


######################
# DOCUMENTATION
######################

clean:
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} \; || true
	@rm -Rf dist/ .make-* .mypy_cache .pytest_cache .ruff_cache

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'clean                        - run docs_clean and api_docs_clean'
	@echo 'spell_check               	- run codespell on the project'
	@echo 'spell_fix               		- run codespell on the project and fix the errors'


.PHONY: dist
dist:
	poetry build

# ---------------------------------------------------------------------------------------
# SNIPPET pour tester la publication d'une distribution
# sur test.pypi.org.
.PHONY: test-twine
## Publish distribution on test.pypi.org
test-twine: dist
ifeq ($(OFFLINE),True)
	@echo -e "$(red)Can not test-twine in offline mode$(normal)"
else
	@$(VALIDATE_VENV)
	rm -f dist/*.asc
	twine upload --sign --repository-url https://test.pypi.org/legacy/ \
		$(shell find dist -type f \( -name "*.whl" -or -name '*.gz' \) -and ! -iname "*dev*" )
endif

# ---------------------------------------------------------------------------------------
# SNIPPET pour publier la version sur pypi.org.
.PHONY: release
## Publish distribution on pypi.org
release: validate integration_tests clean dist
ifeq ($(OFFLINE),True)
	@echo -e "$(red)Can not release in offline mode$(normal)"
else
	@$(VALIDATE_VENV)
	[[ $$( find dist -name "*.dev*" | wc -l ) == 0 ]] || \
		( echo -e "$(red)Add a tag version in GIT before release$(normal)" \
		; exit 1 )
	rm -f dist/*.asc
	echo "Enter Pypi password"
	twine upload  \
		$(shell find dist -type f \( -name "*.whl" -or -name '*.gz' \) -and ! -iname "*dev*" )

endif

.venv poetry.lock: pyproject.toml
	poetry lock
	git add poetry.lock
	poetry install $(POETRY_EXTRA) --with $(POETRY_WITH)


## docs

.PHONY: docs
docs/api_reference: $(PYTHON_FILES)
	cd docs && sphinx-apidoc --remove-old -M -f -e -o _build/api_reference ..
	cd docs && sphinx-apidoc --remove-old -M -f -e -o _build/unstructured/api_reference ../patch_partners/unstructured

docs/nb: | $(NB_FILES)
	jupyter nbconvert --to markdown --output-dir=docs/_build/nb/ docs/docs/**/*.ipynb

docs: docs/api_reference docs/nb
	cd docs && cp -r conf.py _static index.rst _build && sphinx-build -a -E -b html _build _build/html
	xdg-open docs/_build/html/index.html

check_docs:
	python docs/scripts/check_templates.py docs/docs/integrations/document_loaders/*.ipynb

## Refresh lock
lock: .venv poetry.lock

## Validate the code
validate: poetry.lock format lint spell_check check_docs test integration_tests


init: poetry.lock
	@poetry self update
	@poetry self add poetry-dotenv-plugin
	@poetry self add poetry-plugin-export
	@poetry self add poetry-git-version-plugin
	@poetry config warnings.export false
	@poetry config virtualenvs.in-project true
	@poetry install --sync $(POETRY_EXTRA) --with $(POETRY_WITH)
	# FIXME @pre-commit install
	@git lfs install

# Push to langchain
LANGCHAIN_HOME=../langchain
TARGET:=community
SRC_PACKAGE=patch_langchain_community
DST_PACKAGE=langchain_community
SRC_MODULE:=patch_langchain_community
DST_MODULE:=community


define _push_sync
	@$(eval TARGET=$(TARGET))
	@$(eval SRC_PACKAGE=$(SRC_PACKAGE))
	@$(eval DST_PACKAGE=$(DST_PACKAGE))
	@$(eval WORK_DIR=$(shell mktemp -d --suffix ".rsync"))
	@mkdir -p "${WORK_DIR}/libs/${TARGET}"
	@mkdir -p "${WORK_DIR}/docs/docs"
	@echo Copy and patch $(SRC_PACKAGE) to $(DST_PACKAGE) in $(LANGCHAIN_HOME)
	@( \
		cd $(SRC_PACKAGE)/ ; \
		rsync -a \
		  --exclude ".*" \
		  --exclude __pycache__ \
		  --exclude __init__.py \
		  --exclude "new_*.py" \
		  . "${WORK_DIR}/libs/${TARGET}/$(DST_PACKAGE)" ; \
	)
	@( \
		cd tests/ ; \
		rsync -a \
		  --exclude ".*" \
		  --exclude __pycache__ \
		  --exclude __init__.py \
		  --exclude test_pdf.py \
		  --exclude "test_new_*.py" \
		  --exclude pdf-test-for-parsing.pdf \
		  . "${WORK_DIR}/libs/${TARGET}/tests" ; \
	)
	@( \
		cd docs/docs ; \
		rsync -a \
		  --exclude ".*" \
		  . "${WORK_DIR}/docs/docs" ; \
	)
	@find '${WORK_DIR}' -type f -a -name 'conftest.py' -exec rm {} ';'
	@find '${WORK_DIR}' -type f -a \
		-exec sed -i "s/${SRC_PACKAGE}/${DST_PACKAGE}/g" {} ';' \
		-exec sed -i "s/pip install -q '$(SRC_MODULE)'/pip install -q '$(DST_MODULE)'/g" {} ';'
	@cp -R "${WORK_DIR}/libs" "${WORK_DIR}/docs" $(LANGCHAIN_HOME)/
	@grep -rl '%pip install -qq \.\.\/\.\.\/\.\.\/\.\.\/dist\/patch_langchain_pdf_loader\*.whl' ../langchain/docs/docs --include \*.ipynb
	@find ../langchain/docs/docs -name '*.ipynb' -exec sed -i 's|%pip install -qq \.\.\/\.\.\/\.\.\/\.\.\/dist\/patch_langchain_pdf_loader\*.whl| |g' {} \;
	@rm -Rf '${WORK_DIR}'
	@echo done
endef

## Duplicate and patch files to ../langchain project
push-sync: format
	$(call _push_sync)

#pull-sync:
#	cp -rf $(TARGET)/langchain_experimental/chains/qa_with_references/ \
#		langchain_qa_with_references/chains/
#	cp -f $(TARGET)/langchain_experimental/chains/__init__.py \
#		langchain_qa_with_references/chains/
#	cp -rf $(TARGET)/langchain_experimental/chains/qa_with_references_and_verbatims/ \
#		langchain_qa_with_references/chains/
#	cp -rf $(TARGET)/tests/unit_tests/chains/ \
#		tests/unit_tests/
#	cp $(TARGET)/docs/qa_with_reference*.ipynb .
#	find . -type f \( -name '*.py' -or -name '*.ipynb' \) | xargs sed -i 's/langchain_experimental/langchain_qa_with_references/g'
#	find . -type f -name '*.ipynb' | xargs sed -i 's/langchain\([_-]\)experimental/langchain\1qa_with_references/g'

.PHONY: dist run_notebooks

run_notebooks: dist
	poetry run python parser_comparator/execute_notebooks.py
