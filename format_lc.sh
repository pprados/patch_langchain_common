set +x

LANGCHAIN_HOME=~/workspace.bda/langchain
TARGET=community
#cd ${LANGCHAIN_HOME}/libs/${TARGET}
#poetry make format
( cd ${LANGCHAIN_HOME}/libs/community ;
poetry run ruff check --select I --fix libs/community/langchain_community/document_loaders/parsers/pdf.py
poetry run ruff check --select I --fix libs/community/langchain_community/document_loaders/pdf.py
)
poetry run ruff check --select I --fix docs/docs/integrations/document_loaders/*pdf*.ipynb
#(  cd ${LANGCHAIN_HOME} ;
#  VIRTUAL_ENV= poetry run make lint_package
#)
#cd ${LANGCHAIN_HOME}/docs
# poetry run make build start

