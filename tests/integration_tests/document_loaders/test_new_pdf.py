from pathlib import Path

from patch_langchain_community.document_loaders.new_pdf import (
    LlamaIndexPDFLoader,
    PyMuPDF4LLMLoader,
)


def test_pymupdf4llm_loader() -> None:
    """Test PyMuPDF4llm loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PyMuPDF4LLMLoader(file_path)
    docs = loader.load()
    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyMuPDF4LLMLoader(file_path, mode="page")
    docs = loader.load()
    assert len(docs) == 16
    assert loader.web_path is None

    file_path = (
        Path(__file__).parent.parent / "examples/layout-parser-paper-password.pdf"
    )
    loader = PyMuPDF4LLMLoader(file_path, password="password")
    docs = loader.load()
    assert len(docs) == 1

    web_path = "https://people.sc.fsu.edu/~jpeterson/hello_world.pdf"
    loader = PyMuPDF4LLMLoader(web_path)
    docs = loader.load()
    assert loader.web_path == web_path
    assert loader.file_path != web_path
    assert len(docs) == 1


def test_llamaindex_loader() -> None:
    """Test PyMuPDF4llm loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = LlamaIndexPDFLoader(file_path)
    docs = loader.load()
    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = LlamaIndexPDFLoader(file_path, mode="page")
    docs = loader.load()
    assert len(docs) == 16
    assert loader.web_path is None

    file_path = (
        Path(__file__).parent.parent / "examples/layout-parser-paper-password.pdf"
    )

    web_path = "https://people.sc.fsu.edu/~jpeterson/hello_world.pdf"
    loader = LlamaIndexPDFLoader(web_path)
    docs = loader.load()
    assert loader.web_path == web_path
    assert loader.file_path != web_path
    assert len(docs) == 1
