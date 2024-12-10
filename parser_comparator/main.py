# ruff: disable=T201
import argparse
import json
import logging  # Set the logging level to WARNING to reduce verbosity
import os
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents.base import Blob
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.pdf import (
    UnstructuredPDFLoader as old_UnstructuredPDFLoader,
    DedocPDFLoader as old_DedocPDFLoader,
)

from langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser as old_PDFMinerParser,
    PDFPlumberParser as old_PDFPlumberParser,
    PyMuPDFParser as old_PyMuPDFParser,
    PyPDFium2Parser as old_PyPDFium2Parser,
    PyPDFParser as old_PyPDFParser,
)
from langchain_community.document_loaders.parsers import (
    AzureAIDocumentIntelligenceParser,
)
from patch_langchain_unstructured.document_loaders import UnstructuredPDFParser

# %% Import patch
from patch_langchain_community.document_loaders.new_pdf import (
    LlamaIndexPDFParser,
    PDFMultiParser,
    PyMuPDF4LLMParser,
)
from patch_langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
    convert_images_to_text_with_rapidocr, _default_page_delimitor,
)

# %% Meta parameters
# Under each parameter you can read a description of it and its possible values
MODE = "single"
# Extraction mode to use. Either "single" or "paged"
EXTRACT_IMAGES = False
# Whether to extract images from the PDF. True/False
IMAGE_FORMAT = "text"  # FIXME "markdown"
# Format to use for the extracted images. Either "text", "html" or "markdown"
conv_images = convert_images_to_text_with_rapidocr(format=IMAGE_FORMAT)
# Function to extract text from images using rapid OCR
# conv_images=convert_images_to_text_with_tesseract(langs=['eng'], format=IMAGE_FORMAT)
# Function to extract text from images using tesseract
# conv_images=convert_images_to_description(model=ChatOpenAI(model='gpt-4o'))
# Function to extract text from images using multimodal model
EXTRACT_TABLES = "html"
# Format to use for the extracted tables. Either "text", "html" or "markdown"
SUFFIX = "md"
# If True, compare with the old versions of parsers/loader
USE_OLD_PARSERS = True
# If True, force the MODE to be the default old value
RETRO_COMPATIBLE = True
# If True, invoke online parser. Must have TOKEN API
USE_ONLINE_PARSERS = False
# Number of // workers. Desactivated with USE_OLD_PARSERS
MAX_WORKERS:Optional[int] = None
# If True, continue with the next parser if error. Else, stop at the first error.
CONTINUE_IF_ERROR = True

load_dotenv()

logging.getLogger("azure").setLevel(logging.WARNING)

set_llm_cache(InMemoryCache())

pdf_parsers_updated: dict[str, BaseBlobParser] = {
    # "PDFMinerParser_single_new": PDFMinerParser(
    #     mode="single",  # type:ignore
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ),
    # "PDFMinerParser_page_new": PDFMinerParser(
    #     mode="page",  # type:ignore
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ),
    # # %%
    # "PDFPlumberParser_new": PDFPlumberParser(
    #     mode="page" if RETRO_COMPATIBLE else MODE,  # type:ignore
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,  # type:ignore
    # ),
    # # %%
    # "PyMuPDFParser_new": PyMuPDFParser(
    #     mode="page" if RETRO_COMPATIBLE else MODE,  # type:ignore
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,  # type:ignore
    # ),
    # #%%
    # "PyPDFium2Parser_new": PyPDFium2Parser(
    #     mode="page" if RETRO_COMPATIBLE else MODE,  # type:ignore
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ),
    # #%%
    # "PyPDFParser_plain_new": PyPDFParser(
    #     mode="page" if RETRO_COMPATIBLE else MODE,  # type:ignore
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extraction_mode = "plain",
    # ),
    # #%%
    # "PyPDFParser_layout_new": PyPDFParser(
    #     mode="page" if RETRO_COMPATIBLE else MODE,  # type:ignore
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extraction_mode="layout",
    # ),
    # # %%
    # "UnstructuredPDFParser_auto_new": UnstructuredPDFParser(
    #     mode=MODE,  # type: ignore
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="auto",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,  # type: ignore
    # ),
    # # %%
    # "UnstructuredPDFParser_fast_new": UnstructuredPDFParser(
    #     mode=MODE,  # type: ignore
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="fast",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,  # type: ignore
    # ),
    # # %%
    # "UnstructuredPDFParser_ocr_only_new": UnstructuredPDFParser(
    #     mode=MODE,  # type: ignore
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="ocr_only",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,  # type: ignore
    # ),
    # # %%
    # "UnstructuredPDFParser_hi_res_new": UnstructuredPDFParser(
    #     mode=MODE,  # type: ignore
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="hi_res",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,  # type: ignore
    # ),
    # # %%
    # "UnstructuredPDFParser_elements_new": UnstructuredPDFParser(
    #     mode="elements",  # type: ignore
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="hi_res",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,  # type: ignore
    # ),
}
pdf_parsers_new: dict[str, BaseBlobParser] = {
    # %%
    "PyMuPDF4LLMParser": PyMuPDF4LLMParser(
        mode=MODE,  # type:ignore
        pages_delimitor=_default_page_delimitor,
        to_markdown_kwargs=None,
    ),
}

pdf_online_parsers: dict[str, BaseBlobParser] = {
    # %%
    "AzureAIDocumentIntelligenceParser": AzureAIDocumentIntelligenceParser(
        api_endpoint=os.environ["AZURE_API_ENDPOINT"],
        api_key=os.environ["AZURE_API_KEY"],
    ),
    # %%
    "LlamaIndexPDFParser": LlamaIndexPDFParser(
        mode=MODE,  # type: ignore
        pages_delimitor=_default_page_delimitor,
        extract_tables=EXTRACT_TABLES,  # type:ignore
        language="en",
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
    ),
}

pdf_parsers_old: dict[str, BaseBlobParser] = {
    # # %%
    # "PDFMinerParser_single_old": old_PDFMinerParser(
    #     extract_images=EXTRACT_IMAGES,
    #     concatenate_pages=True,
    # ),
    # # %%
    # "PDFMinerParser_page_old": old_PDFMinerParser(
    #     extract_images=EXTRACT_IMAGES,
    #     concatenate_pages=False,
    # ),
    # # %%
    # "PDFPlumberParser_old": old_PDFPlumberParser(
    #     text_kwargs=None,
    #     dedupe=False,
    #     extract_images=EXTRACT_IMAGES,
    # ),
    # # %%
    # "PyMuPDFParser_old": old_PyMuPDFParser(
    #     text_kwargs=None,
    #     extract_images=EXTRACT_IMAGES,
    # ),
    # # %%
    # "PyPDFium2Parser_old": old_PyPDFium2Parser(
    #     extract_images=False,
    # ),
    # # %%
    # "PyPDFParser_plain_old": old_PyPDFParser(
    #     extract_images=EXTRACT_IMAGES,
    #     extraction_mode="plain",
    # ),
    # # %%
    # "PyPDFParser_layout_old": old_PyPDFParser(
    #     extract_images=EXTRACT_IMAGES,
    #     extraction_mode="layout",
    # ),
}
pdf_loader_old = {
    "DedocPDFLoader": (old_DedocPDFLoader, {}),
    # "UnstructuredPDFParser_fast_old": (old_UnstructuredPDFLoader, {"mode":MODE, "strategy": "fast"}),
    # "UnstructuredPDFParser_auto_old": (old_UnstructuredPDFLoader, {"mode":MODE,"strategy": "auto"}),
    # "UnstructuredPDFParser_ocr_only_old": (old_UnstructuredPDFLoader,{"mode":"single", "strategy": "ocr_only"},),
    # "UnstructuredPDFParser_hi_res_old": (old_UnstructuredPDFLoader, {"mode":MODE, "strategy": "hi_res"}),
    # "UnstructuredPDFParser_elements_old": (old_UnstructuredPDFLoader, {"mode":"elements","strategy": "hi_res"}),
}

if USE_OLD_PARSERS:
    pdf_parsers = {**pdf_parsers_old, **pdf_parsers_updated}
    MAX_WORKERS = 1  # If use Old parser, set to 1
else:
    pdf_parsers = {**pdf_parsers_updated, **pdf_parsers_new}
if USE_ONLINE_PARSERS:
    pdf_parsers = {**pdf_parsers, **pdf_online_parsers}


def compare_parsing(experiment_name: str) -> None:
    global pdf_parsers
    global debug
    base_dir = Path(__file__).parent
    sources_dir_path = base_dir / "sources_pdf"
    results_dir_path = base_dir / "multi_parsing_results"

    # Iterating over the directories in the sources directory
    # for root, dirs, files in os.walk(sources_dir_path):
    for pdf_filename in glob("**/*.pdf", root_dir=sources_dir_path, recursive=True):
        pdf_file_relative_path = Path(pdf_filename)
        experiment_dir = results_dir_path / pdf_filename / experiment_name

        print(f"processing {pdf_filename}... ")  # noqa: T201

        pdf_multi_parser = PDFMultiParser(
            parsers=pdf_parsers,
            max_workers=MAX_WORKERS,
            continue_if_error=CONTINUE_IF_ERROR,
        )
        blob = Blob.from_path(sources_dir_path / pdf_file_relative_path)

        parsings_subdir = experiment_dir / "parsings_by_parser"
        parsings_subdir.mkdir(parents=True, exist_ok=True)
        for item in Path(experiment_dir).rglob("*"):
            if item.is_file():
                item.unlink()

        # get the results per parser and the best parser info
        try:
            parsers_results = pdf_multi_parser.parse_and_evaluate(blob)

            parser_name2concatenated_parsed_docs = _save_results(parsers_results,
                                                                 parsings_subdir,
                                                                 pdf_file_relative_path)

            # get the best parser name and its concatenated parsed docs
            best_parser_name = parsers_results[0][0]
            best_parser_concatenated_docs = parser_name2concatenated_parsed_docs[
                best_parser_name
            ]

            # save the best parsing. Not implemented
            # best_parsing_file_path = (
            #     experiment_dir / f"best_parsing_{best_parser_name}.{SUFFIX}"
            # )
            # with open(best_parsing_file_path, "w", encoding="utf-8") as f:
            #     f.write(best_parser_concatenated_docs)

            # store parsing scores in excel format heatmap
            parser_name2metrics = {
                parser_data[0]: parser_data[2] for parser_data in parsers_results
            }
            df = pd.DataFrame(parser_name2metrics).T
            styled_df = df.style.background_gradient()
            styled_df.to_excel(f"{parsings_subdir}/parsers_metrics_results.xlsx")
        except Exception as e:
            print(f"Error processing {pdf_file_relative_path}: {e}")  # noqa: T201
            raise e

        # To inject older loaders, without parsers
        if USE_OLD_PARSERS:
            parsers_results=[]
            for name, (clazz, kwargs) in pdf_loader_old.items():
                pdf_loader = clazz(
                    file_path=str(sources_dir_path / pdf_filename), **kwargs
                )
                documents=pdf_loader.load()
                if "Unstructured" in name:
                    for doc in documents:
                        doc.page_content=doc.page_content.replace("\n\n","\n")
                parsers_results.append((name,documents,{}))
            _save_results(parsers_results,
                          parsings_subdir,
                          pdf_file_relative_path)
        print(f"processing {pdf_filename} done.")  # noqa: T201


def _save_results(parsers_results, parsings_subdir, pdf_file_relative_path):
    # store parsed documents
    parser_name2concatenated_parsed_docs = {
        parser_data[0]: _default_page_delimitor.join(
            [doc.page_content for doc in parser_data[1]]
        )
        for parser_data in parsers_results
    }
    parser_name2concatenated_parsed_metadata = {
        parser_data[0]: [doc.metadata for doc in parser_data[1]]
        for parser_data in parsers_results
    }
    # save concatenated docs parsings as text files
    for (
            parser_name,
            concatenated_docs,
    ) in parser_name2concatenated_parsed_docs.items():
        output_file_path = (
                parsings_subdir
                / f"{pdf_file_relative_path.name}_parsed_{parser_name}."
        )
        output_file_path.parent.mkdir(exist_ok=True)
        with open(str(output_file_path) + SUFFIX, "w",
                  encoding="utf-8") as f:
            f.write(concatenated_docs)
        with open(str(output_file_path) + "properties", "w",
                  encoding="utf-8") as f:
            metadata=parser_name2concatenated_parsed_metadata[parser_name][0]
            metadata={k.lower():v for k,v in metadata.items()}  # FIXME: remove
            json.dump(metadata,
                      f, indent=2, sort_keys=True)
    return parser_name2concatenated_parsed_docs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Compare PDF parsing results.")
        parser.add_argument("experiment_name", type=str, help="Name of the experiment")
        args = parser.parse_args()
        compare_parsing(args.experiment_name)
    else:
        compare_parsing("default")
