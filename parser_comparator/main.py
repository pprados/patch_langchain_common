import argparse
import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers import (
    AzureAIDocumentIntelligenceParser,
)
from langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser as old_PDFMinerParser,
    PDFPlumberParser as old_PDFPlumberParser,
    PyMuPDFParser as old_PyMuPDFParser,
    PyPDFium2Parser as old_PyPDFium2Parser,
    PyPDFParser as old_PyPDFParser,
)
from langchain_unstructured.document_loaders import UnstructuredLoader

#%% Import patch
from patch_langchain_community.document_loaders.new_pdf import (
    PDFMultiLoader,
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
    convert_images_to_description,
    convert_images_to_text_with_rapidocr,
    convert_images_to_text_with_tesseract,
)
from patch_langchain_unstructured.document_loaders import UnstructuredPDFParser


#%% Meta parameters
# Under each parameter you can read a description of it and its possible values
debug = True

MODE = "single"
# Extraction mode to use. Either "single" or "paged"
EXTRACT_IMAGES = True
# Whether to extract images from the PDF. True/False
IMAGE_FORMAT = "markdown"
# Format to use for the extracted images. Either "text", "html" or "markdown"
conv_images = convert_images_to_text_with_rapidocr(format=IMAGE_FORMAT)
# Function to extract text from images using rapid OCR
# conv_images=convert_images_to_text_with_tesseract(langs=['eng'], format=IMAGE_FORMAT)
# Function to extract text from images using tesseract
# conv_images=convert_images_to_description(model=ChatOpenAI(model='gpt-4o'))
# Function to extract text from images using multimodal model
EXTRACT_TABLES = "markdown"
# Format to use for the extracted tables. Either "text", "html" or "markdown"
_default_page_delimitor = "\f"
# Delimiter that will be put between pages in 'single' mode
SUFFIX = "md"
USE_OLD_PARSERS = True
MAX_WORKERS = 1
CONTINUE_IF_ERROR=False

load_dotenv()
# AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
import logging # Set the logging level to WARNING to reduce verbosity
logging.getLogger('azure').setLevel(logging.WARNING)

pdf_parsers_new: dict[str, BaseBlobParser] = {
    # "PDFMinerParser_new": PDFMinerParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ),
    # %%
    "PDFPlumberParser_new": PDFPlumberParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
        extract_tables=EXTRACT_TABLES,
    ),
    # # %%
    # "PyMuPDFParser_new": PyMuPDFParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,
    # ),
    # # #%%
    # "PyPDFium2Parser_new": PyPDFium2Parser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ),
    # # #%%
    # "PyPDFParser_new": PyPDFParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ),
    # # %%
    # "PyMuPDF4LLMParser_new": PyMuPDF4LLMParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     to_markdown_kwargs=None,
    # ),
    # # %%
    # "UnstructuredPDFParser_fast_new": UnstructuredPDFParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="fast",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,
    # ),
    # # %% BUG avec 11:SIGSEGV
    # "UnstructuredPDFParser_ocr_only_new": UnstructuredPDFParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="ocr_only",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,
    # ),
    # # %%
    # "UnstructuredPDFParser_hi_res_new": UnstructuredPDFParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     strategy="hi_res",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,
    # ),
    # # %%
    # "AzureAIDocumentIntelligenceParser": AzureAIDocumentIntelligenceParser(
    #     api_endpoint=os.environ["AZURE_API_ENDPOINT"],
    #     api_key=os.environ["AZURE_API_KEY"],
    #     # api_version=AZURE_API_VERSION,
    # ),
    # %%
    "PyMuPDF4LLMParser": PyMuPDF4LLMParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        to_markdown_kwargs=None,
    ),
    # # %%
    # "LlamaIndexPDFParser": LlamaIndexPDFParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     extract_tables=EXTRACT_TABLES,
    #     language="en",
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ),
}

pdf_parsers_old: dict[str, BaseBlobParser] = {
    # %%
    "PDFMinerParser_old": old_PDFMinerParser(
        extract_images=EXTRACT_IMAGES,
        concatenate_pages=(MODE == "single"),
    ),
    # %%
    "PDFPlumberParser_old": old_PDFPlumberParser(
        text_kwargs=None,
        dedupe=False,
        extract_images=EXTRACT_IMAGES,
    ),
    # %%
    "PyMuPDFParser_old": old_PyMuPDFParser(
        text_kwargs=None,
        extract_images=EXTRACT_IMAGES,
    ),
    # %%
    "PyPDFium2Parser_old": old_PyPDFium2Parser(
        extract_images=False,
    ),
    # %%
    "PyPDFParser_old": old_PyPDFParser(
        extract_images=EXTRACT_IMAGES,
        extraction_mode="plain",
    ),
}
pdf_loader_old={
    "UnstructuredPDFParser_fast_new": (UnstructuredLoader,{"strategy":"fast"}),
    "UnstructuredPDFParser_auto_new": (UnstructuredLoader,{"strategy":"auto"}),
    "UnstructuredPDFParser_ocr_only_new": (UnstructuredLoader,{"strategy":"ocr_only"}),
    "UnstructuredPDFParser_hi_res_new": (UnstructuredLoader,{"strategy":"hi_res"}),
}

if USE_OLD_PARSERS:
    pdf_parsers = {**pdf_parsers_old , **pdf_parsers_new}
    MAX_WORKERS = 1  # If use Old parser, set to 1
else:
    pdf_parsers = pdf_parsers_new

def compare_parsing(experiment_name: str):
    global pdf_parsers
    global debug
    base_dir = Path(__file__).parent
    sources_dir_path = base_dir / "sources_pdf"
    results_dir_path = base_dir / "multi_parsing_results"

    # Iterating over the directories in the sources directory
    # for root, dirs, files in os.walk(sources_dir_path):
    # FIXME
    for pdf_filename in glob("**/*.pdf", root_dir=sources_dir_path, recursive=True):
    # for pdf_filename in glob("pdfminer/nonfree/naa*.pdf", root_dir=sources_dir_path, recursive=True):
    # for pdf_filename in glob("*/*.pdf", root_dir=sources_dir_path, recursive=False):
        pdf_file_relative_path = Path(pdf_filename)
        experiment_dir = results_dir_path / pdf_filename / experiment_name

        print(f"processing {pdf_filename}... ")

        pdf_multi_parser = PDFMultiParser(
            parsers=pdf_parsers,
            max_workers=MAX_WORKERS,
            continue_if_error=CONTINUE_IF_ERROR,
        )
        blob = Blob.from_path(sources_dir_path / pdf_file_relative_path)

        # get the results per parser and the best parser info
        try:
            parsers_results = pdf_multi_parser.parse_and_evaluate(
                blob
            )

            # create a sub directory to store the parsings by parser
            parsings_subdir = experiment_dir / "parsings_by_parser"
            parsings_subdir.mkdir(parents=True, exist_ok=True)

            # if the experiment directory contains some files, delete them
            for item in Path(experiment_dir).rglob('*'):
                if item.is_file():
                    item.unlink()

            # parser_name2list_parsed_docs_list = {parser_data[0]: parser_data[1][0] for parser_data in parsers_result}
            # for parser_name, parsed_docs_list in parser_name2list_parsed_docs_list.items(): #FIXME delete when done
            #     if len(parsed_docs_list) > 1:
            #         print(f"returned docs list by {parser_name} : {parsed_docs_list}")
            #         raise Exception(f"{parser_name} works as if in paged mode")
            # store parsed documents
            parser_name2concatenated_parsed_docs = {
                parser_data[0]:
                    _default_page_delimitor.join([doc.page_content for doc in parser_data[1]])
                for parser_data in parsers_results}

            # save concatenated docs parsings as text files
            for (
                    parser_name,
                    concatenated_docs,
            ) in parser_name2concatenated_parsed_docs.items():
                output_file_path = (
                        parsings_subdir
                        / f"{pdf_file_relative_path.name}_parsed_{parser_name}.{SUFFIX}"
                )
                output_file_path.parent.mkdir(exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(concatenated_docs)

            # get the best parser name and its concatenated parsed docs
            best_parser_name = parsers_results[0][0]
            best_parser_concatenated_docs = parser_name2concatenated_parsed_docs[
                best_parser_name
            ]

            # save the best parsing as .txt file
            best_parsing_file_path = (
                    experiment_dir / f"best_parsing_{best_parser_name}.{SUFFIX}"
            )
            with open(best_parsing_file_path, "w", encoding="utf-8") as f:
                f.write(best_parser_concatenated_docs)

            # store parsing scores in excel format heatmap
            parser_name2metrics = {
                parser_data[0]: parser_data[2] for parser_data in parsers_results
            }
            df = pd.DataFrame(parser_name2metrics).T
            styled_df = df.style.background_gradient()
            styled_df.to_excel(f"{parsings_subdir}/parsers_metrics_results.xlsx")
        except Exception as e:
            print(f"Error processing {pdf_file_relative_path}: {e}")
            raise e

            # To inject older loaders, without parsers
            if USE_OLD_PARSERS:
                for name,(clazz,kwargs) in pdf_loader_old.items():
                    pdf_loader = clazz(file_path=sources_dir_path / pdf_filename, **kwargs)
                    pdf_loader.load()
        print(f"processing {pdf_filename} done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Compare PDF parsing results.")
        parser.add_argument("experiment_name", type=str, help="Name of the experiment")
        args = parser.parse_args()
        compare_parsing(args.experiment_name)
    else:
        compare_parsing("default")
