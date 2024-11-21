import argparse
import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers import (
    AzureAIDocumentIntelligenceParser,
)
from langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser as old_PDFMinerParser,
)
from langchain_community.document_loaders.parsers.pdf import (
    PDFPlumberParser as old_PDFPlumberParser,
)
from langchain_community.document_loaders.parsers.pdf import (
    PyMuPDFParser as old_PyMuPDFParser,
)
from langchain_community.document_loaders.parsers.pdf import (
    PyPDFium2Parser as old_PyPDFium2Parser,
)
from langchain_community.document_loaders.parsers.pdf import (
    PyPDFParser as old_PyPDFParser,
)
from patch_langchain_unstructured.document_loaders import UnstructuredPDFParser

from patch_langchain_community.document_loaders.new_pdf import (
    PDFMultiLoader,
)
from patch_langchain_community.document_loaders.parsers.new_pdf import (
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

# Under each parameter you can read a description of it and its possible values

MODE = "single"
# Extraction mode to use. Either "single" or "paged"
EXTRACT_IMAGES = False
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
MAX_WORKERS = 1
# If use Old parser, set to 1

load_dotenv()
# AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')


pdf_parsers_dict: dict[str, BaseBlobParser] = {
    "PDFMinerParser_new": PDFMinerParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
    ),
    # # %%
    # "PDFPlumberParser_new": PDFPlumberParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,
    # ),
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
    # "PDFMinerParser_old": old_PDFMinerParser(
    #     extract_images=EXTRACT_IMAGES,
    #     concatenate_pages=(MODE == "single"),
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
    # "PyPDFParser_old": old_PyPDFParser(
    #     extract_images=EXTRACT_IMAGES,
    #     extraction_mode="plain",
    # ),
    # # %%
    # "AzureAIDocumentIntelligenceParser": AzureAIDocumentIntelligenceParser(
    #     api_endpoint=os.environ["AZURE_API_ENDPOINT"],
    #     api_key=os.environ["AZURE_API_KEY"],
    #     # api_version=AZURE_API_VERSION,
    # ),
    # # %%
    # "PyMuPDF4LLMParser": PyMuPDF4LLMParser(
    #     mode=MODE,
    #     pages_delimitor=_default_page_delimitor,
    #     to_markdown_kwargs=None,
    # ),
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


def compare_parsing(experiment_name: str):
    debug = True
    base_dir = Path(__file__).parent
    sources_dir_path = base_dir / "sources_pdf"
    results_dir_path = base_dir / "multi_parsing_results"

    # Iterating over the directories in the sources directory
    # for root, dirs, files in os.walk(sources_dir_path):
    for pdf_file_relative_path in glob("**/*.pdf", root_dir=sources_dir_path, recursive=True):
        pdf_file_relative_path = Path(pdf_file_relative_path)
        experiment_dir = results_dir_path / pdf_file_relative_path / experiment_name

        print(f"processing {pdf_file_relative_path}... ")

        pdf_multi_parser = PDFMultiParser(
            parsers=pdf_parsers_dict,
            debug=debug,
            max_workers=MAX_WORKERS,
        )
        # pdf_multi_loader = PDFMultiLoader(
        #     file_path=sources_dir_path / pdf_file_relative_path,
        #     pdf_multi_parser=pdf_multi_parser,
        # )

        blob = Blob.from_path(sources_dir_path / pdf_file_relative_path)

        # get the results per parser and the best parser info
        try:
            parser_name2parser_results = pdf_multi_parser.parse_and_return_all_results(blob)

            # create a sub directory to store the parsings by parser
            parsings_subdir = experiment_dir / "parsings_by_parser"
            parsings_subdir.mkdir(parents=True, exist_ok=True)

            # parser_name2list_parsed_docs_list = {parser_data[0]: parser_data[1][0] for parser_data in parsers_result}
            # for parser_name, parsed_docs_list in parser_name2list_parsed_docs_list.items(): #FIXME delete when done
            #     if len(parsed_docs_list) > 1:
            #         print(f"returned docs list by {parser_name} : {parsed_docs_list}")
            #         raise Exception(f"{parser_name} works as if in paged mode")
            # store parsed documents
            parser_name2concatenated_parsed_docs = {
                parser_name: _default_page_delimitor.join(
                    [doc.page_content for doc in parser_results['documents']]
                )
                for parser_name, parser_results in parser_name2parser_results.items()
            }

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
            best_parser_name = [parser_name for parser_name, parser_results in parser_name2parser_results.items() if parser_results['is_best']][0]
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
                parser_name: parser_results['metrics']
                for parser_name, parser_results in parser_name2parser_results.items()
            }
            df = pd.DataFrame(parser_name2metrics).T
            styled_df = df.style.background_gradient()
            styled_df.to_excel(
                f"{parsings_subdir}/parsers_metrics_results.xlsx"
            )
        except Exception as e:
            print(f"Error processing {pdf_file_relative_path}: {e}")
            raise e


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Compare PDF parsing results.")
        parser.add_argument("experiment_name", type=str, help="Name of the experiment")
        args = parser.parse_args()
        compare_parsing(args.experiment_name)
    else:
        compare_parsing("default")
