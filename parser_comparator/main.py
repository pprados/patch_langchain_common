import argparse
import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders.base import BaseBlobParser
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

from patch_langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
    convert_images_to_text_with_rapidocr,
    convert_images_to_text_with_tesseract,
    convert_images_to_description,
)
from patch_langchain_unstructured.document_loaders import UnstructuredPDFParser

from patch_langchain_community.document_loaders.parsers.new_pdf import (
    PyMuPDF4LLMParser,
    PDFMultiParser,
    LlamaIndexPDFParser,
)
from patch_langchain_community.document_loaders.new_pdf import (
    PDFMultiLoader,
)

import pandas as pd
import shutil

# Under each parameter you can read a description of it and its possible values

MODE = "single"
# Extraction mode to use. Either "single" or "paged"
EXTRACT_IMAGES = True
# Whether to extract images from the PDF. True/False
IMAGE_FORMAT = "markdown"
# Format to use for the extracted images. Either "text", "html" or "markdown"
conv_images=convert_images_to_text_with_rapidocr(format=IMAGE_FORMAT)
# Function to extract text from images using rapid OCR
#conv_images=convert_images_to_text_with_tesseract(langs=['eng'], format=IMAGE_FORMAT)
# Function to extract text from images using tesseract
#conv_images=convert_images_to_description(model=ChatOpenAI(model='gpt-4o'))
# Function to extract text from images using multimodal model
EXTRACT_TABLES = "markdown"
# Format to use for the extracted tables. Either "text", "html" or "markdown"
_default_page_delimitor = "\f"
# Delimiter that will be put between pages in 'single' mode

load_dotenv()
AZURE_API_ENDPOINT = os.getenv('AZURE_API_ENDPOINT')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')


pdf_parsers_dict : dict[str, BaseBlobParser] = {
    "PDFMinerParser_new" :
    PDFMinerParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
    ),
    #%%
    "PDFPlumberParser_new" :
    PDFPlumberParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
        extract_tables=EXTRACT_TABLES,
    ),
    #%%
    "PyMuPDFParser_new" :
    PyMuPDFParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
        extract_tables=EXTRACT_TABLES,
    ),
    # #%%
    "PyPDFium2Parser_new" :
    PyPDFium2Parser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
    ),
    # #%%
    "PyPDFParser_new" :
    PyPDFParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
    ),
    #%%
    "PyMuPDF4LLMParser_new" :
    PyMuPDF4LLMParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        to_markdown_kwargs=None,
    ),
    #%%
    "UnstructuredPDFParser_new" :
    UnstructuredPDFParser(
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        strategy='hi_res',
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
        extract_tables=EXTRACT_TABLES,
    ),
    #%%
    "PDFMinerParser_old" :
    old_PDFMinerParser(
        extract_images=EXTRACT_IMAGES,
        concatenate_pages=(MODE=="single"),
    ),
    #%%
    "PDFPlumberParser_old" :
    old_PDFPlumberParser(
        text_kwargs=None,
        dedupe=False,
        extract_images=EXTRACT_IMAGES,
    ),
    #%%
    "PyMuPDFParser_old" :
    old_PyMuPDFParser(
        text_kwargs=None,
        extract_images=EXTRACT_IMAGES,

    ),
    #%%
    "PyPDFium2Parser_old" :
    old_PyPDFium2Parser(
        extract_images=False,
    ),
    #%%
    "PyPDFParser_old" :
    old_PyPDFParser(
        extract_images=EXTRACT_IMAGES,
        extraction_mode="plain",
    ),
    #%%
    "AzureAIDocumentIntelligenceParser" :
    AzureAIDocumentIntelligenceParser(
        api_endpoint=AZURE_API_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    ),
    #%%
    "PyMuPDF4LLMParser":
        PyMuPDF4LLMParser(
            mode=MODE,
            pages_delimitor=_default_page_delimitor,
            to_markdown_kwargs=None,
        ),
    #%%
    "LlamaIndexPDFParser":
        LlamaIndexPDFParser(
            mode=MODE,
            pages_delimitor=_default_page_delimitor,
            extract_tables=EXTRACT_TABLES,
            language='en',
            extract_images=EXTRACT_IMAGES,
            images_to_text=conv_images,
        ),

}


def compare_parsing(experiment_name : str):
    debug_mode=True
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir_path = os.path.join(base_dir, 'sources_pdf')
    results_dir_path = os.path.join(base_dir, 'multi_parsing_results')

    # Iterating over the directories in the sources directory
    for root, dirs, files in os.walk(sources_dir_path):
        for dir_name in dirs:
            doc_db_source_path = os.path.join(root, dir_name)

            # Create the documentary database results directory if it does not exist
            doc_db_results_path = os.path.join(results_dir_path, dir_name)
            os.makedirs(doc_db_results_path, exist_ok=True)

            experiment_path = os.path.join(doc_db_results_path, experiment_name)
            # Delete the experiment directory if it already exists
            shutil.rmtree(experiment_path, ignore_errors=True)
            # Create the experiment directory
            os.makedirs(experiment_path)

            # Iterating over the pdf files in the documentary database source directory
            for pdf_filename in os.listdir(doc_db_source_path):

                if pdf_filename.endswith('.pdf'):
                    print(f'processing {pdf_filename}... ')
                    # Create a sub directory to store the parsings of the experiment
                    sub_dir_pdf_path = os.path.join(experiment_path, pdf_filename)
                    os.makedirs(sub_dir_pdf_path)


                    # Load the pdf file
                    pdf_path = os.path.join(doc_db_source_path, pdf_filename)

                    pdf_multi_parser = PDFMultiParser(parsers_dict=pdf_parsers_dict,
                                                      debug_mode=debug_mode,
                                                      )
                    pdf_multi_loader = PDFMultiLoader(file_path=pdf_path,
                                                      pdf_multi_parser=pdf_multi_parser,
                                                      )

                    if pdf_multi_loader.parser.debug_mode :
                        # get the results per parser and the best parser info
                        parsers_result, best_parser_name = pdf_multi_loader.load()

                        # create a sub directory to store the parsings by parser
                        sub_sub_parsings_dir = os.path.join(sub_dir_pdf_path, 'parsings_by_parser')
                        os.makedirs(sub_sub_parsings_dir)

                        #parser_name2list_parsed_docs_list = {parser_data[0]: parser_data[1][0] for parser_data in parsers_result}
                        # for parser_name, parsed_docs_list in parser_name2list_parsed_docs_list.items(): #FIXME delete when done
                        #     if len(parsed_docs_list) > 1:
                        #         print(f"returned docs list by {parser_name} : {parsed_docs_list}")
                        #         raise Exception(f"{parser_name} works as if in paged mode")
                        # store parsed documents
                        parser_name2concatenated_parsed_docs = {parser_data[0]: _default_page_delimitor.join([doc.page_content for doc in parser_data[1][0]]) for parser_data in parsers_result}

                        # save concatenated docs parsings as text files
                        for parser_name, concatenated_docs in parser_name2concatenated_parsed_docs.items():
                            output_file_path = os.path.join(sub_sub_parsings_dir, f"{pdf_filename}_parsed_{parser_name}.txt")
                            with open(output_file_path, 'w', encoding='utf-8') as f:
                                f.write(concatenated_docs)

                        # get the best parser name and its concatenated parsed docs
                        best_parser_concatenated_docs = parser_name2concatenated_parsed_docs[best_parser_name]

                        # save the best parsing as .txt file
                        best_parsing_file_path = os.path.join(sub_dir_pdf_path,
                                                              f"best_parsing_{best_parser_name}.txt")
                        with open(best_parsing_file_path, 'w', encoding='utf-8') as f:
                            f.write(best_parser_concatenated_docs)

                        # store parsing scores in excel format heatmap
                        parser_name2metrics = {parser_data[0]: parser_data[1][1] for parser_data in parsers_result}
                        df = pd.DataFrame(parser_name2metrics).T
                        styled_df = df.style.background_gradient()
                        styled_df.to_excel(f'{sub_sub_parsings_dir}/parsers_metrics_results.xlsx')

                    # if not debug mode only save the best parsing as .txt file
                    else:
                        best_parser_associated_documents_list = pdf_multi_loader.load()
                        # save the best parsing as .txt file
                        best_parser_concatenated_docs = _default_page_delimitor.join(
                            [doc.page_content for doc in best_parser_associated_documents_list])
                        best_parsing_file_path = os.path.join(sub_dir_pdf_path,
                                                              f"best_parsing.txt")
                        with open(best_parsing_file_path, 'w', encoding='utf-8') as f:
                            f.write(best_parser_concatenated_docs)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Compare PDF parsing results.")
        parser.add_argument("experiment_name", type=str, help="Name of the experiment")
        args = parser.parse_args()
        compare_parsing(args.experiment_name)
    else:
        compare_parsing("default_experiment_name")
