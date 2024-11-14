import os

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser as old_PDFMinerParser,
    PDFPlumberParser as old_PDFPlumberParser,
    PyMuPDFParser as old_PyMuPDFParser,
    PyPDFium2Parser as old_PyPDFium2Parser,
    PyPDFParser as old_PyPDFParser,
)

from patch_langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser, convert_images_to_text_with_rapidocr, convert_images_to_text_with_tesseract,
    convert_images_to_description,
)
#FIXME rajouter unstructured

from patch_langchain_community.document_loaders.parsers.new_pdf import PyMuPDF4LLMParser, PDFMultiParser
from patch_langchain_community.document_loaders.new_pdf import PDFMultiLoader


import pandas as pd
import shutil

# FIXME ajouter des instructions en commentaire pour qu'ils comprenennt les valeurs possibles et les foncitons des paramtetres
PASSWORD = None
MODE = "single"
EXTRACT_IMAGES = False
conv_images=convert_images_to_text_with_rapidocr()
#conv_images=convert_images_to_text_with_tesseract()
#conv_images=convert_images_to_description()
EXTRACT_TABLES = "markdown"
PAGE_DELIMITER = "\n------------------\n"
IMAGE_FORMAT = "markdown"

_format_image_str = "\n{image_text}\n"
_join_images = "\n"
_join_tables = "\n"
_default_page_delimitor = "\f"  # PPR: \f ?

#FIXME utiliser les vzleurs par defaut dans les declarations

pdf_parsers_dict : dict[str, BaseBlobParser] = {
    "new-PDFMinerParser" :
    PDFMinerParser(
        password=PASSWORD,
        extract_images=EXTRACT_IMAGES,
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        images_to_text=conv_images,
    ),
    "new-PDFPlumberParser" :
    PDFPlumberParser(
        text_kwargs=None,
        dedupe=False,
        extract_images=EXTRACT_IMAGES,
        password=PASSWORD,
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        images_to_text=conv_images,
        extract_tables=EXTRACT_TABLES,
        extract_tables_settings=None,
    ),
    "new-PyMuPDFParser" :
    PyMuPDFParser(
        password=PASSWORD,
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
        extract_tables=EXTRACT_TABLES,
        extract_tables_settings=None,
        text_kwargs=None,
    ),
    "new-PyPDFium2Parser" :
    PyPDFium2Parser(
        extract_images=EXTRACT_IMAGES,
        password=PASSWORD,
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        images_to_text=conv_images,
    ),
    "new-PyPDFParser" :
    PyPDFParser(
        password=PASSWORD,
        extract_images=EXTRACT_IMAGES,
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        images_to_text=conv_images,
        extraction_mode="plain",
        extraction_kwargs=None,
    ),
    "new-PyMuPDF4LLMParser" :
    PyMuPDF4LLMParser(
        password=PASSWORD,
        mode=MODE,
        pages_delimitor=_default_page_delimitor,
        to_markdown_kwargs=None,
    ),
    "old-PDFMinerParser" :
    old_PDFMinerParser(
        extract_images=EXTRACT_IMAGES,
        concatenate_pages=(MODE=="single"),
    ),
    "old-PDFPlumberParser" :
    old_PDFPlumberParser(
        text_kwargs=None,
        dedupe=False,
        extract_images=EXTRACT_IMAGES,
    ),
    "old-PyMuPDFParser" :
    old_PyMuPDFParser(
        text_kwargs=None,
        extract_images=EXTRACT_IMAGES,

    ),
    "old-PyPDFium2Parser" :
    old_PyPDFium2Parser(
        extract_images=False,
    ),
    "old-PyPDFParser" :
    old_PyPDFParser(
        password=PASSWORD,
        extract_images=EXTRACT_IMAGES,
        extraction_mode="plain",
        extraction_kwargs=None,
    ),
}


def compare_parsing():
    debug_mode=True
    sources_dir_path = './sources_pdf'
    results_dir_path = './multi_parsing_results'
    dir_to_remove = os.path.join(results_dir_path, 'documentary_database_example') #FIXME
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)

    # Parcourir le dossier sources_pdf
    for root, dirs, files in os.walk(sources_dir_path):
        for dir_name in dirs:
            # Chemin complet du dossier source
            source_path = os.path.join(root, dir_name)

            # Chemin complet du dossier de résultats correspondant
            result_path = os.path.join(results_dir_path, dir_name)

            # Vérifier si le dossier n'existe pas dans multi_parsing_results
            if not os.path.exists(result_path): #FIXME supprimer et remplacer le dossier avec la nouvelle exp + param sys.arg de l'exp à process
                # Créer le dossier d'expérience
                os.makedirs(result_path)

                # Parcourir les fichiers PDF dans le dossier source
                for pdf_filename in os.listdir(source_path):

                    if pdf_filename.endswith('.pdf'):
                        print(f'processing {pdf_filename}... ')
                        # Créer le sous-dossier avec le nom du pdf
                        sub_dir_pdf_path = os.path.join(result_path, pdf_filename)
                        os.makedirs(sub_dir_pdf_path)


                        # Créer l'objet PDFMultiLoader et parser le fichier
                        pdf_path = os.path.join(source_path, pdf_filename)

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

                            # Save the best parsing as .txt file
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
                            # Save the best parsing as .txt file
                            best_parser_concatenated_docs = _default_page_delimitor.join(
                                [doc.page_content for doc in best_parser_associated_documents_list])
                            best_parsing_file_path = os.path.join(sub_dir_pdf_path,
                                                                  f"best_parsing.txt")
                            with open(best_parsing_file_path, 'w', encoding='utf-8') as f:
                                f.write(best_parser_concatenated_docs)
            else:
                print('experience with this name has already been processed, create a directory with a different name '
                      'if you want to process it again')

if __name__ == "__main__":
    compare_parsing()
