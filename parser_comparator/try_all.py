# Pas de gestion de tableau ou de liste.
# Mais retourne un texte formaté comme un tableau
import logging

import os
from pprint import pprint
from typing import List

from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader, \
    PDFMinerLoader, PyMuPDFLoader, PyMuPDF4LLMLoader, PyPDFium2Loader
from langchain_community.document_loaders.pdf import BasePDFLoader, PDFRouterLoader, \
    DedocPDFLoader
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredPDFLoader
from try_pdf import filename, PASSWORD, MODE, EXTRACT_IMAGES, EXTRACT_TABLES, \
    conv_images, PAGE_DELIMITER


from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from pdf_multi_parser import PDFMultiParser


# Utilise pdfminer en mode fast
# Implementation is based on the `extract_text` implemenation in pdfminer.six, but
#     modified to support tracking page numbers and working with file-like objects.
# PPR https://github.com/Unstructured-IO/unstructured-inference/pull/392
# PPR https://github.com/Unstructured-IO/unstructured/pull/3721
# PPR https://github.com/pymupdf/RAG/pull/170
#
pdf_loaders: list[BasePDFLoader] = [
    (AzureAIDocumentIntelligenceLoader(
        api_endpoint=endpoint,
        api_key=key,
        file_path=filename,
    ), 'AzureAIDocumentIntelligenceLoader'),
    # %%
    # (PyPDFLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #
    #     extraction_mode="plain",
    #     extraction_kwargs={
    #         "layout_mode_space_vertically": True,
    #         "layout_mode_scale_weight": 1.25,
    #         "layout_mode_strip_rotated": True,
    #         # "layout_mode_debug_path":Path("tmp/"),  # None
    #     },
    # ), "PyPDFLoader extraction_mode='layout'"),

    # (PyPDFLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #
    #     # extraction_mode="plain",
    #     extraction_mode="layout", # mode simily tableau, avec espace
    #
    #     extraction_kwargs={
    #         "layout_mode_space_vertically": True,
    #         "layout_mode_scale_weight": 1.25,
    #         "layout_mode_strip_rotated": True,
    #         # "layout_mode_debug_path":Path("tmp/"),  # None
    #     },
    # ), "PyPDFLoader extraction_mode='plain'"),

    #%%
    # https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pdf
    # (PDFMinerLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ), 'PDFMinerLoader'),
    #
    #%%
    (PyMuPDFLoader(
        filename,
        password=PASSWORD,
        mode=MODE,
        pages_delimitor = PAGE_DELIMITER,
        extract_images=EXTRACT_IMAGES,
        images_to_text=conv_images,
        extract_tables=EXTRACT_TABLES,

    ), 'PyMuPDFLoader'),

    # (PyMuPDF4LLMLoader(  #FIXME
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER, # FIXME
    #     # extract_images=EXTRACT_IMAGES,  # FIXME
    #
    #     table_strategy="lines_strict",
    #     dpi=75,
    #     margins=(90, 90),
    #     show_progress=False,
    #
    #     # embed_images=True,  # FIXME
    #     write_images=True,
    #     extract_words=True,
    #
    # ), None),

    #%%
    # (PyPDFium2Loader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    # ), 'PyPDFium2loader'),

    #%%
    # (PDFPlumberLoader(
    #     filename,
    #
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images, # FIXME: extract_images fail
    #     extract_tables=EXTRACT_TABLES,  # FIXME: csv,html,markdown
    #
    #     dedupe=False,
    #     text_kwargs={
    #         "use_text_flow": True,
    #         "presorted": False,
    #     },
    #     extract_tables_settings={
    #         "vertical_strategy": "lines",
    #         "horizontal_strategy": "lines",
    #         "snap_y_tolerance": 5,
    #         "intersection_x_tolerance": 15,
    #     }
    # ), 'PDFPlumberLoader'),

    #%%
    # (UnstructuredPDFLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_tables=EXTRACT_TABLES,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #
    #     # Paramètre pour récupérer les images des PDF
    #     # TODO: lancer une analyse description via un modèle multi-modal
    #     # extract_images_in_pdf=True,
    #     # extract_image_block_output_dir="data/example-docs/pdf/extracted-images",
    #     # extract_image_block_types=["Image", "Table"],
    #
    #     strategy="auto",
    #
    #     partition_via_api=False,
    #     skip_infer_table_types=["jpg", "png", "heic"],
    #     max_characters=5000,
    #     languages=["eng", "fra"],
    #     include_page_breaks=True,
    # ), "UnstructuredPDFLoader strategy='auto'"),

    # (UnstructuredPDFLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     extract_tables=EXTRACT_TABLES,
    #     images_to_text=conv_images,
    #
    #     strategy="fast",  # Pour extraire depuis le texte, sans OCR
    #
    #     partition_via_api=False,
    #     skip_infer_table_types=["jpg", "png", "heic"],
    #     max_characters=5000,
    #     languages=["eng", "fra"],
    #     include_page_breaks=True,
    # ), "UnstructuredPDFLoader strategy='fast'"),

    # (UnstructuredPDFLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor=PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     extract_tables=EXTRACT_TABLES,
    #     images_to_text=conv_images,
    #             strategy="hi_res",
    #     partition_via_api=False,
    #     skip_infer_table_types=["jpg", "png", "heic"],
    #     max_characters=5000,
    #     languages=["eng", "fra"],
    #     include_page_breaks=True,
    # ), "UnstructuredPDFLoader strategy='hi_res'"),

    # (UnstructuredPDFLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     pages_delimitor = PAGE_DELIMITER,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,
    #
    #     strategy="ocr_only",
    #     partition_via_api=False,
    #     skip_infer_table_types=["jpg", "png", "heic"],
    #     max_characters=5000,
    #     languages=["eng", "fra"],
    #     include_page_breaks=True,
    #
    # ), "UnstructuredPDFLoader strategy='ocr_only'"),

    # (DedocPDFLoader(  # PPR: DedocPDFLoader dependences incompatibles
    #     filename,
    #     with_tables=True,
    #     # mode="single",
    #     # pages_delimitor = PAGE_DELIMITER,
    #     # password=PASSWORD,
    #     # extract_images=True,
    # ), "DedocPDFLoader"),
]


def parse_pdf():
    for loader, name in pdf_loaders:
        if not name:
            name = type(loader).__qualname__
        print(f"\n{name}...")
        documents=loader.load()
        doc=None
        for i,doc in enumerate(documents):
            print(f"============================ {i} ============================\n{doc.page_content}")
        # print("----------------")
        # if doc:
        #     pprint(doc.metadata)
        print("-----------------------------------------------")


def _test_pdfrouter() -> List[Document]:
    from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
    from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
    from langchain_community.document_loaders.parsers import PDFPlumberParser
    routes = [
        ("Microsoft", "Microsoft", None, PyMuPDFParser(password=PASSWORD, mode=MODE)),
        ("LibreOffice", None, None, PDFPlumberParser(password=PASSWORD, mode=MODE)),
        (None, None, None, PyPDFium2Parser(password=PASSWORD, mode=MODE))
    ]
    return PDFRouterLoader(
        filename,
        routes,
        password=PASSWORD,
    ).load()

def compare_parsing():
    from pdf_multi_loader import PDFMultiLoader
    import pandas as pd
    import shutil


    loaders_dict = {name: loader for loader, name in pdf_loaders}
    parsers_dict = {name: loader.parser for loader, name in pdf_loaders}
    debug_mode=True
    sources_dir_path = './data/example-docs/test_dir_for_multi_loader'
    results_dir_path = './data/example-docs/multi_parsing_results'
    dir_to_remove = os.path.join(results_dir_path, 'experience_testname1')
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
            if not os.path.exists(result_path):
                # Créer le dossier d'expérience
                os.makedirs(result_path)

                # Parcourir les fichiers PDF dans le dossier source
                for pdf_filename in os.listdir(source_path):

                    if pdf_filename.endswith('.pdf'):
                        # Créer le sous-dossier avec le nom du pdf
                        sub_dir_pdf_path = os.path.join(result_path, pdf_filename)
                        os.makedirs(sub_dir_pdf_path)


                        # Créer l'objet PDFMultiLoader et parser le fichier
                        pdf_path = os.path.join(source_path, pdf_filename)

                        pdf_multi_parser = PDFMultiParser(parsers_dict=parsers_dict,
                                                          debug_mode=debug_mode,
                                                          )
                        pdf_multi_loader = PDFMultiLoader(file_path=pdf_path,
                                                          pdf_multi_parser=pdf_multi_parser,
                                                          )

                        if pdf_multi_loader.parser.debug_mode :
                            # Get the results per parser and the best parser list index
                            parsers_result, best_parser_data = pdf_multi_loader.load()

                            #best_parser_data = list(parsers_result)[best_parser_idx]
                            best_parser_name = best_parser_data[0]
                            best_parser_associated_documents_list = best_parser_data[1][0]

                            # Save the best parsing as .txt file
                            best_parser_concatenated_docs = "\n\n".join([doc.page_content for doc in best_parser_associated_documents_list])
                            best_parsing_file_path = os.path.join(sub_dir_pdf_path,
                                                            f"best_parsing_{best_parser_name}.txt")
                            with open(best_parsing_file_path, 'w', encoding='utf-8') as f:
                                f.write(best_parser_concatenated_docs)

                            # Créer le sous sous dossier collectant les parsings par parser
                            sub_sub_parsings_dir = os.path.join(sub_dir_pdf_path, 'parsings_by_parser')
                            os.makedirs(sub_sub_parsings_dir)

                            # store parsed documents
                            parser_name2concatenated_parsed_docs = {parser_data[0]: "\n\n".join([doc.page_content for doc in parser_data[1][0]]) for parser_data in parsers_result}

                            # Save concatenated docs as text files
                            for parser_name, concatenated_docs in parser_name2concatenated_parsed_docs.items():
                                output_file_path = os.path.join(sub_sub_parsings_dir, f"{pdf_filename}_parsed_{parser_name}.txt")
                                with open(output_file_path, 'w', encoding='utf-8') as f:
                                    f.write(concatenated_docs)

                            # store parsing scores in excel format heatmap
                            parser_name2metrics = {parser_data[0]: parser_data[1][1] for parser_data in parsers_result}
                            df = pd.DataFrame(parser_name2metrics).T
                            styled_df = df.style.background_gradient()
                            styled_df.to_excel(f'{sub_sub_parsings_dir}/parsers_metrics_results.xlsx')
                            print('debug')

                        # if not debug mode only save the best parsing as .txt file
                        else:
                            best_parser_associated_documents_list = pdf_multi_loader.load()
                            # Save the best parsing as .txt file
                            best_parser_concatenated_docs = "\n\n".join(
                                [doc.page_content for doc in best_parser_associated_documents_list])
                            best_parsing_file_path = os.path.join(sub_dir_pdf_path,
                                                                  f"best_parsing.txt")
                            with open(best_parsing_file_path, 'w', encoding='utf-8') as f:
                                f.write(best_parser_concatenated_docs)


if __name__ == "__main__":
    from langchain_community.document_loaders.parsers.pdf import logger as pdf_logger
    pdf_logger.setLevel(logging.DEBUG)

    # loader = PyPDFLoader(
    #     filename,
    #     password=PASSWORD,
    #     mode=MODE,
    #     extract_images=EXTRACT_IMAGES,
    #     images_to_text=conv_images,
    #     extract_tables=EXTRACT_TABLES,
    #
    #     # extraction_mode="plain",
    #     extraction_mode="layout",  # mode simily tableau
    #
    #     extraction_kwargs={
    #         "layout_mode_space_vertically": True,
    #         "layout_mode_scale_weight": 1.25,
    #         "layout_mode_strip_rotated": True,
    #         # "layout_mode_debug_path":Path("tmp/"),  # None
    #     },
    # )
    # loader.load()
    compare_parsing()
