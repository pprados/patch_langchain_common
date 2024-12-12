# Refactoring all PDF loader and parser: community

- **Description:** refactoring of PDF parsers and loaders. See below
- **Issue:** missing lock, parameter inconsistency, missing lazy approach, split loader and parser, etc.
- **Twitter handle:** pprados

- [X] **Add tests and docs**: 
  - Add tests to check the consistency of different implementations
  - Add tests to check the array and images extraction
  - Update or add notebooks in `docs/docs/integrations` directory

- [X] **Lint and test**: done


We propose a substantial PR to improve the different PDF parser integrations. All my clients struggle with PDFs. I took the initiative to address this issue at its root by refactoring the various integrations of Python PDF parsers. The goal is to standardize a minimum set of parameters and metadata and bring improvements to each one (bug fixes, feature additions).

We're sorry it may take you several hours to validate it. The changes are important and cannot be published one after the other, as everything is linked.

``` 
git clone https://github.com/pprados/patch_langchain_common.git
cd patch_langchain_common/compare_old_new
```
Then, you can use **diff** from your devtools

Even though `Document` has a `page_content` parameter (rather than text or body), we believe it’s not good practice to work with pages. Indeed, this approach creates memory gaps in RAG projects. If a paragraph spans two pages, the beginning of the paragraph is at the end of one page, while the rest is at the start of the next. With a page-based approach, there will be two separate chunks, each containing part of a sentence. The corresponding vectors won’t be relevant. These chunks are unlikely to be selected when there’s a question specifically about the split paragraph. If one 
of the chunks is selected, there’s little chance the LLM can answer the question. This issue is worsened by the injection of headers, footers (if parsers haven’t properly removed them), images, or tables at the end of a page, as most current implementations tend to do.

In order to qualify all the code, we worked in a separate project, using the `langchain-common` structure. In this way, we can compare the results of the historical implementation with the new ones.

We understand that it's important to ensure that changes don't have a significant impact on existing code. That's why we used a parallel project, using the `langchain-common` structure, to test PDF readings before and after modifications$. This allows us to compare results. You'll find all the files [here](https://github.com/pprados/patch_langchain_common/tree/master/compare_old_new).
The only difference is the name to import classes.

Why is it important to unify the different parsers? Each has its own characteristics and strategies, more or less effective depending on the family of PDF files. One strategy is to identify the family of the PDF file (by inspecting the metadata or the content of the first page) and then select the most efficient parser in that case. By unifying parsers, the following code doesn't need to deal with the specifics of different parsers, as the result is similar for each. We'll propose a Parser using this strategy in another PR.

## Metadata
All parsers use lowercase keys for pdf file metadata. Except `PDFPlumberParser`. For this particular case, we've added a dictionary wrapper that warns when keys with upper case letters are used.

# Images
The current implementation in LangChain involves asking each parser for the text on a page, then retrieving images to apply OCR. The text extracted from images is then appended to the end of the page text, which may split paragraphs across pages, worsening the RAG model’s performance.

To avoid this, we modified the strategy for injecting OCR results from images. Now, the result is inserted between two paragraphs of text (`\n\n` or `\n`), just before the end of the page. This allows a half-paragraph to be combined with the first paragraph of the following page.

Currently, the LangChain implementation uses RapidOCR to analyze images and extract any text. This algorithm is designed to work with Chinese and English, not other languages. Since the implementation uses a function rather than a method, it’s not possible to modify it. We have modified the various parsers to allow for selecting the algorithm to analyze images. Now, it’s possible to use RapidOCR, Tesseract, or invoke a multimodal LLM to get a description of the image.

To standardize this, we propose a new abstract class:
```python
class ImagesPdfParser(BaseBlobParser):
    …
```

For converting images to text, the possible formats are: text, markdown, and HTML. Why is this important? If it’s necessary to split a result, based on the origin of the text fragments, it’s possible to do so at the level of image translations. An identification rule such as `![text](...)` or `<img …/>` allows us to identify text fragments originating from an image.

# Tables
Tables present in PDF files are another challenge. Some algorithms can detect part of them. This typically involves a specialized process, separate from the text flow. That is, the text extracted from the page includes each cell's content, sometimes in columns, sometimes in rows. This text is challenging for the LLM to interpret. Depending on the capabilities of the libraries, it may be possible to detect tables, then identify the cell boxes during text extraction to inject the table in its entirety. This way, the flow remains coherent. It’s even possible to add a few paragraphs before and after the table to prompt an LLM to describe it. Only the description of the table will be used for embedding.

Tables identified in PDF pages can be translated into markdown (if there are no merged cells) or HTML (which consumes more tokens). LLMs can then make use of them.

Unfortunately, this approach isn’t always feasible. In such cases, we can apply the approach used for images, by injecting tables and images between two paragraphs in the page’s text flow. This is always better than placing them at the end of the page.

# Combining Pages
As mentioned, we want to work with the text flow of a document, rather than by page. A mode is dedicated to this, which can be configured to specify the character to use for page delimiters in the flow. This could simply be `\n`, `------\n` or `\f` to clearly indicate a page change, or `<!-- PAGE BREAK -->` for seamless injection in a Markdown viewer without a visual effect.

Why is it important to identify page breaks when retrieving the full document flow? Because we generally want to provide a URL with the chunk’s location when the LLM answers. While it’s possible to reference the entire PDF, this isn’t practical if it’s more than two pages long. It’s better to indicate the specific page to display in the URL. Therefore, assistance is needed so that chunking algorithms can add the page metadata to each chunk. The choice of delimiter helps the algorithm prioritize this parameter.

Similarly, we’ve added metadata in all parsers with the total number of pages in the document. Why is this important? If we want to reference a document, we need to determine if it’s relevant. A reference is valid if it helps the user quickly locate the fragment within the document (using the page and/or a chunk excerpt). But if the URL points to a PDF file without a page number (for various reasons) and the file has a large number of pages, we want to remove the reference that doesn’t assist the user. There’s no point in referencing a 100-page document! The `total_pages` metadata can then be used. We recommend this approach in an extension to LangChain that we propose for managing document references: [langchain-reference](https://github.com/pprados/langchain-references).


# Compatibility
We have tried, as much as possible, to maintain compatibility with the previous version. This is reflected in preserving the order of parameters and using the default values for each implementation so that the results remain similar. The unit and integration tests for the various parsers have not been modified; they are still valid.

Ideally, we would prefer an interface like:
```python
class XXXLoader(...):
  def __init__(file_path, *, ...):
    ...
```

but this could break compatibility for positional arguments.

Perhaps it would be feasible to plan a migration for LangChain v1.0 by modifying the default parameters to make them mandatory during the transition to v1.0. At that point, we could reintroduce default values.

# New parsers
We took advantage of this refactoring to add new parsers:
- PDFRouterLoader / PDFRouterParser
- PDFMultiLoader / PDFMultiParser
- PyMuPDF4LLMLoader / PyMuPDF4LLMParser
- DoclingPDFLoader / DoclingPDFParser
- LlamaIndexPDFLoader / LlamaIndexPDFParser

We describe them in detail further on. They will be included in a another PR, after this one is approved. One PR for each new parser.

# Normalisation

The `AzureAIDocumentIntelligenceParser` class introduces the `mode` parameter, which accepts the values `single`, `page`, and `markdown`. 
The `UnstructuredPDFLoader` class introduces the `mode` parameter, which accepts the values `single`, `paged`, and `markdown`. 
Based on this model, we are extending the presence of the mode parameter to most parsers, with the value `single`, `page`, and `markdown`. 
`paged` is declared depreciated.

The different `Loader` and `BlobParser` classes now offer the following parameters:
- `file_path` `str` or `PurePath` with the file name.
- `password` str with the file password, if needed.
- `mode` to return a single document per file or one document per page (extended with `elements` in the case of Unstructured or other specific parser).
- `pages_delimiter` to specify how to join pages (`\f` by default).
- `extract_images` to enable image extraction (already present in most Loaders/Parsers).
- `images_to_text` to specify how to handle images (invoking OCR, LLM, etc.).
- `extract_tables` to allow extraction of tables detected by underlying libraries, for certain parsers.

Other parameters are specific to each parser.

The parsers compatible with image extraction are:

- `PyPDFLoader` / `PyPDFParser`
- `PyPDFium2Loader` / `PyPDFium2Parser`
- `PyPDFDirectoryLoader`
- `PyPDFMinerLoader` / `PyPDFMinerParser`
- `PyMuPDFLoader` / `PyMuPDFParser`
- `PDFPlumberLoader` / `PDFPlumberParser`
- `PyMuPDF4LLMLoader` / `PyMuPDF4LLMParser`
- `UnstructuredPDFLoader` / `UnstructuredPDFParser` (other PR)

The parsers compatible with table extraction are:

- `PyPDFDirectoryLoader` / `PyPDFDirectoryParser`
- `PyMuPDFLoader` / `PyMuPDFParser`
- `PDFPlumberLoader` / `PDFPlumberParser`
- `UnstructuredPDFLoader` / `UnstructuredPDFParser` en mode hi_res

For the `images_to_text` parameter, we propose three functions:

- `convert_images_to_text_with_rapidocr()`
- `convert_images_to_text_with_tesseract()`
- `convert_images_to_description()`

Here’s how it’s used:
```python
XXXLoader(
  file_path,
  images_to_text=convert_images_to_description(
    model=ChatOpenAI(model="gpt-4o", max_tokens=1024),
    format="markdown")
)
```

# Metadata
The different parsers offer a minimum set of common metadata:

- `source`
- `page`
- `total_page`
- `creationdate`
- `creator`
- `producer`
- and whatever additional metadata the modules can extract from PDF files. 
- Dates are converted to ISO 8601 format for easier handling and consistency with other file formats.

# New features of parsers

Here, we list all the improvements we’re bringing to each parser.

## PyPDFLoader / PyPDFParser

This parser does not support table extraction.

For this parser, we introduce the following new features:
- `mode` single or page
- `pages_delimitor`
- `images_to_text`
- Integration of image texts between two paragraphs.

## PyPDFium2Loader / PyPDFium2Parser

This parser does not support table extraction.

For this parser, we introduce the following new features:
- `password`
- `mode` single or page
- `pages_delimitor`
- `images_to_text`
- Integration of image texts between two paragraphs
- Lock, as the [code is not reentrant](https://pypdfium2.readthedocs.io/en/stable/python_api.html#thread-incompatibility)

## PyPDFDirectoryLoader

This parser does not support table extraction.

For this parser, we introduce the following new features:
- `password`
- `mode` single orou page
- `pages_delimitor`
- `extract_images`
- `images_to_text`
- Integration of image texts between two paragraphs
- Declare `@deprecated` use `langchain_community.document_loaders.generic.GenericLoader`

## PyPDFMinerLoader / PyPDFMinerParser
This parser does not support table extraction.

When the `extract_images` parameter is set to `true`, the current implementation does not respect the `concatenate_pages` parameter. It returns multiple pages instead of a single one, as specified by default. For compatibility reasons, we are keeping this behavior.

For this parser, we introduce the following new features:
- `password`
- `mode` single or page
- `pages_delimitor`
- `images_to_text`
- Integration of image texts between two paragraphs
- `concatenate_pages` is deprecated
- use really `lazy_load`
- Apply the parameter `concatenate_pages` even if `extract_images` is set to `true`

## PyMuPDFLoader / PyMuPDFParser

For this parser, we introduce the following new features:
- `password`
- `mode` single or page
- `pages_delimitor`
- `extract_images`
- `images_to_text`
- `extract_tables` to `csv`, `markdown` or `html` (without `colspan`)
- Lock, as the [code is not reentrant](https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html)

## PDFPlumberLoader / PDFPlumberParser

For this parser, we introduce the following new features:
- `password`
- `mode` single or page
- `pages_delimitor`
- `images_to_text`
- `extract_tables` to `csv`, `markdown` or `html` (with `colspan`) in the stream
- Use `lazy_load`
- Add one TU

## PagedPDFSplitter
This class is an alias for `PyPDFLoader`. I have marked it as `@deprecated`.

## OnlinePDFLoader
This class is a poorly implemented (lacking `lazy_load()`) wrapper around `UnstructuredPDFLoader`. It has been marked as `@deprecated`.

# New loader / parsers
New parsers will be introduced in a separate pull request.

## UnstructuredPDFLoader / UnstructuredPDFParser
The current implementation of `UnstructuredPDFLoader` does not provide an `UnstructuredPDFParser`, which limits its ability to handle PDF files directly from blobs. We have therefore created a new version in `langchain-unstructured`that includes a dedicated parser and offers greater flexibility. This refactoring requires modifications to both modules and has been submitted in another pull request.

For this parser, we introduce the following new features:
- `password`
- `mode` single (with conversion to markdown) or page
- `pages_delimitor`
- `extract_images`
- `images_to_text`
- `extract_tables`
- Image in the stream
- Moving integration tests to `unstructured/tests`
- Lock, as the code is not reentrant

## New parsers (in future PR)
### PDFRouterLoader
This loader relies on PDF file metadata and the content of the first page to select the appropriate PDF parser.

### PyMuPDF4LLMLoader
`Loader` compatible with new specifications, using the [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) library. Currently, the implementation does not export images for OCR/LLM processing

### DoclingPDFLoader
`Loader` to integrate docling

### LlamaIndexPDFLoader
`Loader` to integrate llamaindex

### PDFMultiLoader
Calls multiple parsers and selects the best result.


# Next PR
| Classes                  | What                                                       |
|--------------------------|------------------------------------------------------------|
| `UnstructuredPDFLoader`  | Extend unstructured to be conform to the new specification |
| `LlamaIndexPDFLoader`    | Integratio of LlamaIndex API                               |
| `PyMuPDF4LLMLoader`      | Integration of PyMuPDF4LLM                                 |
| `PDFRouterParser`        | Select the parser                                          |
| `DoclingPDFLoader`       | Use Docling                                                |
| `PDFMultiLoader`         | Use multiple parser and select the best                    |

