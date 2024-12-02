# ExtractFlora 1.0: A pipeline for transforming a flora into a database for ecological and evolutionary study

Version 1.0 of ExtractFlora pipeline. This first version is a "soft launch" to accompany Chapter 1 of Sedaghatpour, (2024).

Sedaghatpour, M. (2024). Spatial and phylogenetic insights into floristic diversity across the mediterranean Bilad al Sham [Unpublished doctoral dissertation]. University of California, Berkeley.


Important dependencies / prereqs:
- PyMuPDF 1.21.1: Python bindings for the MuPDF 1.21.1 library.
Version date: 2022-12-13 00:00:01.
Built for Python 3.10 on darwin (64-bit).
-
- pdfs:
    - NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf
    - NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2.pdf
    - NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf
    - vol2_last_8_pages.pdf
- pandas, PyMuPDF (fitz) -- version 1.21.1, matplotlib, tqdm, difflib, seaborn, unidecode, etc.

Order of running scripts:
- prereqs (above section)
- 1_Preprocessing
    - get_char_df.py
- 2_Index
    - Index.py
- 3_MainText
    - 1_find_entry_boxes.py
    - 2_entry_bbox_page_cont.py
    - 3_parsing_locations.py

To run pipeline all at once, execute run_pipeline_basic.sh or run_pipeline.sh directly from command line. Intended to be run in one calendar day. 