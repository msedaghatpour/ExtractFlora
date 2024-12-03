# ExtractFlora/MainText script 2
# parsing speices entries 
# last update by shide october 2024
# to do: where are the unidentified progress bars coming from? 

# --------------------- imports --------------------- #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cProfile import label #?not sure
import re
# from fuzzywuzzy import fuzz
import difflib
# from fuzzywuzzy import process
import time
from tqdm import tqdm
import fitz
import os
import datetime
from datetime import date # Import date class from datetime module for saving
from functools import reduce
from fitz.utils import getColor
# --------------------------------------------------- #


# Set directory 
os.chdir('/Users/maryamsedaghatpour/Desktop/extract-flora/ExtractFlora-v1.0')

# Return current local date for saving
today = date.today()
today = today.strftime("%Y%b%d") # format YYYYmonDD
print(f"today is {today}")

# %% [markdown] 
# ### importing books

# define PDF paths
vol1_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf'
vol2_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2 COMPLETE.pdf'
vol3_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf'

# open PDF 
vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)

# create list of page objects
vol1_pages = [vol1_doc[i] for i in range(vol1_doc.page_count)]
vol2_pages = [vol2_doc[i] for i in range(vol2_doc.page_count)]
vol3_pages = [vol3_doc[i] for i in range(vol3_doc.page_count)]

# find most recent output folder
output_folders = [folder for folder in os.listdir(".") # Find the all folders that start with output
               if folder.startswith("output") and folder != "output"]
# print(output_folders)

output_folders.sort(                   # Sort folders by date, descending order
    key=lambda x: datetime.datetime.strptime(x[6:],"%Y%b%d"),
    reverse=True)
output_dir = output_folders[0]         # clean
output_dir = "output2024Dec02"
print(f"getting data from {output_dir}")

# %%
# load index
vol1_index_path = f'{output_dir}/index_output/vol1_nonitalics.csv'
vol2_index_path = f'{output_dir}/index_output/vol2_nonitalics.csv'
vol3_index_path = f'{output_dir}/index_output/vol3_nonitalics.csv'

vol1_index_df = pd.read_csv(vol1_index_path)
vol2_index_df = pd.read_csv(vol2_index_path)
vol3_index_df = pd.read_csv(vol3_index_path)

# rename columns of mout. indecies 
vol1_index_df.rename(columns={'closest_genus': 'mouterde_genus', 'closest_epithet': 'mouterde_epithet', 'authors':'mouterde_author', 'closest_infra_name':'mouterde_infra'}, inplace=True)
vol2_index_df.rename(columns={'closest_genus': 'mouterde_genus', 'closest_epithet': 'mouterde_epithet', 'authors':'mouterde_author', 'closest_infra_name':'mouterde_infra'}, inplace=True)
vol3_index_df.rename(columns={'closest_genus': 'mouterde_genus', 'closest_epithet': 'mouterde_epithet', 'authors':'mouterde_author', 'closest_infra_name':'mouterde_infra'}, inplace=True)


# %%
# load preprocessed df 
vol1_char_df = pd.read_pickle(f"{output_dir}/char_df/vol1_df.pkl")
vol2_char_df = pd.read_pickle(f"{output_dir}/char_df/vol2_df.pkl")
vol3_char_df = pd.read_pickle(f"{output_dir}/char_df/vol3_df.pkl")

# create new df from preprocessed df
vol1_word_df = vol1_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox']].drop_duplicates()

vol2_word_df = vol2_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox']].drop_duplicates()

vol3_word_df = vol3_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox']].drop_duplicates()

# %%
# reset doc:
vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)

# extract genera from index
vol1_genera = vol1_index_df[vol1_index_df['taxon_rank'] == 'genus']['mouterde_genus'].str.upper().tolist()
vol2_genera = vol2_index_df[vol2_index_df['taxon_rank'] == 'genus']['mouterde_genus'].str.upper().tolist()
vol3_genera = vol3_index_df[vol3_index_df['taxon_rank'] == 'genus']['mouterde_genus'].str.upper().tolist()


# %%
# --------------------- processing functions --------------------- #
def process_volume(volume, volume_index_df, volume_char_df, volume_doc, volume_genera):
    print(volume)
    # %% [markdown]
    # ### getting word pairs

    # list of genera from index -- uppercased to match main text pattern
    # is this to find genus descriptions? 
    volume_genera = volume_index_df[volume_index_df['taxon_rank'] == 'genus']['mouterde_genus'].str.upper().tolist()

    # list of species binomial from main text
    volume_species_temp_df = volume_index_df[(volume_index_df['taxon_rank'] == 'epithet') & (~volume_index_df['mouterde_genus'].isna())]
    volume_species_binomial_list = list(zip(volume_species_temp_df['mouterde_genus'], volume_species_temp_df['mouterde_epithet']))
    volume_species = list(map(lambda x: f"{x[0]} {x[1]}", volume_species_binomial_list))
    volume_species_abriviation = list(map(lambda x: f"{x[0][0]}. {x[1]}", volume_species_binomial_list))

    # %%
    def is_italic(flags):
        return flags & 2 ** 1 != 0

    # %%
    tqdm.pandas()

    # %%
    # -----------------------find binomial lines-------------------------- #
    # data from 1_find_entry_boxes
    volume_word_df = pd.read_pickle(f"{output_dir}/desc_box_df/{volume}_entry_df.pkl").loc[:, ['vol_num', 'page_num', 'block_num', 'block_num_absolute', 'block_bbox',
                                                                                          'line_num', 'line_wmode', 'line_dir', 'line_bbox', 'span_num',
                                                                                          'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender',
                                                                                          'span_descender', 'span_origin', 'span_bbox', 'word_num', 'word',
                                                                                          'word_bbox', 'pruned_word', 'pruned_word_bbox', '1_words', '1_flags',
                                                                                          '1_words_match', '1_words_match_score', '2_words', '2_flags',
                                                                                          '2_words_match', '2_words_match_score', '3_words', '3_flags',
                                                                                          '3_words_match', '3_words_match_score', '4_words', '4_flags',
                                                                                          '4_words_match', '4_words_match_score']]
    # new df with likely binomials
    # conditions: not italicized and greater than 0.85 score
    likely_results = volume_word_df[((~(volume_word_df['1_flags'].apply(is_italic)) &
                                      (volume_word_df['1_words_match_score'] > 0.85)) |
                                     (~(volume_word_df['2_flags'].apply(is_italic)) &
                                      (volume_word_df['2_words_match_score'] > 0.85)) |
                                     (~(volume_word_df['3_flags'].apply(is_italic)) &
                                      (volume_word_df['3_words_match_score'] > 0.85)) |
                                     (~(volume_word_df['1_flags'].apply(is_italic)) &  # duplicate
                                      (volume_word_df['1_words_match_score'] > 0.85))) &
                                    (volume_word_df['page_num'] < 616)]

    # add new column, line_id: tuple with the page, block, line numbers for each row 
    volume_word_df['line_id'] = volume_word_df.progress_apply(lambda r: (r['page_num'], r['block_num'], r['line_num']), axis=1)
    # --------------------------------------------------------------------- #


    # %%
    # ----------------------remove title lines---------------------------- #
    is_page_title = volume_word_df.groupby('line_id')['word'].transform(lambda x: x.isin(['NOUVELLE', 'FLORE']).any())
    volume_word_df['is_page_title'] = is_page_title

    title_df = volume_char_df.loc[volume_char_df.groupby(['page_num', 'block_num', 'line_num'])['word'].transform(lambda x: x.isin(['NOUVELLE', 'FLORE']).any())]
    title_char_height = title_df.groupby(['page_num', 'block_num', 'line_num'])['char_bbox'].transform(lambda x: x.apply(lambda y: y[3] - y[1])).mean()

    def get_page_title_mean_y(row):
        if row['is_page_title']:
            return (row['line_bbox'][1] + row['line_bbox'][3]) / 2
        else:
            return np.nan

    volume_word_df['page_title_mean_y'] = volume_word_df.progress_apply(get_page_title_mean_y, axis=1)

    num_pages = volume_word_df['page_num'].max() + 1
    for page_num in tqdm(range(num_pages), desc = "finding and removing book title lines"):
        volume_word_df.loc[volume_word_df['page_num'] == page_num, 'page_title_mean_y'] = volume_word_df.loc[(volume_word_df['page_num'] == page_num) & 
                                                                                                         (volume_word_df['is_page_title'] == True), 'page_title_mean_y'].mean()

    def is_title_line(row):
        if abs(row['page_title_mean_y'] - ((row['line_bbox'][1] + row['line_bbox'][3]) / 2)) < (title_char_height / 2):
            return True
        else:
            return False

    volume_word_df['is_title_line'] = volume_word_df.progress_apply(is_title_line, axis=1)
    volume_word_df = volume_word_df[volume_word_df['is_title_line'] == False].copy()
    # ------------------------------------------------------------------- #

    # %%
    # -----------------Determine the end of a species entries--------------------- #
    # identify genus descriptions
        # also identifies family desc?
    # keep rows where the 'word' column matches any of the genera in volume_genera
    genus_parts = volume_word_df[(volume_word_df['word'].isin(volume_genera))]
    middle_uppers = volume_word_df[(volume_word_df['line_bbox'].apply(lambda x: x[0] > 120)) &  # entirely uppercase
                                   (volume_word_df['word'].str.isupper())]
    
    # identify section breaks 
    # duplicated from script 1: likely_results
    possible_stops = volume_word_df[((volume_word_df['word'].isin(volume_genera))) |
                                    ((volume_word_df['line_bbox'].apply(lambda x: x[0] > 120)) &
                                     (volume_word_df['word'].str.isupper()) &
                                     (volume_word_df['pruned_word'].apply(len) > 2))]
    # move this code to above title code? 
    # identifies binomial rows as section break 
    # duplicated from script 1
    is_binomial = ((~(volume_word_df['1_flags'].apply(is_italic)) & 
                    (volume_word_df['1_words_match_score'] > 0.85)) |
                   (~(volume_word_df['2_flags'].apply(is_italic)) & 
                    (volume_word_df['2_words_match_score'] > 0.85)) |
                   (~(volume_word_df['3_flags'].apply(is_italic)) & 
                    (volume_word_df['3_words_match_score'] > 0.85)) |
                   (~(volume_word_df['1_flags'].apply(is_italic)) & 
                    (volume_word_df['1_words_match_score'] > 0.85)))

    # identical to possible_stops above
    # also duplicated from script 1
    is_stop = (((volume_word_df['word'].isin(volume_genera))) |
               ((volume_word_df['line_bbox'].apply(lambda x: x[0] > 120)) &
                (volume_word_df['word'].str.isupper()) &
                (volume_word_df['pruned_word'].apply(len) > 2)))
    
    # Create unique ID: tuples of page, block, line numbers of potential section breaks.
    # repeated from script 1
    break_page_num =  volume_word_df[(is_binomial) | (is_stop)]['page_num']
    break_block_num = volume_word_df[(is_binomial) | (is_stop)]['block_num']
    break_line_num =  volume_word_df[(is_binomial) | (is_stop)]['line_num']
    break_id = list(zip(break_page_num, break_block_num, break_line_num))

    # new boolean column: species entry break or no
    volume_word_df['section_break'] = volume_word_df['line_id'].isin(break_id)
    # ------------------------------------------------------------------------------ #


    # ---------assign section IDs, ie a unqie ID to each species entry-------------- # 
    def get_section_id(row):
        if row['section_break']:
            return row['line_id']
        else:
            return np.nan
        
    # new col for section_id    
    volume_word_df['section_id'] = volume_word_df.progress_apply(get_section_id, axis=1)

    # fill section IDs in rows that belong to the same section
    volume_word_df['section_id'].ffill(inplace=True)

    # Extracts y-coordinate of the top edge of the bounding box for a section (species entry).
    def get_section_y(row):
        if row['section_break']:
            return row['line_bbox'][1]
        else:
            return np.nan
        
    # new col for section start    
    volume_word_df['section_start_y'] = volume_word_df.progress_apply(get_section_y, axis=1)

    # fill in rows that belong to the same section
    volume_word_df['section_start_y'].ffill(inplace=True)
    # ------------------------------------------------------------------------------ #


    # ------------------------------data wrangling to get bboxes------------------------- #
    # new cols for x0, y0, x1, y1 coordinaetes of species entries
    # The lambda function extracts the corresponding coordinate from the tuple and assigns it to the new column.
    volume_word_df['line_x0'] = volume_word_df["line_bbox"].apply(lambda x: x[0])
    volume_word_df['line_y0'] = volume_word_df["line_bbox"].apply(lambda x: x[1])
    volume_word_df['line_x1'] = volume_word_df["line_bbox"].apply(lambda x: x[2])
    volume_word_df['line_y1'] = volume_word_df["line_bbox"].apply(lambda x: x[3])
    
    # new cols for x0, y0, x1, y1 coordinaetes of species entries 
    volume_word_df["section_x0"] = volume_word_df.groupby(['page_num', 'section_id'])['line_x0'].transform('min')
    volume_word_df["section_y0"] = volume_word_df.groupby(['page_num', 'section_id'])['line_y0'].transform('min')
    volume_word_df["section_x1"] = volume_word_df.groupby(['page_num', 'section_id'])['line_x1'].transform('max')
    volume_word_df["section_y1"] = volume_word_df.groupby(['page_num', 'section_id'])['line_y1'].transform('max')

     # new col: section bbox
    volume_word_df["section_bbox"] = volume_word_df.apply(lambda r: (r["section_x0"], r["section_y0"], r["section_x1"], r["section_y1"]), axis=1)
    # drop unnecessary columns 
    volume_word_df.drop(columns=["line_x0", "line_y0", "line_x1", "line_y1", "section_x0", "section_y0", "section_x1", "section_y1"], inplace=True)

    # identify and mark lines that are likely part of binomial names,
    binom_page_num = volume_word_df[(is_binomial)]['page_num']
    binom_block_num = volume_word_df[(is_binomial)]['block_num']
    binom_line_num = volume_word_df[(is_binomial)]['line_num']
    binom_id = list(zip(binom_page_num, binom_block_num, binom_line_num))
    volume_word_df['binom_section'] = volume_word_df['line_id'].isin(binom_id)
    # ------------------------------------------------------------------------------ #


    # ----------------------------iterate--------------------------- #
    # iterates through each page and highlight sections containing potential binomial
    for page_num in tqdm(range(num_pages), desc = "finding binomials"):
        section_groups = volume_word_df[volume_word_df['page_num'] == page_num].groupby('section_id') # Grouping by Section
        for name, section in section_groups:           # Processing Each Section
            page = volume_doc[page_num]                # Retrieves the current page
            section_id = section.iloc[0]['section_id']      # Extracts the section ID
            section_bbox = section.iloc[0]['section_bbox']  # Extracts the bounding box coordinates
    # Highlighting Binomial Sections
            if section_id in binom_id:
                r_box = fitz.Rect(section_bbox)
                annot_rect = page.add_rect_annot(r_box)
                annot_rect.update()
                
    print("saving PDF")
    if not os.path.exists(f"output{today}/main_text"):
        os.makedirs(f"output{today}/main_text")
    marked_epithet_fname = f"output{today}/main_text/{volume}_binom_sections_new_page_cont_v4.pdf"
    volume_doc.save(marked_epithet_fname)

    print("saving pickle file")
    if not os.path.exists(f"output{today}/desc_box_df"):
        os.makedirs(f"output{today}/desc_box_df")
    volume_word_df.to_pickle(f"output{today}/desc_box_df/{volume}_desc_df_v2.pkl")

# %% 
print("Processing Volume 1")
process_volume("vol1", vol1_index_df, vol1_char_df, vol1_doc, vol1_genera)
print("VOL 1 COMPLETE")
print("\n\n\nProcessing Volume 2")
process_volume("vol2", vol2_index_df, vol2_char_df, vol2_doc, vol2_genera)
print("VOL 2 COMPLETE")
print("\n\n\nProcessing Volume 3")
process_volume("vol3", vol3_index_df, vol3_char_df, vol3_doc, vol3_genera) 
print("VOL 3 COMPLETE")
print("THE END: Proceed to next script")
