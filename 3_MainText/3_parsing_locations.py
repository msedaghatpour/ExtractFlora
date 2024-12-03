# -*- coding: utf-8 -*-
# ExtractFlora/MainText script 3
# parsing within sp entry
# October 2024 update

# --------------------- imports --------------------- #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cProfile import label #?not sure
import re
import difflib 
import time
from tqdm import tqdm
import fitz
import os
from PIL import Image
from functools import reduce
from fitz.utils import getColor # for bbox annotations. color options: https://pymupdf.readthedocs.io/en/latest/colors.html 
import datetime
from datetime import date # Import date class from datetime module for saving
from unidecode import unidecode
# --------------------------------------------------- #

# Set directory 
os.chdir('/Users/maryamsedaghatpour/Desktop/extract-flora/ExtractFlora-v1.0')

# Return current local date for saving
today = date.today()
today = today.strftime("%Y%b%d") # format YYYYmonDD
print(f"today is {today}")

# ------------------------ IMPORT VOLUMES ------------------------ #
# define paths
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

# # find output folder
# # Find the all folders that start with output
# output_folders = [folder for folder in os.listdir(".") 
#                if folder.startswith("output") and folder != "output"]
# # print(output_folders)

# output_folders.sort(                   # Sort folders by date, descending order
#     key=lambda x: datetime.datetime.strptime(x[6:],"%Y%b%d"),
#     reverse=True)
# output_dir = output_folders[0]         # clean
output_dir = "output2024Dec02"
print(f"getting data from {output_dir}")

# load preprocessed df 
vol1_char_df = pd.read_pickle(f"{output_dir}/char_df/vol1_df.pkl")
vol2_char_df = pd.read_pickle(f"{output_dir}/char_df/vol2_df.pkl")
vol3_char_df = pd.read_pickle(f"{output_dir}/char_df/vol3_df.pkl")

# %%
# index: 
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
# # ---------------------------------------------------------------- #

# create new dfs from preprocessed df
vol1_word_df = vol1_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox','char_bbox']].drop_duplicates()

vol2_word_df = vol2_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox','char_bbox']].drop_duplicates()

vol3_word_df = vol3_char_df.loc[:, ['vol_num', 'page_num', 
                                    'block_num', 'block_num_absolute', 'block_bbox',
                                    'line_num', 'line_wmode', 'line_dir', 'line_bbox', 
                                    'span_num', 'span_size', 'span_flags', 'span_font', 'span_color', 'span_ascender', 'span_descender', 'span_origin', 'span_bbox', 
                                    'word_num', 'word','word_bbox', 'pruned_word', 'pruned_word_bbox','char_bbox']].drop_duplicates()
# %%
# reset doc:
vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)


# %%
# --------------------- processing functions --------------------- #
def process_volume(volume, volume_index_df, volume_word_df, volume_char_df, volume_doc):
    vol_genera = volume_index_df[volume_index_df['taxon_rank'] == 'genus']['mouterde_genus'].str.upper().tolist()

    #list of species binomial from main text
    vol_species_temp_df = volume_index_df[(volume_index_df['taxon_rank'] == 'epithet') & 
                                          (~volume_index_df['mouterde_genus'].isna())]
    vol_species_binomial_list = list(zip(vol_species_temp_df['mouterde_genus'], vol_species_temp_df['mouterde_epithet']))
    vol_species = list(map(lambda x: f"{x[0]} {x[1]}", vol_species_binomial_list))
    vol_species_abriviation = list(map(lambda x: f"{x[0][0]}. {x[1]}", vol_species_binomial_list))

    # %%
    def is_italic(flags):
        return flags & 2 ** 1 != 0
# ---------------------------------------------------------------- #

    # %%
    tqdm.pandas()

    # %%
    # read in data per volume
    volume_word_df = pd.read_pickle(f"output2024Dec03/desc_box_df/{volume}_desc_df_v2.pkl")


    # %%
    #############
    ###### move to script 2 ?
    # determine binomial lines again
    is_binomial = ((~(volume_word_df['1_flags'].apply(is_italic)) & 
                    (volume_word_df['1_words_match_score'] > 0.85)) | 
                (~(volume_word_df['2_flags'].apply(is_italic)) & 
                 (volume_word_df['2_words_match_score'] > 0.85)) | 
                (~(volume_word_df['3_flags'].apply(is_italic)) & 
                 (volume_word_df['3_words_match_score'] > 0.85)) # | 
                # (~(volume_word_df['1_flags'].apply(is_italic)) & 
                #  (volume_word_df['1_words_match_score'] > 0.85))
                 ) 
    # create tuple 
    binom_page_num = volume_word_df[(is_binomial)]['page_num']
    binom_block_num = volume_word_df[(is_binomial)]['block_num']
    binom_line_num = volume_word_df[(is_binomial)]['line_num']
    binom_id = list(zip(binom_page_num, binom_block_num, binom_line_num))

    # %%
    # remove non-binomial lines
    volume_char_df['line_id'] = volume_char_df.apply(lambda r : (r['page_num'], 
                                                                 r['block_num'], 
                                                                 r['line_num']), 
                                                                 axis = 1)
    volume_char_binom_df = volume_char_df[volume_char_df['line_id'].isin(binom_id)]

    # %%
    # Calculate Binomial Character Width
    binom_char_width = volume_char_binom_df.groupby('line_id')['char_bbox']\
        .transform(lambda x: x.apply(lambda y: y[2] - y[0])).mean()
    binom_char_width

    # %%
    # For each page, calculates mean x-coordinate of first character in each binomial line
    num_pages = volume_word_df['page_num'].max() + 1
    for page_num in tqdm(range(num_pages), desc = "calculating mean x-coord..."):
        volume_word_df.loc[volume_word_df['page_num'] == page_num, 'mean_binom_x0'] = volume_word_df[(volume_word_df['page_num'] == page_num) & 
                                                                                                     (volume_word_df['line_id'].isin(binom_id))]['line_bbox'].apply(lambda x : x[0]).mean()

    # %%
    # threshold 
    accepted_error = (binom_char_width)*2.5 # eyeballing it ...

    def is_binom_indent(row):
        if abs(row['mean_binom_x0'] - (row['line_bbox'][0])) < accepted_error:
            return True
        else:
            return False

    # applying is_binom_indent: TRUE or FALSE
    volume_word_df['is_binom_indent'] = volume_word_df.progress_apply(is_binom_indent, axis = 1)

    # %% view data
    # volume_word_df['section_break']

    # # %% view
    # volume_word_df['entry_id'] = np.nan
    # %%
    def entry_id(row):
        if row['is_binom_indent'] == True:
            return row['line_id']
        if row['section_break'] == True:
            return row['line_id']
        else:
            return np.nan
##### end move to script 2?
##############


    volume_word_df['entry_id'] = volume_word_df.progress_apply(entry_id, axis = 1)
    volume_word_df['entry_id'].ffill(inplace=True)
    # FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    # The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    # For example, 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, 
    # to perform the operation inplace on the original object.
    # current version: pandas 2.2.3


    # %%
    # break down line coords
    volume_word_df['line_x0'] = volume_word_df["line_bbox"].apply(lambda x: x[0])
    volume_word_df['line_y0'] = volume_word_df["line_bbox"].apply(lambda x: x[1])
    volume_word_df['line_x1'] = volume_word_df["line_bbox"].apply(lambda x: x[2])
    volume_word_df['line_y1'] = volume_word_df["line_bbox"].apply(lambda x: x[3])

    #sections_coords: 
    volume_word_df["paragraph_x0"] = volume_word_df.groupby(['page_num', 'entry_id'])['line_x0'].transform('min')
    volume_word_df["paragraph_y0"] = volume_word_df.groupby(['page_num', 'entry_id'])['line_y0'].transform('min')
    # volume_word_df["section_y0"] = vol1volume_word_df_word_df[['section_y0_all','section_y0_all']].max(axis=1)
    volume_word_df["paragraph_x1"] = volume_word_df.groupby(['page_num', 'entry_id'])['line_x1'].transform('max')
    volume_word_df["paragraph_y1"] = volume_word_df.groupby(['page_num', 'entry_id'])['line_y1'].transform('max')

    #section_bbox:
    volume_word_df["paragraph_bbox"] = volume_word_df.apply(lambda r: (r["paragraph_x0"], 
                                                                       r["paragraph_y0"], 
                                                                       r["paragraph_x1"], 
                                                                       r["paragraph_y1"]), 
                                                                       axis = 1)

    #drop extra cols:
    volume_word_df.drop(columns= ["line_x0", "line_y0", "line_x1", "line_y1", 
                                  "paragraph_x0", "paragraph_y0", "paragraph_x1", "paragraph_y1"], 
                                  inplace = True)

    # %% view 
    # volume_word_df.columns

    # %% view
    # volume_word_df['pruned_word'] 

    # %%
    # Create a new image with a solid color
    width = 50
    height = 50
    colors = [getColor("pink"), getColor("lightblue"),  
              getColor("lightgray"), getColor("lightgreen"), 
              getColor("violetred4"), getColor("plum"), 
              getColor("orchid4"), getColor("orange"), getColor("darkgreen"), ]
    for color_1 in colors:
        color_255 = tuple(int(c_val*255) for c_val in color_1) # red color
        image = Image.new("RGB", (width, height), color_255)
        image.show()
    # Display the image (print temp.png)
    # image
 

    # %%
    # calculate similarity scores of normalized text
    def closest_norm_match(input_str, match_str):
        if isinstance(match_str, str):
            input_str = unidecode("".join([c for c in input_str if c.isalnum()]))
            match_str = unidecode("".join([c for c in match_str if c.isalnum()]))
            score = difflib.SequenceMatcher(None, input_str.lower(), match_str.lower()).ratio()
            return score
        else:
            return np.nan

    # %%
    # title: set up for long loop
    # set up dictionary for specific localitites 
    # create a group for each paragraph.
    paragraph_groups = volume_word_df.groupby('entry_id')
    volume_word_df['paragraph_word_num'] = paragraph_groups.cumcount() + 1
    paragraph_groups_123 = volume_word_df[volume_word_df['page_num'] == 123].groupby('entry_id')
    sub_loc_dict = {}
    other_data = {}

    # title: LONG LOOP
    # Identify Italicized Words
    for name, paragraph in  tqdm(paragraph_groups, desc = "parsing species entries..."):
        paragraph_italics_list = (paragraph[paragraph['span_flags'].apply(is_italic)]['paragraph_word_num'] -1).tolist()
        paragraph_italics_list.append((paragraph['paragraph_word_num']).max())

        paragraph_text = paragraph['word'].tolist()   # convert to list
        no_italics_line = False                       # flag if first word is italic 
        if paragraph_italics_list[0] != 1:
            paragraph_italics_list = [0] + paragraph_italics_list
            no_italics_line = True
            
        # iterate over pairs of consecutive italicized words    
        for i in range(len(paragraph_italics_list) - 1):
            curr_word_i = int(paragraph_italics_list[i])    # store current word index
            next_word_i = int(paragraph_italics_list[i+1])  # store next word index. #only works if the last word of L. is not italics. last sub-location might not extract correctly.
            sub_loc_italics = paragraph_text[curr_word_i]   # Extract italicized word 
            sub_location_list = paragraph_text[curr_word_i + 1: next_word_i]   # extract sub locality
            sub_location_str = " ".join(sub_location_list)  # join specific localities into a string
            # make dict for L. / S. whenever it exists.
            
            sub_loc_result = [match_s.strip() for match_s in re.findall(r'([^,()]+?)(?=[,(])(?![^()]*\))', sub_location_str)]

            # extract non-italisized text (specific locals)
            if no_italics_line:
                sub_loc_italics = 'NO ITALICS'
                sub_location_list = paragraph_text[curr_word_i: next_word_i] 
                sub_loc_result = [" ".join(sub_location_list)]
                    
            paragraph_section_id = paragraph['section_id'].iloc[0] # extract section_id from first row of paragraph
            # check if paragraph_section_id exists as a key in sub_loc_dict dictionary.
            # If it doesn't exist, create new empty dictionary for that key. 
            try:
                sub_loc_dict[paragraph_section_id]
            except:
                sub_loc_dict[paragraph_section_id] = {}
            
            try:
                other_data[paragraph_section_id]
            except:
                other_data[paragraph_section_id] = {}
            
            aire_geogr_match = closest_norm_match(" ".join(paragraph_text[:2]), "aire géogr.") > 0.9 # 90% match
            L_match = paragraph_text[0].lower() in ['l.', 'l']       # does paragraph start with L?
            S_match = paragraph_text[0].lower() in ['s.', 's']      # does paragraph start with S?
            floraison_match = closest_norm_match(paragraph_text[0], "Floraison:") > 0.9
            is_description = paragraph_section_id == name             # for sp description
            if L_match:
                try: 
                    sub_loc_dict[paragraph_section_id]['L.']
                except: 
                    sub_loc_dict[paragraph_section_id]['L.'] = {}
                
                try:
                    sub_loc_dict[paragraph_section_id]['L.'][sub_loc_italics]
                except:
                    sub_loc_dict[paragraph_section_id]['L.'][sub_loc_italics] = []
                sub_loc_dict[paragraph_section_id]['L.'][sub_loc_italics].extend(sub_loc_result)
            
            elif S_match:
                try: 
                    sub_loc_dict[paragraph_section_id]['S.']
                except: 
                    sub_loc_dict[paragraph_section_id]['S.'] = {}

                try:
                    sub_loc_dict[paragraph_section_id]['S.'][sub_loc_italics]
                except:
                    sub_loc_dict[paragraph_section_id]['S.'][sub_loc_italics] = []
                sub_loc_dict[paragraph_section_id]['S.'][sub_loc_italics].extend(sub_loc_result)

            elif aire_geogr_match:
                other_data[paragraph_section_id]["aire géogr."] = " ".join(paragraph_text[2:])
            elif floraison_match:
                other_data[paragraph_section_id]["Floraison"] = " ".join(paragraph_text[1:])
            
            elif is_description:
                other_data[paragraph_section_id]["desc_paragraph"] = " ".join(paragraph_text)
            else:
                try:
                    other_data[paragraph_section_id]["other"]
                except:
                    other_data[paragraph_section_id]["other"] = [] 
                other_data[paragraph_section_id]["other"].append([" ".join(paragraph_text)])
    # end of LONG LOOP
    print("parsing species entries complete")

    
    # %% identify binomials again (remove?)
    # is_binomial = ((~(volume_word_df['1_flags'].apply(is_italic)) & 
    #                 (volume_word_df['1_words_match_score'] > 0.85)) | 
    #             (~(volume_word_df['2_flags'].apply(is_italic)) & 
    #              (volume_word_df['2_words_match_score'] > 0.85)) | 
    #             (~(volume_word_df['3_flags'].apply(is_italic)) & 
    #              (volume_word_df['3_words_match_score'] > 0.85)) | 
    #             (~(volume_word_df['1_flags'].apply(is_italic)) & 
    #              (volume_word_df['1_words_match_score'] > 0.85))) 

    # %% remove function? replaced with get_binomial() below.
    def get_binomial_string(row):
        all_combos = []
        for i in range(1, 5):
            all_combos.append((row[f'{i}_flags'], row[f'{i}_words'], row[f'{i}_words_match_score']))
        non_italics_combs = [comb for comb in all_combos if is_italic(comb[0])]
        if non_italics_combs:
            binom_name = max(non_italics_combs, key = lambda x : x[2])[1] #not using closest match because that sometimes isn't right
            return binom_name 
        else: 
            return np.nan

    # vol1_word_df.apply(get_binomial_string, axis = 1).groupby('section_id').transform('max')
    volume_word_df['binom_string'] = volume_word_df.apply(get_binomial_string, axis = 1)

    # %% identify binomials again (remove?)
    # is_binomial = ((~(volume_word_df['1_flags'].apply(is_italic)) & 
    #                 (volume_word_df['1_words_match_score'] > 0.85)) | 
    #             (~(volume_word_df['2_flags'].apply(is_italic)) & 
    #              (volume_word_df['2_words_match_score'] > 0.85)) | 
    #             (~(volume_word_df['3_flags'].apply(is_italic)) & 
    #              (volume_word_df['3_words_match_score'] > 0.85)) | 
    #             (~(volume_word_df['1_flags'].apply(is_italic)) & 
    #              (volume_word_df['1_words_match_score'] > 0.85))) 
    

    # %% VIEW DATA TO CONFIRM
    # view
    # volume_word_df[is_binomial]

    # # %% view filter df page number between 78 and 606 (specific to vol 1)
    # volume_word_df[(is_binomial==True) & 
    #                (volume_word_df['page_num'] >= 78) & 
    #                (volume_word_df['page_num'] <= 606)]

    # # %% view
    # len(volume_word_df[(is_binomial==True) & 
    #                    (volume_word_df['page_num'] >= 78) & 
    #                    (volume_word_df['page_num'] <= 606)].groupby('section_id'))

    # # %% view
    # volume_word_df[(volume_word_df['binom_section']) &  
    #                (volume_word_df['page_num'] >= 78) & 
    #                (volume_word_df['page_num'] <= 606)].groupby('section_id').first()['binom_string']

    # # %% view columns
    # volume_word_df.columns

    # %%
    section_groups = volume_word_df.groupby('section_id')

    # %% Aarons method?
    # why is this down here? 
    def get_binomial(row):
        combs = [(row['1_words'], row['1_flags'], row['1_words_match_score']),
                (row['2_words'], row['2_flags'], row['2_words_match_score']),
                (row['3_words'], row['3_flags'], row['3_words_match_score']),
                (row['4_words'], row['4_flags'], row['4_words_match_score'])]
        combs_valid = [c for c in combs if (is_italic(c[1]) == False and np.isnan(c[2]) == False)]
        return max(combs_valid, key = lambda x: x[2])[0]

    # %%
    # group rows with page_num less than or equal to 609 from volume_word_df based on section_id
    section_groups = volume_word_df[volume_word_df['page_num']<=609].groupby('section_id')
    fake_span = ''
    fake_line = ''
    fake_block = ''
    fake_warning = ''
    items = []
    boxes = []

    # %% view
    # volume_word_df['binom_bbox'] = np.nan
    
    # get bbox coords?
    for name, section in tqdm(section_groups, desc= "getting bboxes..."):
        page_num = int(name[0])
        section_id = section.iloc[0]['section_id']
        if section_id in binom_id:
            section_bbox = section.iloc[0]['section_bbox']
            desc_rect = section_bbox
            binom_section = section[((~(section['1_flags'].apply(is_italic)) &  # repeated again. make into function? 
                                      (section['1_words_match_score'] > 0.85)) | 
                                    (~(section['2_flags'].apply(is_italic)) & 
                                     (section['2_words_match_score'] > 0.85)) | 
                                    (~(section['3_flags'].apply(is_italic)) & 
                                     (section['3_words_match_score'] > 0.85)))] # | 
                                    # (~(section['1_flags'].apply(is_italic)) &  # duplicate
                                    #  (section['1_words_match_score'] > 0.85)))]
                                    
            binom = binom_section.apply(get_binomial, axis = 1).iloc[0]
            binom = "".join(binom)
            num_binom_words = len(binom.split(' '))
            binom_index = []
            #probably not best way of doing this
            start_binom_index = binom_section.index[0]
            start_binom_i = list(section.index).index(start_binom_index)
            for i in range(num_binom_words):
                section.index
                binom_index.append(section.index[start_binom_i])
            binom_x0 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[0]).min()
            binom_y0 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[1]).min()
            binom_x1 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[2]).max()
            binom_y1 = section.loc[binom_index, 'word_bbox'].apply(lambda x : x[3]).max()
            binom_rect = fitz.Rect((binom_x0, binom_y0, binom_x1, binom_y1)) 

            # # variables not used downstream. remove? 
            # span_num = binom_section.iloc[0]['span_num']
            # line_num = binom_section.iloc[0]['line_num']
            # block_num = binom_section.iloc[0]['block_num']

            section_shape = volume_word_df.loc[volume_word_df['section_id'] == section_id].shape[0]
            volume_word_df.loc[volume_word_df['section_id'] == section_id, 'binom_name'] = binom
            volume_word_df.loc[(volume_word_df['section_id'] == section_id), 'binom_bbox'] = \
                volume_word_df.loc[(volume_word_df['section_id'] == section_id)].apply(
                lambda _ : (binom_x0, binom_y0, binom_x1, binom_y1), 
                axis = 1)
            # this line was not working the normal way ... seems fine now
            # prob something to do with this: "FutureWarning: Setting an item of incompatible dtype 
            # is deprecated and will raise an error in a future version of pandas. 
            # Value '[(np.float64(179.96920776367188), ... "
    print("bboxes obtained")
    

   # %%
    # moved from above
    # render PDF of parsed spp entries
    for page_num in tqdm(range(num_pages), desc = "rendering PDF..."):
        section_groups = volume_word_df[volume_word_df['page_num'] == page_num].groupby('section_id')
        page = volume_doc[page_num]
        colors = [getColor("plum"), getColor("orchid4")]
        
        paragraph_groups = volume_word_df[volume_word_df['page_num'] == page_num].groupby('entry_id')
        for name, paragraph in paragraph_groups:
            i = 0
            entry_id = paragraph.iloc[0]['entry_id']
            paragraph_section_id = paragraph.iloc[0]['section_id']
            paragraph_bbox = paragraph.iloc[0]['paragraph_bbox']
            is_L_loc = paragraph.iloc[0]['word'].lower() in ["l.", "l"]
            is_S_loc = paragraph.iloc[0]['word'].lower() in ["s.", "s"] # is first word "s." or "s".
            is_Floraison = paragraph.iloc[0]['word'] # take out function, add [word]
            is_aire_geor = " ".join(paragraph.iloc[:2]['word'].tolist()) 
            is_description = paragraph_section_id == name     # sp description. exact code from parsing


            c = getColor("lightgray")         # default color 
            if is_L_loc: 
                c = getColor("lightblue")
            if is_S_loc:
                c = getColor("pink")
            if closest_norm_match(is_Floraison, "Floraison:") > 0.9:
                c = getColor("plum")
            if closest_norm_match(is_aire_geor, "aire géogr.") > 0.9:
                c = getColor("lightgreen")
            if is_description:
                c = getColor("darkgreen")

            # add rectangles to binomials? 
            if paragraph_section_id in binom_id:
                r_box = fitz.Rect(paragraph_bbox)
                annot_rect = page.add_rect_annot(r_box)
                annot_rect.set_colors({"stroke":c})
                annot_rect.update()
                i += 1
            c = getColor("lightgray")            # reset color? 

        for name, section in section_groups:
            section_id = section.iloc[0]['section_id']
            section_bbox = section.iloc[0]['section_bbox']
            if section_id in binom_id:
                r_box = fitz.Rect(section_bbox)
                #r_box.set_stroke_color(stroke=getColor("violetred4"))
                annot_rect = page.add_rect_annot(r_box)
                annot_rect.set_colors({"stroke": getColor("violetred4")})
                annot_rect.update()

    if not os.path.exists(f"output{today}/main_text"):
        os.makedirs(f"output{today}/main_text")                    
    marked_epithet_fname = f"output{today}/main_text/{volume}_binom_sections_paragraphs_LS_v1.pdf"
    volume_doc.save(marked_epithet_fname)
    print("PDF saved")

    # %% create nested dict
    dict_test = {'L.': {'Ctlitt.': ['Plage de Khaldé'],
            'Ct.':     ['Beyrouth', 'adventice en pleine ville']},
    'S.': {'Haur.':   ["Ezra'a", 'Sanamein'], 
            'J.D.': ['Mourdouk']}}
    # turning this into a csv thing is left

    # %% new dictionary that flattens dict_test. used below to make all_sections_df
    # {  'country':      [country for country in dict_test.keys() 
    #                  for general_loc in dict_test[country] 
    #                  for specific_loc in dict_test[country][general_loc]],
    # 'general_loc':  [general_loc for country in dict_test.keys() 
    #                  for general_loc in dict_test[country] 
    #                  for specific_loc in dict_test[country][general_loc]],
    # 'specific_loc': [specific_loc for country in dict_test.keys() 
    #                  for general_loc in dict_test[country] 
    #                  for specific_loc in dict_test[country][general_loc]]}
   
    # %% view 
    sub_loc_dict

    # %%
    k = 'L.'
    len(dict_test[k].values())   # calculate number of values within a specific key-value pair in dict_test
    
    # %% print country name (remove)
    # for country in dict_test.keys():
    #     for general_loc in dict_test[country]:
    #         for specific_loc in dict_test[country][general_loc]:
    #             print(country)
 
    # %% print general_loc name (remove)
    # for country in dict_test.keys():
    #     for general_loc in dict_test[country]:
    #         for specific_loc in dict_test[country][general_loc]:
    #             print(general_loc)

    # %%
    # final df of parsed localities 
    all_sections_df = [] # initiate list 
    for section_id in tqdm(sub_loc_dict, desc= "converting dictionaries to df..."):
        dict_test = sub_loc_dict[section_id]
        dict_else = other_data[section_id]
        # calculate total number of specific locations in dict_test
        num_loc_data = len([country 
                            for country in dict_test.keys() 
                            for general_loc in dict_test[country] 
                            for specific_loc in dict_test[country][general_loc]])
        # if no location data not counted? 
        name = volume_word_df[(volume_word_df['section_id'] == section_id)]['binom_name'].iloc[0]
        name_data = [name] * num_loc_data
        page_num_data = [section_id[0]] * num_loc_data
        # list countries 
        country_data = [country 
                        for country in dict_test.keys() 
                        for general_loc in dict_test[country] 
                        for specific_loc in dict_test[country][general_loc]]
        aire_geor = [np.nan] * num_loc_data
        floraison = [np.nan] * num_loc_data
        desc = [np.nan] * num_loc_data
        other = [np.nan] * num_loc_data
        basic_author = [np.nan] * num_loc_data
        author = np.nan
        # iterated through dict_else 
        for k in dict_else:
            if k == "aire géogr.":
                aire_geor = [dict_else["aire géogr."]] * num_loc_data
            elif k == "Floraison":
                floraison = [dict_else["Floraison"]] * num_loc_data
            elif k == "desc_paragraph":
                desc = [dict_else["desc_paragraph"]] * num_loc_data
                if isinstance(name, str):
                    author = " ".join(dict_else["desc_paragraph"].split("—")[0].split(" ")[len(name.split(" ")):])

                    text = author
                    if len(re.findall(r"\(.*?\)", text)) > 0:
                        #isn't perfect because of infra species info on fisrt line + its authors
                        first_paran = re.findall(r"\(.*?\)", text)[0]
                        first, rest = text.split(first_paran, 1)
                        rest = re.sub(r"\(.*?\)*", "", rest)
                        if rest == '':
                            author = first + first_paran
                        if first == '':
                            author = first_paran + rest
                        else:
                            author = first
                
                basic_author = [author]* num_loc_data
            elif k == "other":
                other = [dict_else["other"]] * num_loc_data
        # if name is a string, create dict
        if isinstance(name, str):   # make new dict 
            section_data = {'page_num': page_num_data,
                            'binomial': name_data,
                            'basic_author': basic_author,
                            'desc_paragraph': desc,
                            'floraison': floraison,
                            'aire_geor': aire_geor,
                            'other': other,
                            # flatten dict_test
                            'country':      [country for country in dict_test.keys() 
                                             for general_loc in dict_test[country] 
                                             for specific_loc in dict_test[country][general_loc]],
                            'general_loc':  [general_loc for country in dict_test.keys() 
                                             for general_loc in dict_test[country] 
                                             for specific_loc in dict_test[country][general_loc]],
                            'specific_loc': [specific_loc for country in dict_test.keys() 
                                             for general_loc in dict_test[country] 
                                             for specific_loc in dict_test[country][general_loc]]
                        }
            # convert dict to df
            df = pd.DataFrame.from_dict(section_data)
            all_sections_df.append(df)
    print("dictionaries convered to df")

    # %%
    result_df = pd.concat(all_sections_df)

    # %% view
    # result_df['basic_author'].unique()

    # %%
    # test = result_df[result_df['binomial'] == 'Isoetes olympica']
    # name =  'Isoetes olympica'

    # %% view 
    # result_df[result_df['page_num'] == 79]

    # %% 
    print("saving as CSV")
    result_df.to_csv(f'output{today}/main_text/{volume}_location_parsed_v4.csv')


# %%
print("Processing vol1")
process_volume("vol1", vol1_index_df, vol1_word_df, vol1_char_df, vol1_doc)
print("VOL 1 COMPLETE")
print("Processing vol2")
process_volume("vol2", vol2_index_df, vol2_word_df, vol2_char_df, vol2_doc)
print("VOL 2 COMPLETE")
print("Processing vol3")
process_volume("vol3", vol3_index_df, vol3_word_df, vol3_char_df, vol3_doc)
print("VOL 3 COMPLETE")
print("THE END.")