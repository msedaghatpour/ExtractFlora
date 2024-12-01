


cd /Users/maryamsedaghatpour/Desktop/extract-flora/shidedh\:ExtractFloraOrganizedV0-p0_draft2024Nov18/scripts
cd ./1_Preprocessing
echo "Preprocessing is running..."
python3.10 get_char_df.py
cd ..

cd ./2_Index
echo "Index is running..."
python3.10 Index.py
cd ..

cd ./3_MainText
echo "Main text is running..."
python3.10 1_find_entry_boxes.py
python3.10 2_entry_bbox_page_cont.py
python3.10 3_parsing_locations.py
cd ..
