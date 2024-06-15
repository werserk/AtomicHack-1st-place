import pandas as pd
import os
import re
from prepare_data.labels import get_box_cords
from prepare_data.utils import calc_count

def calculate_stats(df: pd.DataFrame):
    data = pd.DataFrame(columns=['path_to_label', 'path_to_image', 'name', 'video_id', 'frame_id', 'pat0', 'pat1', 'pat2', 'pat3', 'pat4'])

    for _, row in df.iterrows():
        path_to_label = row['path_to_label']
        path_to_image = row['path_to_image']
        defects = get_box_cords(path_to_label)

        name = row['file_name']

        pattern_before_parentheses = r'(\d+)\s*\('
        video_id = int(re.search(pattern_before_parentheses, name).group(1))

        pattern_inside_parentheses = r'\((\d+)\)'
        frame_id = int(re.search(pattern_inside_parentheses, name).group(1))

        data = pd.concat((data, pd.DataFrame({'path_to_label': path_to_label,
                                              'path_to_image': path_to_image,
                                              'name': name,
                                              'video_id': video_id,
                                              'frame_id': frame_id,
                                              'pat0': calc_count(defects, '0'),
                                              'pat1': calc_count(defects, '1'),
                                              'pat2': calc_count(defects, '2'),
                                              'pat3': calc_count(defects, '3'),
                                              'pat4': calc_count(defects, '4')}, index=[0])), ignore_index=True)

    return data