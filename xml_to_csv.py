#usage python xml_to_csv.py dataset/training_data training_data/train.csv

import glob
import pandas as pd
import xml.etree.ElementTree as ET
import os
import sys


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            jpg_name = os.path.basename(xml_file)
            value = (jpg_name[:-4] + '.jpg',
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':
    image_path = sys.argv[1]
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(sys.argv[2], index=None)
    print('Successfully converted xml to csv.')
