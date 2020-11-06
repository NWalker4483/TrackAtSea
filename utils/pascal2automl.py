[set,]image_path[,label,x1,y1,,,x2,y2,,]
TRAIN,gs://My_Bucket/sample1.jpg,cat,0.125,0.25,,,0.375,0.5,,
VALIDATE,gs://My_Bucket/sample1.jpg,cat,0.4,0.3,,,0.55,0.55,,
TEST,gs://My_Bucket/sample1.jpg,dog,0.5,0.675,,,0.75,0.875,,

<annotation>
	<folder>images</folder>
	<filename>6.100.png</filename>
	<path>/Users/nilewalker/Projects/GitHub/MsuTrackingAI/generated_data/images/6.100.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1280</width>
		<height>720</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>target</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>966</xmin>
			<ymin>424</ymin>
			<xmax>1018</xmax>
			<ymax>451</ymax>
		</bndbox>
	</object>
</annotation>

import random 

import xml.etree.ElementTree as ET
import pickle

from os import listdir
from os.path import isfile, join
source_folder = "generated_data/target/train/annotations"
bucket_name = "gs://msu_track_bucket"

xml_files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
for xml_file in xml_files:
    print(join(source_folder, xml_file))
    # tree = ET.parse(
    #     'raw_data/main_boat_position/onboard_gps_source1/AI Tracks at Sea High Frequency GPS_train.txt')

    # root = tree.getroot()
    # for entry in root.iter("trkpt"):
        
