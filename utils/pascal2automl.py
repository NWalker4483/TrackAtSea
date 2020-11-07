import random 

import xml.etree.ElementTree as ET
import pickle

from os import listdir
from os.path import isfile, join
source_folder = "generated_data/target/train/annotations"
bucket_name = "gs://msu_track_bucket"

xml_files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
csv = open("0.csv",'w+')
try: 
    for xml_file in xml_files:
        tree = ET.parse(join(source_folder, xml_file))
        root = tree.getroot()
        pasc = XmlDictConfig(root)

        Height, Width = int(pasc["size"]["height"]), int(pasc["size"]["width"])
        Fname = pasc["filename"]
        Class = pasc["object"]["name"]
        x1 = round(int(pasc["object"]["bndbox"]["xmin"]) / Width, 3)
        y1 = round(int(pasc["object"]["bndbox"]["ymin"]) / Height, 3)
        x2 = round(int(pasc["object"]["bndbox"]["xmax"]) / Width, 3)
        y2 = round(int(pasc["object"]["bndbox"]["ymax"]) / Height, 3)
        Set = random.choices(["TRAIN", "TEST", "VALIDATE"], weights = [6, 3, 1])[0]

        line = f"{Set},{join(bucket_name, Fname)},{Class},{x1},{y1},,,{x2},{y2},,\n"
        print(line)
        csv.write(line)
finally:
    csv.close()
    pass