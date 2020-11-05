import cv2 
import os 
in_dir = "generated_data/images"
out_dir = 
path = os.path.join(out_path, f"{vid_path.split('/')[-1].split('.')[0]}.{count}.png")

# <annotation verified="yes">
# 	<folder>images</folder>
# 	<filename>raccoon-1.jpg</filename>
# 	<path>/Users/datitran/Desktop/raccoon/images/raccoon-1.jpg</path>
# 	<source>
# 		<database>Unknown</database>
# 	</source>
# 	<size>
# 		<width>650</width>
# 		<height>417</height>
# 		<depth>3</depth>
# 	</size>
# 	<segmented>0</segmented>
# 	<object>
# 		<name>raccoon</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>81</xmin>
# 			<ymin>88</ymin>
# 			<xmax>522</xmax>
# 			<ymax>408</ymax>
# 		</bndbox>
# 	</object>
# </annotation>