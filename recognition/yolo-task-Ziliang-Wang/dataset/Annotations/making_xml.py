import cv2
import os
from xml.dom import minidom

pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
folder = father_path + "/Mask_labeling/"
print(folder)
files = os.listdir(folder)


def file_filter(file):
	"""
	  Read all files that is specified, the format of mask of ISIC dataset is png.
	"""
	if file[-4:] in ['.png']:
		return True
	else:
		return False


files_name = list(filter(file_filter, files))


def create_xml_test(file_name, pic_name, width_, height_, x_min, y_min, x_max, y_max):
	xml = minidom.Document()
	annotation = xml.createElement('annotation')
	xml.appendChild(annotation)
	text_node = xml.createElement('folder')
	text_node.appendChild(xml.createTextNode('ISIC'))
	annotation.appendChild(text_node)

	text_node = xml.createElement('filename')
	text_node.appendChild(xml.createTextNode(pic_name))
	annotation.appendChild(text_node)

	text_node = xml.createElement('path')
	text_node.appendChild(xml.createTextNode('dataset\mask\ISIC2018_Task1_Training_GroundTruth_x2\l' + pic_name))
	annotation.appendChild(text_node)

	source = xml.createElement('source')
	annotation.appendChild(source)

	text_node = xml.createElement('database')
	text_node.appendChild(xml.createTextNode('ISIC Database'))
	source.appendChild(text_node)

	size = xml.createElement('size')
	width = xml.createElement('width')
	width.appendChild(xml.createTextNode(str(width_)))
	height = xml.createElement('height')
	height.appendChild(xml.createTextNode(str(height_)))
	depth = xml.createElement('depth')
	depth.appendChild(xml.createTextNode('3'))

	size.appendChild(width)
	size.appendChild(height)
	size.appendChild(depth)
	annotation.appendChild(size)

	text_node = xml.createElement('segmented')
	text_node.appendChild(xml.createTextNode('0'))
	annotation.appendChild(text_node)

	object = xml.createElement('object')
	name = xml.createElement('name')
	name.appendChild(xml.createTextNode('Lesion'))
	pose = xml.createElement('pose')
	pose.appendChild(xml.createTextNode('Unspecified'))
	truncated = xml.createElement('truncated')
	truncated.appendChild(xml.createTextNode('0'))
	difficult = xml.createElement('difficult')
	difficult.appendChild(xml.createTextNode('0'))
	bndbox = xml.createElement('bndbox')
	xmin = xml.createElement('xmin')
	xmin.appendChild(xml.createTextNode(str(y_min)))
	ymin = xml.createElement('ymin')
	ymin.appendChild(xml.createTextNode(str(x_min)))
	xmax = xml.createElement('xmax')
	xmax.appendChild(xml.createTextNode(str(y_max)))
	ymax = xml.createElement('ymax')
	ymax.appendChild(xml.createTextNode(str(x_max)))

	bndbox.appendChild(xmin)
	bndbox.appendChild(ymin)
	bndbox.appendChild(xmax)
	bndbox.appendChild(ymax)

	object.appendChild(name)
	object.appendChild(pose)
	object.appendChild(truncated)
	object.appendChild(difficult)
	object.appendChild(bndbox)
	annotation.appendChild(object)

	f = open(file_name, 'w')
	f.write(xml.toprettyxml())
	f.close()


for index, file_name in enumerate(files_name):
	print(folder + "/" + file_name)
	imagess = cv2.imread(folder + "/" + file_name, -1)

	white_area_x = []
	white_area_y = []
	width = imagess.shape[1]
	heigth = imagess.shape[0]

	for i in range(len(imagess)):
		for j in range(len(imagess[i])):
			if imagess[i][j] == 255:
				white_area_x.append(i)
				white_area_y.append(j)

	x_min, y_min = min(white_area_x), min(white_area_y)
	x_max, y_max = max(white_area_x), max(white_area_y)
	pic_name = file_name[0:12]
	create_xml_test(pic_name + ".xml", pic_name, width, heigth, x_min, y_min, x_max, y_max)
