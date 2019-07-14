import os
import subprocess
import numpy as np
import random
import pickle
import cv2
import math
import xml.etree.cElementTree as ET


def assert_path(path, error_message):
    assert os.path.exists(path), error_message


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def init_directories():
    # Setup the directory structure.
    if not os.path.exists(os.path.join(destination_path, 'JPEGImages')):
        os.makedirs(os.path.join(destination_path, 'JPEGImages'))

    if not os.path.exists(os.path.join(destination_path, 'ImageSets', 'Main')):
        os.makedirs(os.path.join(destination_path, 'ImageSets', 'Main'))

    if not os.path.exists(os.path.join(destination_path, 'Annotations')):
        os.makedirs(os.path.join(destination_path, 'Annotations'))

    if not os.path.exists(os.path.join(destination_path, 'pickle_store')):
        os.makedirs(os.path.join(destination_path, 'pickle_store'))

    # Flush the train-val-test split. A new split will be created each time this script is run.
    for f in os.listdir(os.path.join(destination_path, 'ImageSets', 'Main')):
        os.remove(os.path.join(destination_path, 'ImageSets', 'Main', f))

    # Creating empty files.
    touch(os.path.join(destination_path, 'ImageSets', 'Main', 'train.txt'))
    touch(os.path.join(destination_path, 'ImageSets', 'Main', 'val.txt'))
    touch(os.path.join(destination_path, 'ImageSets', 'Main', 'test.txt'))
    touch(os.path.join(destination_path, 'ImageSets', 'Main', 'trainval.txt'))


def write_to_file(filename, content):
    f = open(filename, 'a')
    f.write(content+'\n')


def parse_annotation():
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()

    annotation_dict = {}
    for annotation_row in annotations:
        annotation_parsed = annotation_row.split(' ')
        annotation_parsed[-1] = annotation_parsed[-1][:-1]
        if annotation_parsed[0] not in annotation_dict:
            annotation_dict[annotation_parsed[0]] = [annotation_parsed[1:]]
        else:
            annotation_dict[annotation_parsed[0]].append(annotation_parsed[1:])

    return annotation_dict


def annotate_frames(dest_path):
    # Create VOC style annotation.

    annotations_dict = parse_annotation()

    for annotation_name in annotations_dict:
        image_path = os.path.join(destination_path, 'JPEGImages', annotation_name + '.jpg')
        image = cv2.imread(image_path)
        height, width, depth = image.shape

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = destination_folder_name
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = 'Stanford Drone Dataset'
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        ET.SubElement(annotation, "segmented").text = '0'
        ET.SubElement(annotation, "filename").text = annotation_name

        for annotation_data in annotations_dict[annotation_name]:
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = annotation_data[1]
            ET.SubElement(object, "pose").text = 'Unspecified'
            ET.SubElement(object, "truncated").text = '0'
            ET.SubElement(object, "difficult").text = '0'
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "track_id").text = annotation_data[0]
            ET.SubElement(bndbox, "xmin").text = annotation_data[2]
            ET.SubElement(bndbox, "ymin").text = annotation_data[3]
            ET.SubElement(bndbox, "xmax").text = annotation_data[4]
            ET.SubElement(bndbox, "ymax").text = annotation_data[5]

        xml_annotation = ET.ElementTree(annotation)
        print(os.path.join(dest_path, annotation_name + ".xml"))
        xml_annotation.write(os.path.join(dest_path, annotation_name + ".xml"))


def split_dataset_uniformly():
    # jpeg_path = os.path.join(destination_path, 'JPEGImages')
    for index in range(1000):
        file_name_prefix = "scene1_"
        test_path = destination_path + "/Annotations/" + file_name_prefix + str(index) + ".xml"
        if os.path.exists(test_path):
            if index % 3 != 0:
                # Training

                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'train.txt'),
                              file_name_prefix + str(index))
                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'trainval.txt'),
                              file_name_prefix + str(index))
            else:
                # Validation
                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'val.txt'), file_name_prefix + str(index))
                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'trainval.txt'),
                              file_name_prefix + str(index))

    for index in range(1000):
        file_name_prefix = "scene2_"
        test_path = destination_path + "/Annotations/" + file_name_prefix + str(index) + ".xml"
        if os.path.exists(test_path):
            if index % 3 != 0:
                # Training

                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'train.txt'),
                              file_name_prefix + str(index))
                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'trainval.txt'),
                              file_name_prefix + str(index))
            else:
                # Validation
                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'val.txt'), file_name_prefix + str(index))
                write_to_file(os.path.join(destination_path, 'ImageSets', 'Main', 'trainval.txt'),
                              file_name_prefix + str(index))


def split_and_annotate(dirs):
    init_directories()

    jpeg_path = os.path.join(destination_path, 'JPEGImages')
    for dir in dirs:
        full_dir = os.path.join(dataset_path, dir)
        for image in os.listdir(full_dir):
            os.system("cp {}/{} {}/{}_{}".format(full_dir, image, jpeg_path, dir, image))

    annotate_frames(os.path.join(destination_path, 'Annotations'))
    split_dataset_uniformly()


if __name__ == '__main__':

    dataset_path = '/Users/ekaterinapogodina/'
    destination_folder_name = 'SIMPLEdevkit'
    destination_path = os.path.join(dataset_path, destination_folder_name)
    annotation_file = os.path.join(dataset_path, 'annotations.txt')

    split_and_annotate(['scene1',  'scene2'])
