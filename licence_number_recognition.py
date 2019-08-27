# Usage python licence_number_recognition.py test_case/test_case_input test_case/test_case_output


import numpy as np
import os
import sys
import tensorflow as tf
from pytesseract import pytesseract
from PIL import Image
from utils import allow_needed_values as anv
from utils import do_image_conversion as dic
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

tessdata_dir_config = r'--tessdata-dir "/home/genesys/MyLibs/tesseract/tessdata"'
PATH_TO_CKPT = 'trained_model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training_data/class_labels.pbtxt'
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        test_img_path = sys.argv[1]
        for image_name in os.listdir(test_img_path):
            image = Image.open(sys.argv[1] + '/' + image_name)
            print('Image Name: ' + image_name)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            ymin = boxes[0, 0, 0]
            xmin = boxes[0, 0, 1]
            ymax = boxes[0, 0, 2]
            xmax = boxes[0, 0, 3]
            (im_width, im_height) = image.size
            (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn), int(ymaxx - yminn),
                                                          int(xmaxx - xminn))
            img_data = sess.run(cropped_image)
            filename = dic.yo_make_the_conversion(img_data, image_name)
            text = pytesseract.image_to_string(Image.open(filename), lang='eng', config=tessdata_dir_config)
            print('Licence Number : ', anv.catch_rectify_plate_characters(text))
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)
            out_img = Image.fromarray(image_np)
            out_img.save(sys.argv[2] + '/' + image_name)
