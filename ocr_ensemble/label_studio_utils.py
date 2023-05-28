import json
import os 
import numpy as np
import math
import glob

def convert_json2prediction(filepath, fdict):
    with open(filepath, 'r') as f:
        data = json.load(f)

    label_studio_format = {
        'data': {
            'ocr': fdict[os.path.basename(filepath).replace('.json', '.jpg')],
        },
        'annotations': [],
        'predictions': [{'model_version': 'PaddleOCR(use_angle_cls=True, lang="en")',  
            'score': np.mean(data['ocr_result']['confidences']), "result":[]}],
    }

    for i, (bbox, text) in enumerate(zip(data['ocr_result']['bounding_boxes'], data['ocr_result']['text'])):
        
        upper_left, upper_right, lower_right, lower_left = bbox

        # Calculate the rotation angle of the box
        dx = upper_right[0] - upper_left[0]
        dy = upper_right[1] - upper_left[1]
        rotation_angle = math.degrees(math.atan2(dy, dx))
        
        x, y, w, h = float(bbox[0][0]), float(bbox[0][1]), float(abs(bbox[1][0] - bbox[0][0])), float(abs(bbox[3][1] - bbox[0][1]))
        x/= data['width']
        y/= data['height']
        w/= data['width']
        h/= data['height']
        x *= 100
        y *= 100
        w *= 100
        h *= 100
        bbox_annotation = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'rotation': rotation_angle,
        }
        
        result_id = f'bb{i+1}'  # Create an ID for each bounding box and transcription
        label_studio_format['predictions'][0]['result'].extend([
                {
                    'original_width': data["width"],
                    'original_height': data["height"],
                    'image_rotation': 0,
                    'value': bbox_annotation,
                    'id': result_id,
                    'from_name': 'bbox',
                    'to_name': 'image',
                    'type': 'rectangle',
                },
                {
                    'original_width': data["width"],
                    'original_height': data["height"],
                    'image_rotation': 0,
                    'value': {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'rotation': rotation_angle,
                        'text': [text],
                    },
                    'id': result_id,
                    'from_name': 'transcription',
                    'to_name': 'image',
                    'type': 'textarea',
                }
        ])

    return label_studio_format

def convert_json2labelstudio(input_folder, fdict):
    files = glob.glob(os.path.join(input_folder, '*.json'))
    converted_data = []

    for filepath in files:
        converted = convert_json2prediction(filepath, fdict)
        converted_data.append(converted)
    return converted_data

def rotate_point(center, point, angle):
    """Rotate a point counterclockwise by a given angle around a given center.

    Args:
        center (tuple): The center of rotation, represented as (x, y).
        point (tuple): The point to rotate, represented as (x, y).
        angle (float): The angle of rotation in degrees.

    Returns:
        (tuple): The rotated point, represented as (x, y).
    """
    angle = math.radians(angle)
    x, y = point[0] - center[0], point[1] - center[1]
    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)
    return [new_x + center[0], new_y + center[1]]

def convert_labelstudio2json(label_studio_dataset):
    """
    Converts a dataset in Label Studio format back into individual JSON dictionaries.

    Args:
        label_studio_dataset: A list of Label Studio data entries.

    Returns:
        A list of dictionaries, each representing a single JSON file's data.
    """
    json_dicts = []

    for i, entry in enumerate(label_studio_dataset):
        # Prepare the data for the individual file
        individual_file_data = {
            'ocr_annotation': {
                'bounding_boxes': [],
                'text': [],
                'rotation': [],
            },
            'width': None,
            'height': None,
            'label_studio_fname': entry['data']['ocr'],
            'label_studio_id': entry['id'],
        }

        # Iterate through the results
        for prediction in entry['annotations']:
            for result in prediction['result']:
                # Save original image size
                individual_file_data['width'] = result['original_width']
                individual_file_data['height'] = result['original_height']
                
                if result['type'] == 'rectangle':
                    # Convert the bbox coordinates back to the original format
                    x = result['value']['x'] / 100 * individual_file_data['width']
                    y = result['value']['y'] / 100 * individual_file_data['height']
                    w = result['value']['width'] / 100 * individual_file_data['width']
                    h = result['value']['height'] / 100 * individual_file_data['height']
                    rotation = result['value']['rotation']

                    center = (x, y)#(x + w / 2, y + h / 2)
                    corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    rotated_corners = [rotate_point(center, corner, rotation) for corner in corners]
                    individual_file_data['ocr_annotation']['rotation'].append(rotation)
                    individual_file_data['ocr_annotation']['bounding_boxes'].append(rotated_corners)
                elif result['type'] == 'textarea':
                    # Save the recognized text
                    text = result['value']['text'][0]
                    individual_file_data['ocr_annotation']['text'].append(text)
        json_dicts.append(individual_file_data)
    return json_dicts


