#!/usr/bin/env python3
"""
Data Quality Viewer for WindBlade and SolarPanel datasets.
Web-based interface using Flask.
"""

from flask import Flask, render_template, jsonify, send_from_directory
import json
import os
from pathlib import Path
import glob
import base64
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
import io
from collections import OrderedDict
import subprocess

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuration
DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "labels"
IMAGES_DIR = DATA_DIR / "images"

# Color scheme for annotations
COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", 
    "#FF00FF", "#00FFFF", "#FFA500", "#800080", "#008000"
]


def get_all_json_files():
    """Get all JSON files in the labels directory."""
    return sorted(glob.glob(str(LABELS_DIR / "**/*.json"), recursive=True))


def find_image_path(json_path, image_filename):
    """Find the corresponding image path based on JSON path."""
    json_rel_path = Path(json_path).relative_to(LABELS_DIR)
    image_path = IMAGES_DIR / json_rel_path.parent / image_filename
    return str(image_path)


def extract_exif(image_path, dataset_type=None):
    """
    Extract EXIF data from image using exiftool.

    Args:
        image_path: Path to the image file
        dataset_type: 'WindBlade' or 'SolarPanel' (determines which fields to extract)

    Returns:
        OrderedDict with selected EXIF fields based on dataset type
    """
    exif_data = OrderedDict()

    try:
        # Run exiftool to get EXIF data in JSON format
        result = subprocess.run(
            ['exiftool', '-j', str(image_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            exif_data['error'] = f"exiftool error: {result.stderr}"
            return exif_data

        # Parse JSON output
        exif_list = json.loads(result.stdout)
        if not exif_list:
            exif_data['message'] = 'exiftool returned empty result'
            return exif_data

        exif = exif_list[0]

        # Common fields for all datasets
        # 1. DateTimeOriginal
        if 'DateTimeOriginal' in exif:
            # Convert YYYY:MM:DD HH:MM:SS to YYYY-MM-DD HH:MM:SS
            datetime_str = exif['DateTimeOriginal']
            exif_data['DateTimeOriginal'] = datetime_str.replace(':', '-', 2)

        # 2. Camera Model (CameraModelName or Model)
        camera_model = exif.get('CameraModelName') or exif.get('Model')
        if camera_model:
            exif_data['CameraModel'] = camera_model

        # Dataset-specific fields
        if dataset_type == 'WindBlade':
            # 3. CropHiSpeed (풍력 전용)
            if 'CropHiSpeed' in exif:
                crop_hi_speed = exif['CropHiSpeed']
                # Extract first value (e.g., "FX 5568x3712 5568x3712" -> "FX")
                exif_data['CropHiSpeed'] = crop_hi_speed.split(' ')[0] if crop_hi_speed else crop_hi_speed
                exif_data['CropHiSpeed_Full'] = crop_hi_speed  # Also keep full value

        elif dataset_type == 'SolarPanel':
            # 4. ImageSize (태양광 전용)
            if 'ImageSize' in exif:
                exif_data['ImageSize'] = exif['ImageSize']

        # Additional useful fields (always include)
        if 'ImageWidth' in exif:
            exif_data['ImageWidth'] = exif['ImageWidth']
        if 'ImageHeight' in exif:
            exif_data['ImageHeight'] = exif['ImageHeight']
        if 'FileSize' in exif:
            exif_data['FileSize'] = exif['FileSize']
        if 'FileType' in exif:
            exif_data['FileType'] = exif['FileType']

    except subprocess.TimeoutExpired:
        exif_data['error'] = 'exiftool timeout'
    except FileNotFoundError:
        exif_data['error'] = 'exiftool not found - please install exiftool'
    except json.JSONDecodeError as e:
        exif_data['error'] = f'Failed to parse exiftool JSON output: {str(e)}'
    except Exception as e:
        exif_data['error'] = f'Error extracting EXIF: {str(e)}'
        import traceback
        exif_data['traceback'] = traceback.format_exc()

    return exif_data if exif_data else {'message': 'No EXIF data found'}


def draw_annotations(img, data, dataset_type=None):
    """Draw annotations on the image."""
    draw = ImageDraw.Draw(img)

    db_name = data.get('info', {}).get('db_name', '')

    # Draw segmentation polygons (for PositiveDB only)
    if db_name == "PositiveDB" and 'annotations' in data:
        for idx, ann in enumerate(data['annotations']):
            if 'segmentation' in ann:
                seg = ann['segmentation']
                points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]

                if len(points) >= 2:
                    # For SolarPanel: use blue color and smaller markers
                    if dataset_type == 'SolarPanel':
                        color = '#0000FF'
                        line_width = 1
                        marker_radius = 0.5
                    else:
                        color = COLORS[idx % len(COLORS)]
                        line_width = 3
                        marker_radius = 2

                    draw.line(points + [points[0]], fill=color, width=line_width)
                    for point in points:
                        draw.ellipse(
                            [point[0]-marker_radius, point[1]-marker_radius,
                             point[0]+marker_radius, point[1]+marker_radius],
                            fill=color
                        )
    
    # Draw VisionQA bounding boxes
    if 'visionqa' in data:
        visionqa = data['visionqa']
        if 'defect_localization_option' in visionqa:
            correct_answer = visionqa.get('defect_localization_a', '')
            cropped_bbox = visionqa.get('cropped_bbox', None)
            
            # Draw cropped_bbox if it exists (WindBlade)
            if cropped_bbox:
                x, y, w, h = cropped_bbox
                draw.rectangle([x, y, x+w, y+h], outline='#00FFFF', width=3)
                draw.text((x+5, y+5), "Cropped Region", fill='#00FFFF')
            
            # Draw localization options
            options = visionqa['defect_localization_option']
            for label in ['a', 'b', 'c', 'd']:
                option_key = f'localization_option_{label}'
                if option_key in options:
                    bbox_str = options[option_key]
                    bbox = eval(bbox_str)
                    x, y, w, h = bbox
                    
                    # Adjust coordinates if cropped_bbox exists
                    if cropped_bbox:
                        x += cropped_bbox[0]
                        y += cropped_bbox[1]
                    
                    # Choose color based on correct answer and dataset type
                    if label == correct_answer:
                        color = '#00FF00'
                        width = 4
                    else:
                        # For SolarPanel, use white for incorrect options
                        if dataset_type == 'SolarPanel':
                            color = '#FFFFFF'
                        else:
                            color = '#FF0000'
                        width = 2
                    
                    draw.rectangle([x, y, x+w, y+h], outline=color, width=width)
                    label_text = f"{label.upper()}" + (" ✓" if label == correct_answer else "")
                    draw.text((x+5, y+5), label_text, fill=color)
    
    return img


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/files')
def get_files():
    """Get list of all JSON files."""
    files = get_all_json_files()
    return jsonify({
        'files': [str(Path(f).relative_to(LABELS_DIR)) for f in files],
        'total': len(files)
    })


@app.route('/api/file/<int:index>')
def get_file(index):
    """Get file data by index."""
    files = get_all_json_files()
    
    if index < 0 or index >= len(files):
        return jsonify({'error': 'Index out of range'}), 404
    
    json_path = files[index]
    
    try:
        # Load JSON preserving field order
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        
        # Get image
        image_filename = data.get('image', {}).get('filename', '')
        image_path = find_image_path(json_path, image_filename)
        
        image_data = None
        cropped_image_data = None
        dataset_type = None
        exif_data = {'message': 'Image file not found'}
        img_original = None
        image_size = None

        if os.path.exists(image_path):
            # Determine dataset type first (needed for EXIF extraction)
            if 'WindBlade' in json_path:
                dataset_type = 'WindBlade'
            elif 'SolarPanel' in json_path:
                dataset_type = 'SolarPanel'

            # Extract EXIF from image file using exiftool
            try:
                exif_data = extract_exif(image_path, dataset_type)
            except Exception as e:
                exif_data = {'error': f'Failed to extract EXIF: {str(e)}'}

            # Load original image for processing
            img_original_temp = Image.open(image_path)

            # Store image size
            image_size = [img_original_temp.width, img_original_temp.height]

            # Convert to RGB for processing
            img_original = img_original_temp.convert('RGB')

            # Check if this is WindBlade with cropped_bbox
            visionqa = data.get('visionqa', {})
            cropped_bbox = visionqa.get('cropped_bbox', None)
            
            # For WindBlade with cropped_bbox, create two images
            if cropped_bbox and dataset_type == 'WindBlade':
                # Image 1: Original with cropped bbox only
                img1 = img_original.copy()
                draw1 = ImageDraw.Draw(img1)
                
                # Draw segmentation polygons
                db_name = data.get('info', {}).get('db_name', '')
                if db_name == "PositiveDB" and 'annotations' in data:
                    for idx, ann in enumerate(data['annotations']):
                        if 'segmentation' in ann:
                            seg = ann['segmentation']
                            points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                            if len(points) >= 2:
                                color = COLORS[idx % len(COLORS)]
                                draw1.line(points + [points[0]], fill=color, width=12)
                                for point in points:
                                    draw1.ellipse(
                                        [point[0]-8, point[1]-8, point[0]+8, point[1]+8],
                                        fill=color
                                    )

                # Draw cropped bbox
                x, y, w, h = cropped_bbox
                draw1.rectangle([x, y, x+w, y+h], outline='#00FFFF', width=8)
                draw1.text((x+5, y+5), "Cropped Region", fill='#00FFFF', font=None)
                
                # Convert to base64
                buffer1 = io.BytesIO()
                img1.save(buffer1, format='JPEG', quality=90)
                image_data = base64.b64encode(buffer1.getvalue()).decode()
                
                # Image 2: Cropped region with localization options
                x, y, w, h = cropped_bbox
                img2 = img_original.crop((x, y, x+w, y+h))
                draw2 = ImageDraw.Draw(img2)
                
                # Draw localization options (relative to cropped region)
                correct_answer = visionqa.get('defect_localization_a', '')
                options = visionqa.get('defect_localization_option', {})
                
                for label in ['a', 'b', 'c', 'd']:
                    option_key = f'localization_option_{label}'
                    if option_key in options:
                        bbox_str = options[option_key]
                        bbox = eval(bbox_str)
                        bx, by, bw, bh = bbox
                        
                        # These are already relative to cropped region
                        if label == correct_answer:
                            color = '#00FF00'
                            width = 4
                        else:
                            color = '#FF0000'
                            width = 2
                        
                        draw2.rectangle([bx, by, bx+bw, by+bh], outline=color, width=width)
                        label_text = f"{label.upper()}" + (" ✓" if label == correct_answer else "")
                        draw2.text((bx+5, by+5), label_text, fill=color, font=None)
                
                # Convert to base64
                buffer2 = io.BytesIO()
                img2.save(buffer2, format='JPEG', quality=90)
                cropped_image_data = base64.b64encode(buffer2.getvalue()).decode()
                
            else:
                # For SolarPanel or WindBlade without cropped_bbox, use original approach
                img = draw_annotations(img_original, data, dataset_type)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=90)
                image_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Convert data to JSON string to preserve order
        data_json_str = json.dumps(data, indent=2, ensure_ascii=False)
        exif_json_str = json.dumps(exif_data, indent=2, ensure_ascii=False)

        return jsonify({
            'index': index,
            'total': len(files),
            'filename': str(Path(json_path).relative_to(LABELS_DIR)),
            'data': data,
            'data_json_str': data_json_str,
            'exif': exif_data,
            'exif_json_str': exif_json_str,
            'image': image_data,
            'cropped_image': cropped_image_data,
            'dataset_type': dataset_type,
            'image_size': image_size
        })
    
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'file': json_path if 'json_path' in locals() else 'unknown'
        }
        print(f"Error processing file: {error_details}")
        return jsonify(error_details), 500


@app.route('/api/search')
def search_files():
    """Search files by query."""
    from flask import request
    query = request.args.get('q', '').lower()
    
    files = get_all_json_files()
    if query:
        matching_indices = [i for i, f in enumerate(files) if query in f.lower()]
    else:
        matching_indices = list(range(len(files)))
    
    return jsonify({
        'indices': matching_indices,
        'total': len(matching_indices)
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Data Quality Viewer...")
    print(f"Found {len(get_all_json_files())} JSON files")
    print("Open http://localhost:5001 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
