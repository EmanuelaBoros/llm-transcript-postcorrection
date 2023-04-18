import os
from bs4 import BeautifulSoup
import argparse


def extract_text_equiv(gt_path, pred_path):
    # Recursively search for XML files in the ground truth folder
    for root, dirs, files in os.walk(gt_path):
        for gt_file in files:
            # Check if the file is an XML file
            if gt_file.endswith('.xml'):
                # Construct the corresponding prediction file path
                pred_file = os.path.join(pred_path, gt_file)

                # Parse the ground truth file
                with open(os.path.join(root, gt_file), 'r') as f:
                    gt_soup = BeautifulSoup(f, 'xml')

                # Parse the prediction file
                with open(pred_file.replace('.xml', '_p1_1.xml'), 'r') as f:
                    pred_soup = BeautifulSoup(f, 'xml')

                # Iterate over each TextRegion in the ground truth file
                for gt_region in gt_soup.find_all('TextRegion'):
                    # Find the corresponding TextRegion in the prediction file
                    region_id = gt_region['id']
                    pred_region = pred_soup.find('TextRegion', {'id': region_id})

                    # Extract the TextEquiv content for the TextRegion in the ground truth and prediction files
                    gt_region_text = gt_region.find('TextEquiv').text
                    pred_region_text = pred_region.find('TextEquiv').text

                    # Print the extracted TextEquiv content for the TextRegion in the ground truth and prediction files
                    print(f'Ground truth TextRegion {region_id}:\n{gt_region_text}\n')
                    print(f'Prediction TextRegion {region_id}:\n{pred_region_text}\n')

                    # Iterate over each TextLine in the TextRegion
                    for gt_line in gt_region.find_all('TextLine'):
                        # Find the corresponding TextLine in the prediction file
                        line_id = gt_line['id']
                        pred_line = pred_soup.find('TextLine', {'id': line_id})

                        # Extract the TextEquiv content for the TextLine in the ground truth and prediction files
                        gt_line_text = gt_line.find('TextEquiv').text
                        pred_line_text = pred_line.find('TextEquiv').text

                        # Print the extracted TextEquiv content for the TextLine in the ground truth and prediction files
                        print(f'Ground truth TextLine {line_id}:\n{gt_line_text}\n')
                        print(f'Prediction TextLine {line_id}:\n{pred_line_text}\n')


if __name__ == '__main__':
    # Create an argument parser for the ground truth and prediction paths
    parser = argparse.ArgumentParser(description='Extract TextEquiv content for ground truth and prediction files.')
    parser.add_argument('gt_path', type=str, help='Path to ground truth folder')
    parser.add_argument('pred_path', type=str, help='Path to prediction folder')
    args = parser.parse_args()

    # Call the extract_text_equiv function with the provided paths
    extract_text_equiv(args.gt_path, args.pred_path)
