import json
import argparse
import os


# Create the parser
parser = argparse.ArgumentParser(description="Process and sort JSON data.")

# Add the arguments
parser.add_argument('--in', dest='InputFilePath', metavar='input_file_path', type=str, help='the path of the input file to process', default='output_beir_400_train.json')
parser.add_argument('--out', dest='OutputFilePath', metavar='output_file_path', type=str, help='the path of the output file to write to', default='questions.json')

# Parse the arguments
args = parser.parse_args()

if os.path.exists(args.OutputFilePath):
    os.remove(args.OutputFilePath)

# Load the data
with open(args.InputFilePath, 'r') as f:
    data = json.load(f)

# Sort the data
# data.sort(key=lambda x: x['query_id'])

# Create new data structure
new_data = [{'question_id': i, 'question': item['question']} for i, item in enumerate(data)]

# Write to a new file
with open(args.OutputFilePath, 'w') as f:
    json.dump(new_data, f, indent=4)
