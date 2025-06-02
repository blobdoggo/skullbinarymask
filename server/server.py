from flask import Flask, request, jsonify
import os
from main import process_data
from pathlib import Path

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process_folders():
    
    data = request.args
    #data = request.get_json()
    input_folder = data.get('input')
    output_folder = data.get('output')
    base_folder = Path.cwd()

    print (f"Current working directory: {base_folder}")

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    if not input_folder:
        input_folder = os.path.join(base_folder, 'dataset', 'input')
        #return jsonify({'status': 'error', 'message': 'Input Folder not specified. Using default input.'}), 200

    if not output_folder:
        output_folder = os.path.join(base_folder, 'dataset', 'output')
        #return jsonify({'status': 'error', 'message': 'Output Folder not specified. Using default output.'}), 200

    '''if not os.path.isdir(input_folder):
        return jsonify({'status': 'error', 'message': f'Input folder {input_folder} does not exist'}), 400'''

    '''if not os.path.isdir(output_folder):
        return jsonify({'status': 'error', 'message': f'Output folder {output_folder} does not exist'}), 400'''

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Placeholder for processing logic
    # ...
    process_data(input_folder, output_folder)

    return jsonify({'status': 'success', 'message': f'Folders processed successfully - {input_folder} -> {output_folder}'}), 200

if __name__ == '__main__':
    app.run(debug=True)