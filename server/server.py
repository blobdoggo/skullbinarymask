from flask import Flask, request, jsonify
import os
from main import process_data

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process_folders():
    
    data = request.args
    #data = request.get_json()
    input_folder = data.get('input')
    output_folder = data.get('output')

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    if not input_folder:
        return jsonify({'status': 'error', 'message': 'Missing input'}), 400

    if not os.path.isdir(input_folder):
        return jsonify({'status': 'error', 'message': f'Input folder {input_folder} does not exist'}), 400

    '''if not os.path.isdir(output_folder):
        return jsonify({'status': 'error', 'message': f'Output folder {output_folder} does not exist'}), 400'''

    # Placeholder for processing logic
    # ...
    process_data(input_folder, output_folder)

    return jsonify({'status': 'success', 'message': f'Folders processed successfully - {input_folder} -> {output_folder}'}), 200

if __name__ == '__main__':
    app.run(debug=True)