import sys
from flask import Blueprint, jsonify, request, render_template, send_file

from modules.users.bp.user import bp as user_bp
from modules.users.bp.image import bp as image_bp

bp = Blueprint('frontend', __name__)

#, url_prefix='/api'


@bp.route('/')
def index():
    return send_file('./static/index.html')

@bp.route('/predict_text', methods=['POST'])
def predict_text():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the input text from the data
    input_text = data.get('text', '')

    # Modify the text (you can customize this part)
    modified_text = input_text.upper()  # Example: Convert to uppercase

    # Create a response dictionary
    response_data = {'modified_text': modified_text}

    # Return the modified text as JSON
    return jsonify(response_data)
