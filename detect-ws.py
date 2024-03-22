from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import face_recognition
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allow CORS for all routes
#CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# Load known encodings and names from file
known_encodings = np.load("known_encodings.npy")
with open("known_names.txt", "r") as f:
    known_names = f.read().splitlines()

@app.route('/recognize', methods=['POST'])
@cross_origin()  # Allow CORS for this route
def recognize_faces():
    # Get the JSON data from the request
    data = request.json

    if 'image' not in data:
        return jsonify({"error": "No image data provided"})

    # Decode the Base64 image
    base64_image = data['image']
    decoded_image = base64.b64decode(base64_image)

    # Convert the image to a numpy array
    nparr = np.frombuffer(decoded_image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # List to store recognized names
    recognized_names = []

    # Loop through each face found in the image
    for face_encoding in face_encodings:
        # Compare this face to the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the matched known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Add the recognized name to the list
        recognized_names.append(name)

    # Return the recognized names as JSON response    
    return jsonify({"recognized_names": recognized_names})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
