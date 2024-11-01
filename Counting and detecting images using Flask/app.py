from flask import Flask, request, send_file, render_template_string
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import requests

app = Flask(__name__)

# Serve the HTML form directly from within the Flask app
@app.route('/')
def upload_image():
    # The HTML form is now rendered directly from the code
    html_form = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vehicle Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f8ff;
                color: #333;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            h1 {
                color: #ff4500;
            }
            form {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                width: 300px;
                text-align: center;
            }
            input[type="text"],
            input[type="file"] {
                width: 90%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            input[type="submit"] {
                background-color: #ff4500;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            input[type="submit"]:hover {
                background-color: #ff6347;
            }
            label {
                margin: 10px 0;
                display: block;
            }
        </style>
    </head>
    <body>
        <h1>Vehicle Detection</h1>
        <form method="POST" action="/detect" enctype="multipart/form-data">
            <label for="image_url">Image URL:</label>
            <input type="text" name="image_url" id="image_url" placeholder="Enter image URL">
            <label for="image">Or upload an image:</label>
            <input type="file" name="image" id="image">
            <input type="submit" value="Detect Vehicles">
        </form>
    </body>
    </html>
    '''
    return render_template_string(html_form)

@app.route('/detect', methods=['POST'])
def detect_vehicles():
    
    if 'image' in request.files:
        # Read the uploaded image file
        image_file = request.files['image']
        image = Image.open(image_file)
    else:
        # Get the image URL from the form
        image_url = request.form.get('image_url')
        if not image_url:
            return "No image URL or file provided", 400

        # Fetch the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

    image = image.resize((450, 250))
    image_ar = np.array(image)

    # Convert the image to grayscale
    grey = cv2.cvtColor(image_ar, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Dilation and closing operations
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Load Haar Cascade for car detection
    car_cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)

    # Detect cars and draw rectangles
    for (x, y, w, h) in cars:
        cv2.rectangle(image_ar, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Load Haar Cascade for bus detection
    bus_cascade_src = 'bus_front.xml'
    bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
    buses = bus_cascade.detectMultiScale(closing, 1.1, 1)

    # Detect buses and draw rectangles
    for (x, y, w, h) in buses:
        cv2.rectangle(image_ar, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert the image back to PIL format
    output_image = Image.fromarray(image_ar)

    # Save the output image to a BytesIO object
    img_bytes = BytesIO()
    output_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Return the image
    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
