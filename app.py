import os
import re
from flask import Flask, render_template, request
import nltk
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Set the allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Define a global variable for the loaded model
model = None

def load_machine_learning_model():
    global model
    try:
        model_path = 'model.h5'  # Adjust the model file path
        model = load_model(model_path)  # Load the model
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        model = None  # Set model to None if loading fails

# Load the model when the application starts
load_machine_learning_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define your data preprocessing function here
def preprocess(txt):  # Changed the argument name from 'data' to 'txt'
    # Convert all characters in the string to lower case
    txt = txt.lower()
    # Remove non-english characters, punctuation, and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub('http\S+\s*', ' ', txt)  # Remove URLs
    txt = re.sub('RT|cc', ' ', txt)  # Remove RT and cc
    txt = re.sub('#\S+', '', txt)  # Remove hashtags
    txt = re.sub('@\S+', '  ', txt)  # Remove mentions
    txt = re.sub('\s+', ' ', txt)  # Remove extra whitespace
    # Tokenize words
    txt = nltk.tokenize.word_tokenize(txt)
    # Remove stop words
    txt = [w for w in txt if not w in nltk.corpus.stopwords.words('english')]
    
    return ' '.join(txt)  # Corrected the return statement

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse_resume', methods=['GET', 'POST'])
def parse_resume():
    if request.method == 'POST':
        # Check if a file is selected
        if 'file' not in request.files:
            return render_template('index.html', message='No file selected.')
        
        uploaded_file = request.files['file']
        
        # Check if a valid file type is uploaded
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join('uploads', filename)
            uploaded_file.save(file_path)

            # Ensure that the model is loaded
            if model is None:
                return render_template('index.html', message='Model not loaded.')

            # Read and preprocess the resume data
            with open(file_path, 'rb') as file:
                resume_data = file.read()  # Read the uploaded file
                # Perform data preprocessing
                preprocessed_data = preprocess(resume_data.decode('utf-8'))  # Decode bytes to string
                
                # Convert the preprocessed data into a numpy array
                preprocessed_array = np.array([preprocessed_data])

                # Check data type
                if preprocessed_array.dtype != model.input_dtype:
                    return render_template('index.html', message='Invalid data type.')

                # Check data shape
                if preprocessed_array.shape != model.input_shape[1:]:
                    return render_template('index.html', message='Invalid data shape.')

                # Parse the resume using the model
                try:
                    prediction = model.predict(preprocessed_array)
                except Exception as e:
                    return render_template('index.html', message='Error during prediction.')

                # Render the result template with the prediction
                return render_template('result.html', prediction=str(prediction))

        else:
            return render_template('index.html', message='Invalid file type.')

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

