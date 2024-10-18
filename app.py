# STANDARD python imports
import os
import logging

# Imports for building RESTful API
from flask import Flask
from BRAD.endpoints import bp as endpoints_bp  # Import the Blueprint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up app
app = Flask(__name__)

print(f"{app.root_path=}")

# Data directories setup
DATA_FOLDER = os.path.join(app.root_path, 'data')
UPLOAD_FOLDER = os.path.join(DATA_FOLDER, 'uploads')
DATABASE_FOLDER = os.path.join(DATA_FOLDER, 'RAG_Database')
print(f"{DATA_FOLDER=}")
print(f"{UPLOAD_FOLDER=}")
print(f"{DATABASE_FOLDER=}")

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER

# Register the Blueprint for the endpoints
app.register_blueprint(endpoints_bp)

if __name__ == "__main__":
    app.run(debug=True)
