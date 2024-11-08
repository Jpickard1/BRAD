# STANDARD python imports
import os
import logging
import shutil
import logging

# Imports for building RESTful API
from flask import Flask
from flask_caching import Cache
from BRAD.endpoints import bp as endpoints_bp  # Import the Blueprint
from BRAD.endpoints import set_globals, initiate_start
from BRAD.agent import Agent
from BRAD.constants import TOOL_MODULES


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configue caching
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})


# def create_app():
# Directory structure
UPLOAD_FOLDER = '/usr/src/uploads'
DATABASE_FOLDER = '/usr/src/RAG_Database/'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Redundant
TOOL_MODULES = TOOL_MODULES


# Set up app
app = Flask(__name__)

print(f"{app.root_path=}")

# set cache for the app
cache.init_app(app)
CACHE = cache

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

# app.config['DATA_FOLDER'] = DATA_FOLDER
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['DATABASE_FOLDER'] = DATABASE_FOLDER
set_globals(DATA_FOLDER, UPLOAD_FOLDER, DATABASE_FOLDER, ALLOWED_EXTENSIONS, TOOL_MODULES, CACHE)

initiate_start()

# Register the Blueprint for the endpoints
app.register_blueprint(endpoints_bp)

# removing this in favor of factory
# app.config['agent'] = brad
# if __name__ == "__main__":
#     app = create_app()
#     if not os.getenv("GENERATING_DOCS"):
#         app.run(debug=True)

