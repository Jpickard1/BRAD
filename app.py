# STANDARD python imports
import os
import logging
import shutil
import logging

# Imports for building RESTful API
from flask import Flask
from BRAD.endpoints import bp as endpoints_bp  # Import the Blueprint
from BRAD.endpoints import set_brad_instance, set_globals
from BRAD.agent import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_dirs_without_log(directory):
    # List only first-level subdirectories
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            log_file_path = os.path.join(subdir_path, 'log.json')
            
            # If log.json does not exist in the subdirectory, delete the subdirectory
            if not os.path.exists(log_file_path):
                shutil.rmtree(subdir_path)  # Recursively delete directory and its contents
                print(f"Deleted directory: {subdir_path}")

# def create_app():
# Directory structure
UPLOAD_FOLDER = '/usr/src/uploads'
DATABASE_FOLDER = '/usr/src/RAG_Database/'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

TOOL_MODULES = ['RAG']

brad = Agent(interactive=False, tools=TOOL_MODULES)
set_brad_instance(brad)

PATH_TO_OUTPUT_DIRECTORIES = brad.state['config'].get('log_path')
delete_dirs_without_log(PATH_TO_OUTPUT_DIRECTORIES)


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

# app.config['DATA_FOLDER'] = DATA_FOLDER
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['DATABASE_FOLDER'] = DATABASE_FOLDER
set_globals(DATA_FOLDER, UPLOAD_FOLDER, DATABASE_FOLDER, ALLOWED_EXTENSIONS, TOOL_MODULES, PATH_TO_OUTPUT_DIRECTORIES)

# Register the Blueprint for the endpoints
app.register_blueprint(endpoints_bp)
app.config['agent'] = brad
# if __name__ == "__main__":
#     app = create_app()
#     if not os.getenv("GENERATING_DOCS"):
#         app.run(debug=True)

