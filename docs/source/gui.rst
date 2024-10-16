The GUI for BRAD uses the python package as a backend and deploys a local server with the following structure:

- **Backend**: A Flask API that handles requests and communicates with the Python-based BRAD package.
- **Frontend**: A React GUI that provides an interactive user interface for sending queries and displaying responses.

Backend API (Flask)
-------------------
The Flask API serves as the backend, enabling communication between the `Agent` logic and the front-end React GUI. This API exposes the following endpoints:

- `/invoke`: accepts POST requests containing user inputs or messages.
- `/configure`: accepts POST requests containing settings that update the `Agent` configurations
- `/pdfs`: accepts POST requests with JSON payloads containing information to build a RAG database from PDFs located on the users system.

The API processes the messages using the logic in the `Agent` class and returns a JSON response to the frontend.

React JS GUI
------------
The React frontend offers a graphical user interface for users to interact with the chatbot. Users can send messages, build RAG databases, or change the system configurations, which the frontend captures and sends to the Flask API. Upon receiving the chatbot's response, the GUI updates the chat interface to display both user and bot messages, facilitating a smooth and engaging conversation experience.

API Enpoints
------------

.. automodule:: app
   :members:
   :undoc-members:

