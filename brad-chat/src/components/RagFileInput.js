import React, { useState, useEffect } from 'react';
import axios from 'axios';

function RagFileInput() {
    const [files, setFiles] = useState([]);
    const [dbName, setDbName] = useState('');  // State to hold the name input
    const [uploadProgress, setUploadProgress] = useState(0);
    const [ragDatabase, setRagDatabase] = useState('Wikipedia');  // Default RAG choice
    // const [llmChoice, setLlmChoice] = useState("gpt-3.5-turbo-0125");  // Default to GPT-4
    // const [isApiEntryVisible, setIsApiEntryVisible] = useState(false); // Only if you want to show/hide the popup
    const [availableDatabases, setAvailableDatabases] = useState([]);  // Hold available databases

    // New state variables for Search Settings
    const [numArticles, setNumArticles] = useState('');
    const [retrievalMechanism, setRetrievalMechanism] = useState('MMR');
    const [contextualCompression, setContextualCompression] = useState(false);

    function handleChange(event) {
        setFiles(event.target.files);
    }

    const handleNameChange = (e) => {
        setDbName(e.target.value);  // Update the state when name input changes
    };

    // Handlers for new Search Settings
    const handleNumArticlesChange = async (e) => {
      if (e.key != "Enter") {
        return 
      }

        const newValue = e.target.value; // Get the updated value from the input field
        setNumArticles(newValue); // Update the state with the new value
    
        // Ensure the newValue is a valid number before proceeding
        if (!newValue || isNaN(newValue) || parseInt(newValue, 10) < 1) {
            alert("Please enter a valid number greater than or equal to 1.");
            return;
        }
    
        try {
            // API endpoint URL
            const url = '/api/configure/RAG/numberArticles';
            console.log("url:", url);
            // JSON payload to send
            const payload = {
                session: localStorage.getItem('current-session'), // Pass your session identifier here
                number_articles: parseInt(newValue, 10),
            };
            console.log("payload:", payload);
    
            // Make the POST request
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });
            console.log("response:", response);
    
            // Handle the response
            if (response.ok) {
                const data = await response.json();
                console.log("Configuration updated successfully:", data);
                alert("Number of articles retrieved updated successfully!");
            } else {
                const errorData = await response.json();
                console.error("Error updating configuration:", errorData);
                alert(`Failed to update configuration: ${errorData.message || 'Unknown error'}`);
            }
        } catch (error) {
            console.error("An error occurred while updating configuration:", error);
            alert("An error occurred while updating the number of articles retrieved.");
        }
    };
  
    const handleRetrievalMechanismChange = async (e) => {
      let newretrievalmechanism = e.target.value 
      setRetrievalMechanism(newretrievalmechanism);
      let data = {
        session: localStorage.getItem('current-session'),
        "search_method": newretrievalmechanism
      }

      try{
        const response = await fetch('api/configure/RAG/searchMechanism', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(data),
        });

            if (response.ok) {
                const data = await response.json();
                console.log("Search mechanism updated successfully:", data);
            } else {
                const errorData = await response.json();
                console.error("Error updating Search mechanism:", errorData);
            }
      }
      catch{
        console.log("error occured setting Search mechanism")
      }
    }

    const handleContextualCompressionToggle = async () => {
      let newcontextualCompression = !contextualCompression 
      setContextualCompression(newcontextualCompression);

      let data = {
        session: localStorage.getItem('current-session'),
        "contextual_compression": newcontextualCompression
      }

      try{
        const response = await fetch('/api/configure/RAG/contextualCompression', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(data),
        });
            if (response.ok) {
                const data = await response.json();
                console.log("contextual compression updated successfully:", data);
            } else {
                const errorData = await response.json();
                console.error("Error updating Contextual compression:", errorData);
            }
      }
      catch{
        console.log("error occured setting contextual compression")
      }
    }


    const handleSubmit = async (e) => {
        e.preventDefault();
        const url = '/api/databases/create';
        const formData = new FormData();

        if (files == null) {
          alert("please attach a valid file!")
          return
        }

        // Append files to FormData
        for (let i = 0; i < files.length; i++) {
            formData.append('rag_files', files[i]);
        }

        // Append the 'name' field to FormData
        formData.append('name', dbName);

        const config = {
            headers: {
                'content-type': 'multipart/form-data',
            },
            onUploadProgress: function(progressEvent) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                setUploadProgress(percentCompleted);
            }
        };

        try {
          // Step 1: Upload the files and create a new database
          const response = await axios.post(url, formData, config);
          console.log(response.data);
  
          // Step 2: Fetch the updated list of available databases
          const fetchResponse = await fetch('/api/databases/available');
          if (fetchResponse.ok) {
              const data = await fetchResponse.json();
              setAvailableDatabases(data.databases);
  
              // Step 3: Set the newly created database as the current RAG database
              if (dbName && !data.databases.includes(dbName)) {
                  console.warn('Newly created database not found in available databases.');
              } else {
                  setRagDatabase(dbName);  // Set the new database as the default choice
              }
  
              // Optionally, log the updated list of databases
              console.log('Updated databases:', data.databases);
              alert('Updated databases: ' + data.databases.join(", "));
          } else {
              console.error('Failed to fetch updated databases');
              alert('Failed to fetch updated databases');
          }
      } catch (error) {
          console.error("Error during file upload or fetching databases: ", error);
      }
  
    };

    // Fetch available databases from the Flask API when the component loads
    useEffect(() => {
      const fetchDatabases = async () => {
        try {
          const response = await fetch('/api/databases/available');
          if (response.ok) {
            const data = await response.json();
            console.log(`data.databases: ${data.databases}`);
            setAvailableDatabases(data.databases);
            setRagDatabase(data.databases[0]);  // Set the first available DB as default
          } else {
            console.error('Failed to fetch databases');
          }
        } catch (error) {
          console.error('Error fetching databases:', error);
        }
      };
  
      fetchDatabases();
    }, []);  // Run this effect only once when the component mounts
  
    // Function to handle RAG database change
    const handleRagChange = async (event) => {
      const selectedDatabase = event.target.value;
      console.log(`Setting RAG database to: ${selectedDatabase}`);
  
      try {
        const response = await fetch('/api/databases/set', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ database: selectedDatabase })  // Send selected DB in the request body
        });
        console.log(`Database response: `, response);
        if (response.ok) {
          const data = await response.json();
          console.log(`Database set successfully:`, data.message);
          setRagDatabase(selectedDatabase);  // Update the current selection
        } else {
          const errorData = await response.json();
          console.error('Error setting database:', errorData.message);
        }
      } catch (error) {
        console.error('Error during database request:', error);
      }
    };
  

    return (
      <div className='sidebar-setting'>
        
        <div className="setting-option-choose-db">
          <label htmlFor="rag-database"><b>Choose Database:</b></label>
          <select id="rag-database" value={ragDatabase} onChange={handleRagChange}>
            {availableDatabases.map((db, index) => (
              <option key={index} value={db}>{db}</option>
            ))}
          </select>
        </div>

        <label><b>Upload Files:</b></label>
        <div className='rag-file-upload'>
          <form onSubmit={handleSubmit} className="rag-file-input">
                <input
                    id="rag-db-name"
                    type="text"
                    placeholder="Enter database name"
                    value={dbName}
                    onChange={handleNameChange}
                />
                <input id="rag-file-input" type="file" name="rag_files" onChange={handleChange} multiple />
                <label  for="rag-file-input">Choose File</label>
                {Array.from(files).length > 0 && (
                  <div className="file-names">
                    {/* <strong>Selected Files:</strong> */}
                    <ul>
                      {Array.from(files).map((file, index) => (
                        <li key={index}>{file.name}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <button id="rag-file-submit" type="submit">upload {String.fromCodePoint('0x25B2')}</button>
                <progress value={uploadProgress} max="100"></progress>
            </form>
        </div>

        {/* Search Settings */}
        <label><b>Search Settings:</b></label>
        <div className="search-settings">
          {/* Number of Articles Retrieved */}
          <div className="setting-option">
            <label htmlFor="num-articles"><b>Number of Articles Retrieved:</b></label>
            <input
              id="num-articles"
              type="number"
              min="1"
              value={numArticles}
              onChange={(e) => setNumArticles(e.target.value)} // Updates the state without triggering the API call
              onKeyDown={handleNumArticlesChange} // Trigger API call only on Enter
              placeholder="Enter a number"
            />
          </div>

          {/* Retrieval Mechanism */}
          <div className="setting-option">
            <label htmlFor="retrieval-mechanism"><b>Retrieval Mechanism:</b></label>
            <select
              id="retrieval-mechanism"
              value={retrievalMechanism}
              onChange={handleRetrievalMechanismChange}
            >
              <option value="MMR">MMR</option>
              <option value="similarity">Similarity</option>
              <option value="multiquery">Multiquery</option>
            </select>
          </div>

          {/* Contextual Compression */}
          <div className="setting-option">
            <label htmlFor="contextual-compression"><b>Contextual Compression:</b></label>
            <input
              id="contextual-compression"
              type="checkbox"
              checked={contextualCompression}
              onChange={handleContextualCompressionToggle}
            />
            <span>{contextualCompression ? 'On' : 'Off'}</span>
          </div>
        </div>
      </div>
    );
}

export default RagFileInput;
