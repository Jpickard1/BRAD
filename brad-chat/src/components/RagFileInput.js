import React, { useState, useEffect } from 'react';
import axios from 'axios';

function RagFileInput() {
    const [files, setFiles] = useState();
    const [dbName, setDbName] = useState('');  // State to hold the name input
    const [uploadProgress, setUploadProgress] = useState(0);
    const [ragDatabase, setRagDatabase] = useState('Wikipedia');  // Default RAG choice
    // const [llmChoice, setLlmChoice] = useState("gpt-3.5-turbo-0125");  // Default to GPT-4
    // const [isApiEntryVisible, setIsApiEntryVisible] = useState(false); // Only if you want to show/hide the popup
    const [availableDatabases, setAvailableDatabases] = useState([]);  // Hold available databases

    function handleChange(event) {
        setFiles(event.target.files);
    }

    const handleNameChange = (e) => {
        setDbName(e.target.value);  // Update the state when name input changes
    };

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
      <div>
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
                <button id="rag-file-submit" type="submit">upload {String.fromCodePoint('0x25B2')}</button>
                <progress value={uploadProgress} max="100"></progress>
            </form>
        </div>
        <div className="setting-option-choose-db">
          <label htmlFor="rag-database">Choose RAG Database:</label>
          <select id="rag-database" value={ragDatabase} onChange={handleRagChange}>
            {availableDatabases.map((db, index) => (
              <option key={index} value={db}>{db}</option>
            ))}
          </select>
        </div>
      </div>
    );
}

export default RagFileInput;
