import React, { useState, useEffect } from 'react';
import PopUpApiEntry from './PopUpApiEntry'


function RightSideBar({ setColorScheme }) {
//  const [llmChoice, setLlmChoice] = useState('GPT-4');  // Default LLM choice
  const [ragDatabase, setRagDatabase] = useState('Wikipedia');  // Default RAG choice
  const [llmChoice, setLlmChoice] = useState("gpt-3.5-turbo-0125");  // Default to GPT-4
  const [isApiEntryVisible, setIsApiEntryVisible] = useState(false); // Only if you want to show/hide the popup
  const [availableDatabases, setAvailableDatabases] = useState([]);  // Hold available databases

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

  // Function to handle LLM change
  const handleLlmChange = async (event) => {
    const llmChoice = event.target.value;  // Extract the value from the event

    console.log(`Setting LLM to: ${llmChoice}`);
    
    try {
      const response = await fetch('/api/set_llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ llm: llmChoice })  // Sending LLM choice in the request body
      });

      if (response.ok) {
        const data = await response.json();  // Get the response data
        console.log(`LLM set successfully:`, data.message);
        setLlmChoice(llmChoice);

      } else {
        const errorData = await response.json();  // Parse error message from response
        console.error('Error setting LLM:', errorData.message);
        
        /* TODO: make a warning display that the LLM change didn't work*/
      }
    } catch (error) {
      console.error('Error during LLM request:', error);

        /* TODO: implement error handeling */

    }
  };

  // Function to handle RAG database change
  const handleRagChange = async (event) => {
    const selectedDatabase = event.target.value;
    console.log(`Setting RAG database to: ${selectedDatabase}`);

    try {
      const response = await fetch('/databases/set', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ database: selectedDatabase })  // Send selected DB in the request body
      });

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

  const handleColorSchemeChange = (event) => {
    const scheme = event.target.value;
    setColorScheme(scheme);  // Pass the selected color scheme back to App
  };


  const handleOpenApiEntry = () => {
      setIsApiEntryVisible(true);
  };

  const handleCloseApiEntry = () => {
      setIsApiEntryVisible(false);
  };


  return (
    <div className="sidebar-right">
      <h2>Settings</h2>

      <div className="setting-option">
        <label htmlFor="llm-choice">Choose LLM:</label>
        <select id="llm-choice" value={llmChoice} onChange={handleLlmChange}>
          <option value="gpt-3.5-turbo-0125">GPT-3.5</option>
          <option value="gpt-4o-mini-2024-07-18">GPT-4</option>
        </select>
      </div>

      <div className="setting-option">
        <label htmlFor="rag-database">Choose RAG Database:</label>
        <select id="rag-database" value={ragDatabase} onChange={handleRagChange}>
          {availableDatabases.map((db, index) => (
            <option key={index} value={db}>{db}</option>
          ))}
        </select>
      </div>

      <div className="setting-option">
        <label htmlFor="color-scheme">Color Scheme:</label>
        <select id="color-scheme" onChange={handleColorSchemeChange}>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>

      <div className="right-side-bar">
        <button onClick={handleOpenApiEntry}>Enter API Key</button>
        {isApiEntryVisible && <PopUpApiEntry onClose={handleCloseApiEntry} />}
      </div>
    </div>

  );
}

export default RightSideBar;
