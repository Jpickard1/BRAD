import React, { useState, useEffect } from 'react';
import PopUpApiEntry from './PopUpApiEntry'
import RagFileInput from './RagFileInput';
import ThemeChangeButton from './ThemeChange';

function RightSideBar({ setColorScheme }) {
//  const [llmChoice, setLlmChoice] = useState('GPT-4');  // Default LLM choice
  const [ragDatabase, setRagDatabase] = useState('Wikipedia');  // Default RAG choice
  const [llmChoice, setLlmChoice] = useState("gpt-3.5-turbo-0125");  // Default to GPT-4
  const [isApiEntryVisible, setIsApiEntryVisible] = useState(false); // Only if you want to show/hide the popup
  const [availableDatabases, setAvailableDatabases] = useState([]);  // Hold available databases

  // Function to handle LLM change
  const handleLlmChange = async (event) => {
    const llmChoice = event.target.value;  // Extract the value from the event

    console.log(`Setting LLM to: ${llmChoice}`);
    
    try {
      const response = await fetch('/api/llm/set', {
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


  /*
        <div className="right-side-bar">
        <button onClick={handleOpenApiEntry}>Enter API Key</button>
        {isApiEntryVisible && <PopUpApiEntry onClose={handleCloseApiEntry} />}
      </div>
  */

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

      <RagFileInput />
      <ThemeChangeButton />
      
    </div>

  );
}

export default RightSideBar;
