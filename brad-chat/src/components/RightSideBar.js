import React, { useState, useEffect } from 'react';
import PopUpApiEntry from './PopUpApiEntry'
import RagFileInput from './RagFileInput';
import ThemeChangeButton from './ThemeChange';
import UsageStatistics from './UsageStatistics';

function RightSideBar({ setColorScheme, usageCalls, usageFees }) {
//  const [llmChoice, setLlmChoice] = useState('GPT-4');  // Default LLM choice
  const [ragDatabase, setRagDatabase] = useState('Wikipedia');  // Default RAG choice
  const [llmChoice, setLlmChoice] = useState("gpt-3.5-turbo-0125");  // Default to GPT-4
  const [isApiEntryVisible, setIsApiEntryVisible] = useState(false); // Only if you want to show/hide the popup
  const [availableDatabases, setAvailableDatabases] = useState([]);  // Hold available databases
  const [availableLLMs, setAvailableLLMs] = useState([]);  // State to hold fetched LLM models
  const [loading, setLoading] = useState(true);  // State to track loading

  // Fetch available LLM models on component mount
  // Fetch available LLM models on component mount
  useEffect(() => {
    const fetchLLMs = async () => {
      try {
        console.log('Fetching available LLM models...');
        const response = await fetch('/api/llm/get');  // Fetch from your new endpoint
        console.log('Response:', response);  // Log the entire response
        const data = await response.json();
        
        console.log('Response data:', data);  // Log the entire response

        if (data.success && Array.isArray(data.models)) {
          setAvailableLLMs(data.models);  // Update available LLM models
          setLoading(false);  // Stop loading once data is fetched
        } else {
          console.error("Failed to fetch LLM models or incorrect response format.");
          /* TODO: handle this error */
        }
      } catch (error) {
        console.error('Error fetching LLM models:', error);
        /* TODO: handle fetch error */
      }
    };

    fetchLLMs();  // Call the function to fetch LLM models
  }, []);  // Empty dependency array means this will run only once when the component mounts


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
          {availableLLMs.length > 0 ? (
            availableLLMs.map((model, index) => (
              <option key={index} value={model}>
                {model}
              </option>
            ))
          ) : (
            <option value="" disabled>Loading models...</option>
          )}
        </select>
      </div>


      <RagFileInput />
      <ThemeChangeButton />
      <div className="usage-stats">
          <h3>Usage Statistics</h3>
          <p><b>Session Calls: </b>{usageCalls}</p>
          <p><b>Session Usage Fee: </b>{usageFees}</p>
      </div>
      
    </div>

  );
}

export default RightSideBar;
