import React, { useState, useEffect } from 'react';
import PopUpApiEntry from './PopUpApiEntry'
import RagFileInput from './RagFileInput';
import ThemeChangeButton from './ThemeChange';

function RightSideBar({ setColorScheme, usageCalls, usageFees }) {
//  const [llmChoice, setLlmChoice] = useState('GPT-4');  // Default LLM choice
  const [ragDatabase, setRagDatabase] = useState('Wikipedia');  // Default RAG choice
  const [llmChoice, setLlmChoice] = useState("gpt-3.5-turbo-0125");  // Default to GPT-4
  const [isApiEntryVisible, setIsApiEntryVisible] = useState(false); // Only if you want to show/hide the popup
  const [availableDatabases, setAvailableDatabases] = useState([]);  // Hold available databases
  const [availableLLMs, setAvailableLLMs] = useState([]);  // State to hold fetched LLM models
  const [loading, setLoading] = useState(true);  // State to track loading
  const [isLLMVisible, setIsLLMVisible] = useState(false);
  const [isRAGVisible, setIsRAGVisible] = useState(false);
  const [apiKey, setApiKey] = useState('');

  const toggleLLMVisibility = () => {
    setIsLLMVisible(!isLLMVisible);
  };

  const toggleRAGVisibility = () => {
    setIsRAGVisible(!isRAGVisible);
  };

  const handleApiKeyChange = (e) => {
    setApiKey(e.target.value);
  };  

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

  return (
    <div className="sidebar-right">
      <h2>Settings</h2>

      {/* LLM Settings Section */}
      <div className="setting-group">
        <button className="settings-button" onClick={toggleLLMVisibility}>
          {isLLMVisible ? 'Hide LLM Settings' : 'Show LLM Settings'}
        </button>
        {isLLMVisible && (
          <div className="llm-settings">
            <div className="setting-option sidebar-setting">
              <label htmlFor="llm-choice"><b>Choose LLM:</b></label>
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
            <div className="usage-stats sidebar-setting">
              <h3>Usage Statistics</h3>
              <p><b>Session Calls: </b>{usageCalls}</p>
              <p><b>Session Usage Fee: </b>{usageFees}</p>
            </div>
            <div className="api-key-input sidebar-setting">
              <label htmlFor="api-key"><b>Enter API Key:</b></label>
              <input
                type="text"
                id="api-key"
                value={apiKey}
                onChange={handleApiKeyChange}
                placeholder="Enter your API key"
              />
            </div>
          </div>
        )}
      </div>

      {/* RAG Settings Section */}
      <div className="setting-group">
        <button className="settings-button" onClick={toggleRAGVisibility}>
          {isRAGVisible ? 'Hide RAG Settings' : 'Show RAG Settings'}
        </button>
        {isRAGVisible && (
          <div className="rag-settings">
            <RagFileInput />
          </div>
        )}
      </div>

      {/* Display Section */}
      <div className="setting-group">
        {/*<h3>Display Settings</h3>*/}
        <ThemeChangeButton />
      </div>

    </div>
  );    
}

export default RightSideBar;
