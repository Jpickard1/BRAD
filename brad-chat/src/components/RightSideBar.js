import React, { useState } from 'react';

function RightSideBar({ setColorScheme }) {
//  const [llmChoice, setLlmChoice] = useState('GPT-4');  // Default LLM choice
  const [ragDatabase, setRagDatabase] = useState('Wikipedia');  // Default RAG choice
  const [llmChoice, setLlmChoice] = useState("gpt-4o-mini-2024-07-18");  // Default to GPT-4

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


  const handleRagChange = (event) => {
    setRagDatabase(event.target.value);
  };

  const handleColorSchemeChange = (event) => {
    const scheme = event.target.value;
    setColorScheme(scheme);  // Pass the selected color scheme back to App
  };

  return (
    <div className="sidebar-right">
      <h2>Settings</h2>

      <div className="setting-option">
        <label htmlFor="llm-choice">Choose LLM:</label>
        <select id="llm-choice" value={llmChoice} onChange={handleLlmChange}>
          <option value="gpt-4o-mini-2024-07-18">GPT-4*</option>
          <option value="gpt-3.5-turbo-0125">GPT-3.5*</option>
          <option value="meta / llama3-70b-instruct">Llama3*</option>
        </select>
      </div>

      <div className="setting-option">
        <label htmlFor="rag-database">Choose RAG Database:</label>
        <select id="rag-database" value={ragDatabase} onChange={handleRagChange}>
          <option value="Wikipedia">Wikipedia</option>
          <option value="PubMed">PubMed</option>
          <option value="ArXiv">ArXiv</option>
        </select>
      </div>

      <div className="setting-option">
        <label htmlFor="color-scheme">Color Scheme:</label>
        <select id="color-scheme" onChange={handleColorSchemeChange}>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>
    </div>
  );
}

export default RightSideBar;
