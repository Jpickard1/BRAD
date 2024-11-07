import React, { useState } from 'react';
import ChatContainer from './components/ChatContainer';
import LeftSideBar from './components/LeftSideBar';
import LeftSideBarButton from './components/LeftSidebarButton';
import RightSideBar from './components/RightSideBar';
import handleSessionStorage from './utils/Utils'
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);  // Messages now managed in App
  const [showRightSidebar, setShowRightSidebar] = useState(false);  // Manage right sidebar visibility
  const [colorScheme, setColorScheme] = useState('light');  // Manage color scheme state
  const [usageCalls, setUsageCalls] = useState(localStorage.getItem('llm-calls'));  // Manage color scheme state
  const [usageFees, setUsageFees] = useState(localStorage.getItem('api-fees'));  // Manage color scheme state

  const handleSessionStorage_old = (session) => {
      let current_session = localStorage.getItem('current-session')
      if (current_session != session){
        localStorage.setItem('current-session', session)
      }
  }

  const getActiveSession = () => {
      let active_session = localStorage.getItem('current-session')
      return active_session
  }

  const setStatistics = (bot_usage) => {

      let api_fees = localStorage.getItem('api-fees')
      let llm_calls = localStorage.getItem('llm-calls')
      let updated_api_fees
      let updated_llm_calls

      // updating the localstorage with new usage stats
      if (api_fees && !isNaN(api_fees)){
        updated_api_fees = parseFloat(api_fees) + parseFloat(bot_usage['api-fees'])
      }
      else{
        updated_api_fees = parseFloat(bot_usage['api-fees'])
      }
      if (llm_calls && !isNaN(llm_calls) ){
        updated_llm_calls = Number(llm_calls) + Number(bot_usage['llm-calls'])
      }
      else{
        updated_llm_calls = Number(bot_usage['llm-calls'])
      }

      localStorage.setItem('api-fees', updated_api_fees)
      localStorage.setItem('llm-calls', updated_llm_calls)
      setUsageFees(updated_api_fees)
      setUsageCalls(updated_llm_calls)
  }

  const handleSendMessage = async (message) => {
    // Add the user's message to the message list
    setMessages([...messages, { id: Date.now(), text: message, process: null, sender: 'user' }]);
  
    let data = { message };
  
    try {
      let current_session = getActiveSession()
      if (current_session != null) {
        data.session = current_session
      }
      // Call the backend API using fetch
      const response = await fetch('/api/invoke', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
  
      // Parse the JSON response
      const result = await response.json();
      let bot_response = result['response'] || '[No response]';
      let bot_log = result['response-log'] || '[No log]'; // You mentioned response-log for the process
      let updated_session = result['session-name']

      let bot_usage = result['llm-usage']
      setStatistics(bot_usage)
      handleSessionStorage(updated_session)

      console.log(`bot_response= ${bot_response}`)
      console.log(`bot_log= ${bot_log}`)
      // Add the bot's response to the message list, including both text and process
      setMessages((prevMessages) => [
        ...prevMessages, 
        { id: Date.now(), text: bot_response, process: bot_log, sender: 'bot' }
      ]);
      
    } catch (error) {
      console.error('Error:', error);
    }
  };  

  const toggleRightSidebar = () => {
    setShowRightSidebar(!showRightSidebar);  // Toggle sidebar visibility
  };

  return (
    <div className="App">
      <header className="App-header">
        <LeftSideBarButton />       
        <h2>{String.fromCodePoint('0x1f916')} BRAD </h2>
        <button className="sidebar-toggle" onClick={toggleRightSidebar}>
          {String.fromCodePoint('0x2699')}
          {/* {showRightSidebar ? 'Close Settings' : 'Open Settings'} */}
        </button>  {/* Button to toggle RightSideBar */}
      </header>
      <div className="Main-container">
        <LeftSideBar setMessages={setMessages}/>
        {/* Pass messages and handleSendMessage to ChatContainer */}
        <ChatContainer messages={messages} onSendMessage={handleSendMessage} />
        {showRightSidebar && <RightSideBar setColorScheme={setColorScheme} usageCalls={usageCalls} usageFees={usageFees} />} {/* Conditionally render RightSideBar */}

      </div>
    </div>
  );
}

export default App;
