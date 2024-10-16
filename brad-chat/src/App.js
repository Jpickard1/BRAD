import React, { useState } from 'react';
import ChatContainer from './components/ChatContainer';
import LeftSideBar from './components/LeftSideBar';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);  // Messages now managed in App

  const handleSendMessage = async (message) => {
    // Add the user's message to the message list
    setMessages([...messages, { id: Date.now(), text: message, sender: 'user' }]);

    let data = { "message": message };

    try {
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
      let bot_response = result['response'];

      // Add the bot's response to the message list
      setMessages((prevMessages) => [
        ...prevMessages, 
        { id: Date.now(), text: bot_response, sender: 'bot' }
      ]);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h2>{String.fromCodePoint('0x1f916')} BRAD </h2>
      </header>
      <div className="Main-container">
        <LeftSideBar setMessages={setMessages}/>
        {/* Pass messages and handleSendMessage to ChatContainer */}
        <ChatContainer messages={messages} onSendMessage={handleSendMessage} />
      </div>
    </div>
  );
}

export default App;
