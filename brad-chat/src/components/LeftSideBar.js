import React, { useEffect, useState } from 'react';
// import MessageList from './MessageList';
// import MessageInput from './MessageInput';

import ChatSessions from './ChatSessions';

import "highlight.js/styles/github.css";
import hljs from "highlight.js";
// import RagFileInput from './RagFileInput';

function LeftSideBar() {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    hljs.highlightAll();
  }, [messages]);

  const handleSendMessage = async (message) => {
    setMessages([...messages, { id: Date.now(), text: message, sender: 'user' }]);
    
    console.log("1. setting messages", message)

    let data = {"message": message}
    try {
        // Call the backend API using fetch
        const response = await fetch('/invoke', {
          method: 'POST', // or 'GET' depending on your API
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(data), // Send the form data as a JSON payload
        });
        console.log("2. got response", response)
  
        // Parse the JSON response
        const result = await response.json();
        let bot_response = result['response']
  
        // Handle the response
         // Set the API response to the state
        setMessages((messages) => [...messages, { id: Date.now(), text: bot_response, sender: 'bot' }]);
    } catch (error) {
        console.error('Error:', error);
    }

  };

  return (
    <div className="left-sidebar">
      <ChatSessions />
    </div>
  );
}

export default LeftSideBar;