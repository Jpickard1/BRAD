import React, { useEffect, useState } from 'react';
import MessageList from './MessageList';
// import MessageInput from './MessageInput';

import "highlight.js/styles/github.css";
// import hljs from "highlight.js";
// import RagFileInput from './RagFileInput';

function ChatSessions() {

  const [chatsessions, setSessions] = useState([]);

  /* Whenever this component is mounted to the page, useEffect runs. This is React specific */
  useEffect(() => {fetchSessions()}, []);

  const fetchSessions = async () => {
    
    console.log("1. setting chatsessions", chatsessions)

    try {
        // Call the backend API using fetch
        const response = await fetch('/open_sessions', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        });
        console.log("2. got chat sessions", response)
  
        // Parse the JSON response
        const result = await response.json();
        
        console.log("3. result", result)

        // Handle the response
        // Set the API response to the state
        for (let i = 0; i < result['open_sessions'].length / 2; i++) {
            setSessions((chatsessions) => [...chatsessions, { id: Date.now(), text: result['open_sessions'][i], sender: 'bot' }]);
        }

    } catch (error) {
        console.error('Error:', error);
    }

  };

  return (
    <div className="chat-sessions">
      <p>Chat Sessions</p>
      <MessageList messages={chatsessions} />
    </div>
  );
}

export default ChatSessions;
