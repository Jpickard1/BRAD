import React, { useEffect, useState } from 'react';
import SessionList from './SessionList';
import "highlight.js/styles/github.css";

function ChatSessions() {
  const [chatsessions, setSessions] = useState([]);

  /* Whenever this component is mounted to the page, useEffect runs. This is React specific */
  useEffect(() => {
    const fetchSessions = async () => {
      console.log("1. setting chatsessions", chatsessions);

      try {
        // Call the backend API using fetch 
        const response = await fetch('/api/open_sessions', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        // Check if response is OK
        if (!response.ok) {
          throw new Error(`API request failed with status ${response.status}`);
        }
  
        console.log("2. Received chat sessions response:", response);
   
        // Parse the JSON response
        const result = await response.json();
        
        console.log("3. result", result);

        // Collect new sessions and update the state
        const newSessions = result['open_sessions'].map((session, index) => ({
          id: `${Date.now()}-${index}`,  // Ensure unique IDs for each session
          text: session,
          sender: 'bot',
        }));

        // Set new sessions
        setSessions((prevSessions) => [...prevSessions, ...newSessions]);

      } catch (error) {
        console.error('Error:', error);
      }
    };

    fetchSessions();
  }, []); // Empty dependency array to run once on mount

  return (
    <div className="chat-sessions">
      <p>Chat Sessions</p>
      <SessionList messages={chatsessions} />
    </div>
  );
}

export default ChatSessions;
