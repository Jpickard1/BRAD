import React, { useEffect, useState } from 'react';
import SessionList from './SessionList';
import "highlight.js/styles/github.css";

function ChatSessions({ setMessages }) {
  const [chatsessions, setSessions] = useState([]);

  // Function to handle session removal
  const handleRemoveSession = async (sessionId) => {
    try {
      const response = await fetch('/api/remove_session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: sessionId })
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      console.log(`Session removed: ${sessionId}`);
      
      // Fetch the updated sessions after removal
      await fetchSessions();

    } catch (error) {
      console.error('Error removing session:', error);
    }
  };


  // Function to handle session change
  const handleSessionChange = async (sessionId) => {
    console.log(`Changing to session: ${sessionId}`);
    const response = await fetch('/api/change_session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: sessionId })
    });
  
    if (response.ok) {
      const data = await response.json();  // Get the response data
      const formattedMessages = data.display.map((text, index) => ({
        id: index, // Use index as a temporary unique key (for demo purposes)
        text: text,
        sender: index % 2 === 0 ? 'user' : 'bot' // Example: alternate between 'user' and 'bot'
      }));
  
      setMessages(formattedMessages); // Set the formatted messages
      console.log(`Session changed. Chat history:`, formattedMessages);
  
    } else {
      console.error('Error changing session:', await response.json());
    }
  };
  

  const fetchSessions = async () => {
    console.log("1. setting chatsessions", chatsessions);

    try {
      const response = await fetch('/api/open_sessions', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error('API request failed with status ${response.status}');
      }

      console.log("2. Received chat sessions response:", response);
      const result = await response.json();
      console.log("3. result", result);

      const newSessions = result['open_sessions'].map((session, index) => ({
        id: `${Date.now()}-${index}`,
        text: session,
        sender: 'bot',
      }));

      setSessions(newSessions);

    } catch (error) {
      console.error('Error:', error);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []); // Empty dependency array to run once on mount

  return (
    <div className="chat-sessions">
      <p>Chat Sessions</p>
      <SessionList messages={chatsessions} onRemoveSession={handleRemoveSession} onChangeSession={handleSessionChange} />
    </div>
  );
}

export default ChatSessions;
