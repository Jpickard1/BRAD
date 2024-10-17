import React, { useEffect, useState } from 'react';
import "highlight.js/styles/github.css";
import Markdown from "marked-react";

function ChatSessions({ setMessages }) {
  const [chatsessions, setSessions] = useState([]);
  const [editingSessionId, setEditingSessionId] = useState(null); // Track the session being edited
  const [updatedText, setUpdatedText] = useState(""); // Track the updated session name

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

  const handleNewSession = async () => {
    try {
      const response = await fetch("/api/create_session", {
        method: "GET",
      });
      const result = await response.json();
  
      if (result.success) {
        // Handle success, such as refreshing the list of sessions
        console.log(result.message);

        // Update the session display
        // const data = await response.json();  // Get the response data
        console.log(`result: ${result}`);
        console.log(`result.success: ${result.success}`);
        console.log(`result.message: ${result.message}`);
        const formattedMessages = result.display.map((item, index) => {
          const [text, process] = item;  // Destructure the tuple
          console.log(`text: ${text}`);
          console.log(`process: ${process}`);
          return {
            id: index, // Use index as a temporary unique key (for demo purposes)
            text: text,
            process: process !== null ? process : [], // If the second element is null, set it to an empty array (or other default value)
            sender: index % 2 === 0 ? 'user' : 'bot' // Example: alternate between 'user' and 'bot'
          };
        });
    
        setMessages(formattedMessages); // Set the formatted messages
        console.log(`Session created. Chat history:`, formattedMessages);
  

      } else {
        console.error("Failed to create a new session:", result.message);
      }
      // Fetch updated sessions or update state locally
      await fetchSessions();
    } catch (error) {
      console.error("An error occurred while creating a new session:", error);
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
      console.log(`data: ${data}`);
      console.log(`data.success: ${data.success}`);
      console.log(`data.message: ${data.message}`);
      const formattedMessages = data.display.map((item, index) => {
        const [text, process] = item;  // Destructure the tuple
        console.log(`text: ${text}`);
        console.log(`process: ${process}`);
        return {
          id: index, // Use index as a temporary unique key (for demo purposes)
          text: text,
          process: process !== null ? process : [], // If the second element is null, set it to an empty array (or other default value)
          sender: index % 2 === 0 ? 'user' : 'bot' // Example: alternate between 'user' and 'bot'
        };
      });      
  
      setMessages(formattedMessages); // Set the formatted messages
      console.log(`Session changed. Chat history:`, formattedMessages);
  
    } else {
      console.error('Error changing session:', await response.json());
    }
  };
  
    // Handle session rename
    const handleRenameSession = async (sessionId, newName) => {
      try {
        const response = await fetch('/api/rename_session', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ session_name: sessionId, updated_name: newName })
        });
  
        if (!response.ok) {
          throw new Error(`Error renaming session: ${response.statusText}`);
        }
  
        console.log(`Session renamed from ${sessionId} to ${newName}`);
        await fetchSessions(); // Refresh the session list
  
      } catch (error) {
        console.error('Error renaming session:', error);
      }
    };

    const handleRename = async (session) => {
      // Only call the API if the name has changed
      if (updatedText !== session.text) {
        try {
          const response = await fetch('/api/rename_session', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              session_name: session.text,
              updated_name: updatedText,
            }),
          });
   
          if (response.ok) {
            const data = await response.json();  // Get the response data
            console.log(`data: ${data}`);
            console.log(`data.success: ${data.success}`);
            console.log(`data.message: ${data.message}`);
            const formattedMessages = data.display.map((item, index) => {
              const [text, process] = item;  // Destructure the tuple
              console.log(`text: ${text}`);
              console.log(`process: ${process}`);
              return {
                id: index, // Use index as a temporary unique key (for demo purposes)
                text: text,
                process: process !== null ? process : [], // If the second element is null, set it to an empty array (or other default value)
                sender: index % 2 === 0 ? 'user' : 'bot' // Example: alternate between 'user' and 'bot'
              };
            });
        
            setMessages(formattedMessages); // Set the formatted messages
            console.log(`Session changed and renamed`); // Chat history:`, formattedMessages);        
    
            console.log(`Session renamed: ${session.text} to ${updatedText}`);

          } else {
            throw new Error(`Error: ${response.statusText}`);
          }
    
          // Fetch updated sessions or update state locally
          await fetchSessions(); // or manually update the chatsessions state if necessary
    
        } catch (error) {
          console.error('Error renaming session:', error);
        }
      }
    
      // Exit the editing mode
      setEditingSessionId(null);
    };
    

    // When user clicks outside the input field (blur event)
    const handleBlur = (session) => {
      if (editingSessionId && updatedText !== session.text) {
        handleRenameSession(session.text, updatedText); // Call rename API
      }
      setEditingSessionId(null); // Exit edit mode
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
      <button
        className="session-box"
        onClick={handleNewSession}
      >
        New Session
      </button>
      {chatsessions.map((session) => (
        <div
          key={session.id}
          className="session-box"
          onClick={() => handleSessionChange(session.text)}
        >
          {editingSessionId === session.id ? (
            <input
              type="text"
              value={updatedText}
              onChange={(e) => setUpdatedText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleRename(session); // Trigger rename on Enter/Return key
                }
              }}
              onBlur={() => handleRename(session)} // Trigger rename when focus is lost
              autoFocus // Automatically focus when editing starts
            />
          ) : (
            <div
              onMouseEnter={() => {
                setEditingSessionId(session.id);
                setUpdatedText(session.text); // Set the text for editing
              }}
            >
              <Markdown>{session.text}</Markdown>
            </div>
          )}

          <button
            className="delete-btn"
            onClick={(e) => {
              e.stopPropagation();
              handleRemoveSession(session.text);
            }}
          >
            Delete
          </button>
        </div>
      ))}
    </div>

  );
}

export default ChatSessions;
