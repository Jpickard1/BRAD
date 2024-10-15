import React from 'react';
import "highlight.js/styles/github.css";
import Markdown from "marked-react";
import '../App.css';  // Import the CSS file for custom styles

function SessionList({ messages, onRemoveSession }) {

  // Function to handle session change
  const handleSessionChange = async (sessionId) => {
    try {
      const response = await fetch('/api/change_session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: sessionId })
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      console.log(`Session changed to: ${sessionId}`);
    } catch (error) {
      console.error('Error changing session:', error);
    }
  };

  return (
    <div className="session-list">
      {messages.map((message) => (
        <div
          key={message.id}
          className="session-box"
          onClick={() => handleSessionChange(message.text)}  // Session click
        >
          <Markdown>{message.text}</Markdown>

          {/* Delete button that appears on hover */}
          <button
            className="delete-btn"
            onClick={(e) => {
              e.stopPropagation();  // Prevent triggering session change when delete is clicked
              onRemoveSession(message.text);  // Call remove session from parent
            }}
          >
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}

export default SessionList;
