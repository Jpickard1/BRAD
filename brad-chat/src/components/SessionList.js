// SessionList.js
import React from 'react';
import "highlight.js/styles/github.css";
import Markdown from "marked-react";
import '../App.css';

function SessionList({ messages, onChangeSession, onRemoveSession }) {

  return (
    <div className="session-list">
      {messages.map((message) => (
        <div
          key={message.id}
          className="session-box"
          onClick={() => onChangeSession(message.text)}  // Use onChangeSession prop
        >
          <Markdown>{message.text}</Markdown>

          <button
            className="delete-btn"
            onClick={(e) => {
              e.stopPropagation();
              onRemoveSession(message.text);
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
