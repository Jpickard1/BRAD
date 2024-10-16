import React, { useState } from 'react';
import "highlight.js/styles/github.css";
import Markdown from "marked-react";

function MessageList({ messages }) {
  const [selectedMessageId, setSelectedMessageId] = useState(null);

  const handleTextClick = (id) => {
    if (selectedMessageId === id) {
      // If the message is already selected, clicking will toggle it off
      setSelectedMessageId(null);
    } else {
      // Otherwise, set the clicked message's ID as the selected one
      setSelectedMessageId(id);
    }
  };

  return (
    <div className="message-list">
      {messages.map((message) => (
        <div key={message.id} className={`message ${message.sender}`}>
          <div onClick={() => handleTextClick(message.id)}>
            <Markdown>{message.text}</Markdown>
          </div>
          {selectedMessageId === message.id && message.process && (
            <div className="message-process">
              <Markdown>{Array.isArray(message.process) ? message.process.join(', ') : String(message.process)}</Markdown>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default MessageList;
