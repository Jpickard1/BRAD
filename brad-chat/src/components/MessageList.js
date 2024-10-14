import React from 'react';
import "highlight.js/styles/github.css";
import Markdown from "marked-react";

function MessageList({ messages }) {

  return (
    <div className="message-list">
      {messages.map((message) => (
        <div key={message.id} className={`message ${message.sender}`}>
          <Markdown>{message.text}</Markdown>
        </div>
      ))}
    </div>
  );
}

export default MessageList;