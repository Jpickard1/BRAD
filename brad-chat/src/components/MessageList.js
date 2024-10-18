import React from 'react';
import "highlight.js/styles/github.css";
import Markdown from "marked-react";

function MessageList({ messages }) {

  function format_message(message){
    return message.replace(/\n/g, '  \n')
  }

  return (
    <div id="message-list" className="message-list">
      {messages.map((message) => (
        <div key={message.id} className={`message ${message.sender}`}>
          <Markdown>{format_message(message.text)}</Markdown>
        </div>
      ))}
    </div>
  );
}

export default MessageList;