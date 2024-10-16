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
              {message.process.map(([name, payload], index) => {
                if (name === "RAG-R") {
                  return (
                    <div key={index} className="retriever">
                      {payload.map((item, i) => (
                        <div key={i}>{item}</div>
                      ))}
                    </div>
                  );
                } else if (name === "RAG-G") {
                  return (
                    <div key={index}>
                      {payload.map((paragraph, i) => (
                        <div key={i} className="rag-paragraph">{paragraph}</div>
                      ))}
                    </div>
                  );
                } else {
                  return <div key={index}>{String(payload)}</div>;
                }
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default MessageList;
