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

  function find_element(elem, search){
    let e_name = null
    if (elem != null) {
      [e_name] = elem
    }

    if (e_name === search ) {
      return true
    }
    return false
  }

  function format_message(message){
    if (message){
      return message.replace(/\n/g, '  \n')
    }
    else{
      return ''
    }
  }

  return (
    <div id="message-list" className="message-list">
      {messages.map((message) => (
        <div key={message.id} className={`message ${message.sender}`}>
          <div onClick={() => handleTextClick(message.id)}>
            <Markdown>{format_message(message.text)}</Markdown>
          </div>
          {selectedMessageId === message.id && message.process && (
            <div className="message-process">
              {(() => {
                // Check if message.process is empty
                if (message.process.length === 0) {
                  return (
                    <div className="retriever">
                      Only the LLM was used.
                    </div>
                  );
                }

                let ragR 
                let ragG 

                // Find the payloads for RAG-R and RAG-G
                if (message.process == null){
                  ragR = []
                  ragG = []
                }

                else {
                  ragR = message.process.find((elem) => find_element(elem, "RAG-R"))?.[1] || [];
                  ragR = message.process.find((elem) => find_element(elem, "RAG-G"))?.[1] || [];
                }
            
                // Iterate over both payloads and display the i-th elements together
                return ragR.map((itemR, index) => {
                  // Extract everything after the last / or \ character
                  const processedItemR = itemR.split(/[/\\]/).pop();
            
                  return (
                    <div key={index} className="combined-item">
                      <div className="retriever">{processedItemR}</div>
                      <div className="rag-paragraph">{ragG[index]}</div> {/* Show the i-th element of RAG-G */}
                    </div>
                  );
                });
              })()}
            </div>
          
              
            /*
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
            */
          )}
        </div>
      ))}
    </div>
  );
}

export default MessageList;
