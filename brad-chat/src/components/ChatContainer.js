import React, { useEffect } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import "highlight.js/styles/github.css";
import hljs from "highlight.js";
import RagFileInput from './RagFileInput';

function ChatContainer({ messages, onSendMessage }) {
  // UseEffect to highlight syntax when messages change
  useEffect(() => {
    hljs.highlightAll();
  }, [messages]);

  return (
    <div className="chat-container">
      <RagFileInput />
      {/* Display the messages */}
      <MessageList messages={messages} />
      {/* Pass the onSendMessage handler to MessageInput */}
      <MessageInput onSendMessage={onSendMessage} />
    </div>
  );
}

export default ChatContainer;
