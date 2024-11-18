import React, { useCallback, useEffect } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import Suggestions from './Suggestions';
import "highlight.js/styles/github.css";
import hljs from "highlight.js";
// Import useCallback if needed
// import React, { useCallback, useEffect } from 'react';

function ChatContainer({ messages, onSendMessage }) {

  useEffect(() => {
    hljs.highlightAll();

    const messageList = document.getElementById("message-list");
    if (messageList) {
      console.log("yes message list exists");
      messageList.scrollTop = messageList.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="chat-container">
      {/* Display suggestions if no messages, otherwise display the messages */}
      {messages.length === 0 ? (
        <Suggestions onSuggestionClick={onSendMessage} />
      ) : (
        <MessageList messages={messages} />
      )}
      {/* Message input is displayed regardless of messages */}
      <MessageInput onSendMessage={onSendMessage} />
    </div>
  );
}

export default ChatContainer;
