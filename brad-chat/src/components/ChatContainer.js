import React, { useState } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

function ChatContainer() {
  const [messages, setMessages] = useState([]);

  const handleSendMessage = (message) => {
    setMessages([...messages, { id: Date.now(), text: message, sender: 'user' }]);
  };

  return (
    <div className="chat-container">
      <MessageList messages={messages} />
      <MessageInput onSendMessage={handleSendMessage} />
    </div>
  );
}

export default ChatContainer;