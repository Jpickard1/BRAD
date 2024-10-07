import React, { useState } from 'react';

function MessageInput({ onSendMessage }) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="message-input">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Hey there! I am brad. Ask me to perform bioniformatics tasks"
      />
      <button type="submit">{String.fromCodePoint('0x25B2')}</button>
    </form>
  );
}

export default MessageInput;