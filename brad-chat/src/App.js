import React from 'react';
import ChatContainer from './components/ChatContainer';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>{String.fromCodePoint('0x1f916')} BRAD </h1>
      </header>
      <ChatContainer />
    </div>
  );
}

export default App;