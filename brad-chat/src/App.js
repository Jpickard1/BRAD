import React from 'react';
import ChatContainer from './components/ChatContainer';
import LeftSideBar from './components/LeftSideBar';
import './App.css';



function App() {

  return (
    <div className="App">
      <header className="App-header">
        <h2>{String.fromCodePoint('0x1f916')} BRAD </h2>
      </header>
      <LeftSideBar />
      <ChatContainer />
    </div>
  );
}

export default App;

