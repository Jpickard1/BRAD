import React from 'react';
import ChatContainer from './components/ChatContainer';
import LeftSideBar from './components/LeftSideBar';
import RightSideBar from './components/RightSideBar';
import './App.css';



function App() {

  return (
    <div className="App">
      <header className="App-header">
        <h2>{String.fromCodePoint('0x1f916')} BRAD </h2>
      </header>
      <div className="Main-container">
        <LeftSideBar />
        <ChatContainer />
        <RightSideBar />
      </div>
    </div>
  );

}

export default App;

