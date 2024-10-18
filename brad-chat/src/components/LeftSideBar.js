import React, { useEffect, useState } from 'react';
import ChatSessions from './ChatSessions';
import "highlight.js/styles/github.css";
import hljs from "highlight.js";

function LeftSideBar({ setMessages }) {
  //const [messages, setMessages] = useState([]);

  useEffect(() => {
    hljs.highlightAll();
  }, []);

  return (
    <div id='leftSideBar' className="left-sidebar hidden">
      <ChatSessions setMessages={setMessages} /> {/* Pass setMessages to ChatSessions */}
    </div>
  );
}

export default LeftSideBar;