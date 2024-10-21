import React, { useState, useEffect } from 'react';

function LeftSideBarButton() {
  const [leftSidebar, setLeftSidebar] = useState('hidden'); // Default to light mode

  const toggleLeftSideBar = () => {
    const newLeftSidebar = leftSidebar === 'hidden' ? 'show' : 'hidden'
    setLeftSidebar(newLeftSidebar)
    localStorage.setItem('theme', newLeftSidebar); // Store theme in local storage
  }

  useEffect(() => {
    // Retrieve theme from local storage on initial render

    const storedleftSidebar = localStorage.getItem('leftSidebar');
    if (storedleftSidebar) {
      setLeftSidebar(storedleftSidebar);
    }
  }, []);

  useEffect(() => {
    // Get the root element
    document.getElementById("leftSideBarToggle").addEventListener("click", function() {
      var div = document.getElementById("leftSideBar");
      if (leftSidebar == 'hidden') {
        div.classList.remove("hidden"); // Show the div
      } else {
        div.classList.add("hidden"); // Hide the div if clicked again
      }
    });

  }, [leftSidebar]);

  return (
      <button id='leftSideBarToggle' class='toggle-left-side-bar' onClick={toggleLeftSideBar}>
        {/* {leftSidebar === 'hidden' ? 'Show Sidebar' : 'Hide Sidebar'} */}
        {leftSidebar === 'hidden' ? String.fromCodePoint('0x2630') : String.fromCodePoint('0x271B') }
      </button>);
}


export default  LeftSideBarButton ;