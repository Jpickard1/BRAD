import React, { useState, useEffect } from 'react';

function ThemeChangeButton() {
  const [theme, setTheme] = useState('dark'); // Default to light mode

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme); // Store theme in local storage
  };

  useEffect(() => {
    // Retrieve theme from local storage on initial render
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) {
      setTheme(storedTheme);
    }
  }, []);


  useEffect(() => {
    // Get the root element
    const root = document.querySelector(':root');
    // Change the value of the CSS variable
    if (theme == 'light') {
        root.style.setProperty('--bg-color', '#FFFFFF'); 
        root.style.setProperty('--base-color', '#F4F4F4'); 
        root.style.setProperty('--font-color', '#5D5D5D'); 
        root.style.setProperty('--bold-color', '#F9F9F9');
        root.style.setProperty('--hover-color', '#ECECEC'); 
    }
    if (theme == 'dark') {
        root.style.setProperty('--bg-color', '#212121'); 
        root.style.setProperty('--base-color', '#2f2f2f'); 
        root.style.setProperty('--font-color', '#f7f7f7'); 
        root.style.setProperty('--hover-color', '#797979'); 
        root.style.setProperty('--bold-color', '#878787');
    }

  }, [theme]);

  return (
    <div class='settings'>
      <button class='theme' onClick={toggleTheme}>
        {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
      </button>
    </div>
  );
}

export default ThemeChangeButton;