import React, { useState, useEffect } from 'react';

function ThemeChangeButton() {
  const [theme, setTheme] = useState('light'); // Default to light mode

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
        root.style.setProperty('--bg-color', '#d5d5d5'); 
        root.style.setProperty('--base-color', '#a5a5a5'); 
        root.style.setProperty('--font-color', '#000000'); 
        root.style.setProperty('--hover-color', '#777777'); 
    }
    if (theme == 'dark') {
        root.style.setProperty('--bg-color', '#212121'); 
        root.style.setProperty('--base-color', '#2f2f2f'); 
        root.style.setProperty('--font-color', '#f7f7f7'); 
        root.style.setProperty('--hover-color', '#797979'); 
    }

  }, [theme]);

  return (
    <button class='theme' onClick={toggleTheme}>
      {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
    </button>
  );
}

export default ThemeChangeButton;