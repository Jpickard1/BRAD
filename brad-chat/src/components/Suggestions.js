// Suggestions.js
// Remove `useRef` from the import
import React, { useEffect, useState } from 'react';

function Suggestions({ onSuggestionClick }) {
  // TODO: Add improved list of suggestions
  const suggestionsList  = [
    "How can I analyze single-cell RNA data?",
    "What are some tools for gene network analysis?",
    "Can you help with protein structure prediction?",
    "How do I interpret results from a PCA plot?",
    "Chat with documents",
    "Search the archive",
    "Get Help"
  ];

  const [shuffledSuggestions, setShuffledSuggestions] = useState([]);
  const [loaded, setLoaded] = useState(false); // Track when component is fully loaded

  const colors = [
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffa500"
  ]; // Example color palette

  // Shuffle the suggestions and apply colors
  useEffect(() => {
    const shuffled = [...suggestionsList].sort(() => Math.random() - 0.5); // Randomize order
    setShuffledSuggestions(
      shuffled.map((suggestion, index) => ({
        text: suggestion,
        color: colors[index % colors.length],
      }))
    );

    // Trigger the animation after a short delay
    setTimeout(() => setLoaded(true), 1000); // Add delay to allow the DOM to render
  }, []);

  return (
    <div className={`suggestions-container ${loaded ? 'loaded' : ''}`}>
      {shuffledSuggestions.map((suggestion, index) => (
        <div
          key={index}
          className="suggestion-box"
          style={{
            '--suggestion-color': suggestion.color, // Set background color variable
            '--hover-color': "#d9e9ff", // Set hover color variable
          }}
          onClick={() => onSuggestionClick(suggestion.text)}
        >
          {suggestion.text}
        </div>
      ))}
    </div>
  );

}

export default Suggestions;
