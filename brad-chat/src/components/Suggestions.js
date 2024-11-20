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

  const [finalSuggestions, setFinalSuggestions] = useState([]);
  const [loaded, setLoaded] = useState(false); // Track when component is fully loaded

  const colors = [
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffa500"
  ]; // Example color palette

  // Function to shuffle an array (used for both suggestions and colors)
  const shuffleArray = (array) => {
    const arrayCopy = [...array]; // Copy the array to avoid in-place mutation
    for (let i = arrayCopy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arrayCopy[i], arrayCopy[j]] = [arrayCopy[j], arrayCopy[i]]; // Swap elements
    }
    return arrayCopy;
  };

  // Prepare the suggestions
  useEffect(() => {
    // Filter out "Get Help" from the randomization pool
    const randomPool = suggestionsList.filter(suggestion => suggestion !== "Get Help");

    // Select 3 random suggestions from the pool
    const randomSuggestions = randomPool
      .sort(() => Math.random() - 0.5) // Shuffle
      .slice(0, 3); // Pick first 3 after shuffle

    // Add "Get Help" to the final list
    const selectedSuggestions = [...randomSuggestions, "Get Help"];

    // Shuffle the colors array
    const shuffledColors = shuffleArray([...colors]);

    // Map with colors
    setFinalSuggestions(
      selectedSuggestions.map((suggestion, index) => ({
        text: suggestion,
        color: shuffledColors[index % shuffledColors.length],
      }))
    );

    // Trigger the animation after a short delay
    setTimeout(() => setLoaded(true), 1000); // Add delay to allow the DOM to render
  }, []);

  return (
    <div className={`suggestions-container ${loaded ? 'loaded' : ''}`}>
      {finalSuggestions.map((suggestion, index) => (
        <div
          key={index}
          className="suggestion-box"
          style={{
            '--suggestion-color': suggestion.color, // Set shuffled color variable
            '--hover-color': "#d9e9ff", // Set hover color variable
          }}
        >
          {/* Check if the suggestion is "Get Help" and make it a link */}
          {suggestion.text === "Get Help" ? (
            <a
              href="https://brad-bioinformatics-retrieval-augmented-data.readthedocs.io/_/downloads/en/latest/pdf/" // Replace with your desired URL
              target="_blank"
              rel="noopener noreferrer" // Ensures security when opening a new tab
              style={{
                '--suggestion-color': suggestion.color, // Set shuffled color variable
                '--hover-color': "#d9e9ff", // Set hover color variable
              }}
//              style={{ color: suggestion.color, textDecoration: 'none' }}
            >
              {suggestion.text}
            </a>
          ) : (
            <span onClick={() => onSuggestionClick(suggestion.text)}>{suggestion.text}</span>
          )}
        </div>
      ))}
    </div>
  );

}

export default Suggestions;
