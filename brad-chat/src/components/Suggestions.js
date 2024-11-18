// Suggestions.js
// Remove `useRef` from the import
import React, { useEffect } from 'react';

function Suggestions({ onSuggestionClick }) {
  const suggestions = [
    "How can I analyze single-cell RNA data?",
    "What are some tools for gene network analysis?",
    "Can you help with protein structure prediction?",
    "How do I interpret results from a PCA plot?",
  ];

  return (
    <div className="suggestions-container">
      {suggestions.map((suggestion, index) => (
        <div
          key={index}
          className="suggestion-box"
          onClick={() => onSuggestionClick(suggestion)}
        >
          {suggestion}
        </div>
      ))}
    </div>
  );
}

export default Suggestions;
