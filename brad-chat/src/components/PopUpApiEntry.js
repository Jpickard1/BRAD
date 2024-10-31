import React, { useState } from 'react';

const PopUpApiEntry = ({ onClose }) => {
    const [nvidiaKey, setNvidiaKey] = useState('');

    const handleSubmit = async () => {
        try {
            const response = await fetch('/api/set_llm_api_key', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ "nvidia-api-key": nvidiaKey }) // Sending NVIDIA API key in the request body
            });

            if (response.ok) {
                const data = await response.json(); // Get the response data
                console.log(`NVIDIA API Key set successfully:`, data.message);
            } else {
                const errorData = await response.json(); // Parse error message from response
                console.error('Error setting NVIDIA API Key:', errorData.message);
                // TODO: implement warning display for unsuccessful API key submission
            }
        } catch (error) {
            console.error('Error during API Key request:', error);
            // TODO: implement error handling
        }
        onClose(); // Close the popup after submission
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h2>Enter NVIDIA API Key</h2>
                <label>
                    NVIDIA API Key:
                    <input
                        type="text"
                        value={nvidiaKey}
                        onChange={(e) => setNvidiaKey(e.target.value)}
                    />
                </label>
                <button onClick={handleSubmit}>Submit</button>
                <button onClick={onClose}>Close</button>
            </div>
        </div>
    );
};

export default PopUpApiEntry;
