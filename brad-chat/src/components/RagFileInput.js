import React, { useState } from 'react';
import axios from 'axios';

function RagFileInput() {
    const [file, setFile] = useState();
    const [uploadProgress, setUploadProgress] = useState(0);

    function handleChange(event) {
        setFile(event.target.files[0]);
    }

  const handleSubmit = (e) => {
    e.preventDefault();
    const url = '/rag_upload';
    const formData = new FormData();
    formData.append('file', file);

    const config = {
      headers: {
        'content-type': 'multipart/form-data',
      },
      onUploadProgress: function(progressEvent) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
      }
    };

    axios.post(url, formData, config)
      .then((response) => {
        console.log(response.data);
      })
      .catch((error) => {
        console.error("Error uploading file: ", error);
      });
  };

  return (
    <div className='rag-file-upload'>
        <form onSubmit={handleSubmit} className="rag-file-input">
        <input type="file" onChange={handleChange} />
        <button type="submit">Upload File</button>
        <progress value={uploadProgress} max="100"></progress>
        </form>
    </div>
  );
}

export default RagFileInput;