import React, { useState } from 'react';
import axios from 'axios';

function RagFileInput() {
    const [files, setFiles] = useState();
    const [uploadProgress, setUploadProgress] = useState(0);

    function handleChange(event) {
        setFiles(event.target.files);
    }

  const handleSubmit = (e) => {
    e.preventDefault();
    const url = '/api/rag_upload';
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('rag_files', files[i]);
    }

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
        {/* <label id="rag-file-label" for="rag-file-input">{String.fromCodePoint('0x1f4c1')}</label> */}
        <input id="rag-file-input" type="file" name="rag_files" onChange={handleChange} multiple />
        <button id="rag-file-submit" type="submit">upload {String.fromCodePoint('0x25B2')}</button>
        <progress value={uploadProgress} max="100"></progress>
        </form>
    </div>
  );
}

export default RagFileInput;