import React, { useState } from "react";
import "./App.css";

function App() {
  const [step, setStep] = useState(0);
  const steps = [
    "Welcome to SkullMasker",
    "Choose Image Size & Format",
    "Choose Source Folder",
    "Choose or Create Destination Folder",
    "Preview Your Mask",
    "Processing Masks",
    "Binary Masks Complete!"
  ];

  const nextStep = () => setStep((prev) => Math.min(prev + 1, steps.length - 1));
  const prevStep = () => setStep((prev) => Math.max(prev - 1, 0));

  return (
    <div className="app">
      <h1>{steps[step]}</h1>
      <div className="content">
        {step === 0 && <p>Start Masking! Click next to continue.</p>}
        {step === 1 && (
          <div>
            <label>
              Length: <input type="number" placeholder="e.g. 512" />
            </label>
            <label>
              Width: <input type="number" placeholder="e.g. 512" />
            </label>
            <label>
              Format:
              <select>
                <option value="jpg">JPG</option>
                <option value="png">PNG</option>
              </select>
            </label>
          </div>
        )}
        {step === 2 && <button>Select Source Folder</button>}
        {step === 3 && <button>Select or Create Destination Folder</button>}
        {step === 4 && (
          <div>
            <p>Preview original and masked images here.</p>
            <button>Upload and Preview</button>
          </div>
        )}
        {step === 5 && (
          <div>
            <p>Processing masks...</p>
            <p>Counting → Converting → Relabeling → Structuring</p>
            <p>Progress: X/Y images done</p>
          </div>
        )}
        {step === 6 && (
          <div>
            <p>Binary Masks Complete!</p>
            <p>Check your output folder to confirm the results.</p>
          </div>
        )}
      </div>
      <div className="navigation">
        <button onClick={prevStep} disabled={step === 0}>Back</button>
        <button onClick={nextStep} disabled={step === steps.length - 1}>Next</button>
      </div>
    </div>
  );
}

export default App;