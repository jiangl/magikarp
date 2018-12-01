import React from 'react';

const ConfidenceBar = ({ confidencePercentile }) => {
  return (
    <div className="confidenceBar">
      <div className="barFill" style={{ width: `${confidencePercentile}%` }} />
    </div>
  );
};

export default ConfidenceBar;
