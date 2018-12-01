import React from 'react';
import ConfidenceBar from './ConfidenceBar';

import './HomeInfo.css';

const HomeInfo = ({
  onClick,
  address,
  cost,
  confidencePercentile,
  confidenceLevel
}) => {
  return (
    <div className={`homeInfo ${confidenceLevel}`} onClick={onClick}>
      <span className="address">{address}</span>
      <span className="cost">{`$${cost}k`}</span>
      <ConfidenceBar confidencePercentile={confidencePercentile} />
    </div>
  );
};

export default HomeInfo;
