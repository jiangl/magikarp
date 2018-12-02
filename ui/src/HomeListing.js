import React from 'react';
import HomeInfo from './HomeInfo';

import './HomeListing.css';

const HomeListing = ({ highConfidenceHomes, lowConfidenceHomes }) => {
  return (
    <div className="homeListing">
      <h4>Homes</h4>
      <div className="listingHeaders">
        <span>Address</span>
        <span>Cost</span>
        <span>{'70-100'}</span>
      </div>
      <div className="listingContainer">
        {highConfidenceHomes.map(home => {
          return <HomeInfo {...home} confidenceLevel="high" />;
        })}
      </div>
      <div className="divider">
        <span>Confidence less than</span>
        <span>{'<70'}</span>
      </div>
      <div className="listingContainer">
        {highConfidenceHomes.map(home => {
          return <HomeInfo {...home} confidenceLevel="low" />;
        })}
      </div>
    </div>
  );
};

export default HomeListing;
