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
        <span>Confidence</span>
      </div>
      <div className="listingContainer">
        {highConfidenceHomes.map(home => {
          return <HomeInfo {...home} confidenceLevel="high" />;
        })}
      </div>
      <div className="divider" />
      <div className="listingContainer">
        {highConfidenceHomes.map(home => {
          return <HomeInfo {...home} confidenceLevel="low" />;
        })}
      </div>
    </div>
  );
};

export default HomeListing;
