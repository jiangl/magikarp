import React from 'react';
import HouseFeature from './HouseFeature';

import './HomePanel.css';

const HomePanel = ({ streetAddr, cityAddr, cost, confidence, features }) => {
  return (
    <div className="homePanel">
      <div className="homeDetails">
        <div>
          <h4>{streetAddr}</h4>
          <span className="cityAddr">{cityAddr}</span>
        </div>
        <div>
          <div className="detailValue">{`$${cost}k`}</div>
          <div className="detailType">damage</div>
        </div>
        <div>
          <div className="detailValue">{`${confidence}%`}</div>
          <div className="detailType">features</div>
        </div>
      </div>
      <img src="images/house1.png" />
      <div className="homeFeatures">
        <HouseFeature
          handleSeverityChange={() => console.log('handleSeverityChange')}
          src="txt"
          feature="waterLevel"
          severity={7}
          onRowClick={() => console.log('handleRowClick')}
        />
      </div>
    </div>
  );
};

export default HomePanel;
