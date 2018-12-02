import React from 'react';
import HouseFeature from './HouseFeature';

import './FeaturePanel.css';

const FeaturePanel = ({ featureList }) => {
  const emptyFeatures = featureList.filter(feature => isNaN(feature.severity));
  const fullFeatures = featureList.filter(feature => !isNaN(feature.severity));
  return (
    <div className="featurePanel">
      <div className="featurePanelTitle">
        <h4 className="claim">Claim</h4>
        <span className="notes">Notes(11)</span>
        <span className="addNote">+</span>
      </div>
      <div className="featureHeaders">
        <span className="src">Src</span>
        <span className="feature">Feature</span>
        <span className="severity">Severity</span>
        <span className="verified">Verified</span>
      </div>
      <div className="featureList">
        {emptyFeatures.map(feature => (
          <HouseFeature
            {...feature}
            handleSeverityChange={() =>
              console.log('handleSeverityChange from prop')
            }
          />
        ))}
      </div>
      {fullFeatures.length > 0 && emptyFeatures.length > 0 && (
        <div className="featureDivider" />
      )}
      <div className="featureList">
        {fullFeatures.map(feature => (
          <HouseFeature
            {...feature}
            handleSeverityChange={() =>
              console.log('handleSeverityChange from prop')
            }
            onRowClick={() => console.log('onRowClick from prop')}
          />
        ))}
      </div>
    </div>
  );
};

export default FeaturePanel;
