import React from 'react';
import HouseFeature from './HouseFeature';

import './FeaturePanel.css';

class FeaturePanel extends React.Component {
  render() {
    const { featureList } = this.props;
    const emptyFeatures =
      featureList && featureList.filter(feature => !feature.isVerified);
    const fullFeatures =
      featureList && featureList.filter(feature => feature.isVerified);
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
              handleSeverityChange={this.props.changeSeverity}
              verify={this.props.verify}
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
              handleSeverityChange={this.props.changeSeverity}
              onRowClick={() => console.log('onRowClick from prop')}
              verify={this.props.verify}
            />
          ))}
        </div>
      </div>
    );
  }
}

export default FeaturePanel;
