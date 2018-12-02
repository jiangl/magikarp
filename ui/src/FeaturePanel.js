import React from 'react';
import HouseFeature from './HouseFeature';

import './FeaturePanel.css';

class FeaturePanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      featureList:
        props.featureList &&
        props.featureList.map(feature => {
          return { ...feature, isVerified: false };
        })
    };
    this.verifyFeature = this.verifyFeature.bind(this);
  }
  componentWillReceiveProps(nextProps) {
    const { featureList } = nextProps;
    const newFeatureList =
      featureList &&
      featureList.map(feature => {
        return { ...feature, isVerified: false };
      });
    this.setState({ featureList: newFeatureList });
  }

  verifyFeature(featureName) {
    const newFeatureList = this.state.featureList.slice(0);
    const i = newFeatureList.findIndex(
      feature => feature.feature === featureName
    );
    newFeatureList[i] = {
      ...newFeatureList[i],
      isVerified: !newFeatureList[i].isVerified
    };
  }

  render() {
    const { featureList } = this.props;
    const emptyFeatures = featureList.filter(feature =>
      isNaN(feature.severity)
    );
    const fullFeatures = featureList.filter(
      feature => !isNaN(feature.severity)
    );
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
              verify={this.verifyFeature}
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
              verify={this.verifyFeature}
            />
          ))}
        </div>
      </div>
    );
  }
}

export default FeaturePanel;
