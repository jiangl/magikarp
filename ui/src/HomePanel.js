import React from 'react';
import HouseFeature from './HouseFeature';
import FeaturePanel from './FeaturePanel';

import './HomePanel.css';

class HomePanel extends React.Component {
  render() {
    const {
      address,
      cost,
      confidence,
      featureList,
      verify,
      changeSeverity
    } = this.props;
    return (
      <div className="homePanel">
        <div className="homeDetails">
          <div>
            <h4>{address}</h4>
            <span className="cityAddr">Miami, Florida</span>
          </div>
          <div>
            <div className="detailValue">{`$${cost}k`}</div>
            <div className="detailType">damage</div>
          </div>
          <div>
            <div className="detailValue">{`${confidence}%`}</div>
            <div className="detailType">confidence</div>
          </div>
        </div>
        <img src="images/house1.png" />
        <div className="homeFeatures">
          <FeaturePanel
            featureList={featureList}
            verify={verify}
            changeSeverity={changeSeverity}
          />
        </div>
      </div>
    );
  }
}

export default HomePanel;
