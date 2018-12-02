import React from 'react';
import HouseFeature from './HouseFeature';
import FeaturePanel from './FeaturePanel';

import './HomePanel.css';

class HomePanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isEditing: false,
      newCost: props.cost
    };
    this.onCostClick = this.onCostClick.bind(this);
    this.onCostChange = this.onCostChange.bind(this);
    this.saveCostChange = this.saveCostChange.bind(this);
  }
  componentDidUpdate() {
    this.inputRef && this.inputRef.focus();
  }
  componentWillReceiveProps(nextProps) {
    this.setState({ newCost: nextProps.cost });
  }
  onCostClick() {
    this.setState({ isEditing: true });
  }
  onCostChange(e) {
    this.setState({ newCost: e.target.value });
  }
  saveCostChange() {
    const { newCost } = this.state;
    newCost >= 0 &&
      this.state.newCost !== this.props.cost &&
      this.props.handleCostChange(newCost);
    this.setState({ isEditing: false });
  }
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
          {this.state.isEditing ? (
            <div className="cost">
              <span>$</span>
              <input
                className="costEdit"
                value={this.state.newCost}
                onChange={this.onCostChange}
                onBlur={this.saveCostChange}
                size={2}
                ref={ref => (this.inputRef = ref)}
              />
              <span>k</span>
            </div>
          ) : (
            <div onClick={() => this.setState({ isEditing: true })}>
              <div className="detailValue">{`$${cost}k`}</div>
              <div className="detailType">damage</div>
            </div>
          )}
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
