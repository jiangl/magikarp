import React from 'react';

import './HouseFeature.css';

export default class HouseFeature extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isEditing: false,
      newSeverity: props.severity
    };
    this.onSeverityClick = this.onSeverityClick.bind(this);
    this.onSeverityChange = this.onSeverityChange.bind(this);
    this.saveSeverityChange = this.saveSeverityChange.bind(this);
    this.inputRef = null;
  }
  componentDidUpdate() {
    this.inputRef && this.inputRef.focus();
  }
  componentWillReceiveProps(nextProps) {
    this.setState({ newSeverity: nextProps.severity });
  }
  onSeverityClick() {
    this.setState({ isEditing: true });
  }
  onSeverityChange(e) {
    this.setState({ newSeverity: e.target.value });
  }
  saveSeverityChange() {
    const { newSeverity } = this.state;
    newSeverity >= 0 &&
      this.state.newSeverity !== this.props.severity &&
      this.props.handleSeverityChange(this.props.feature, newSeverity);
    this.setState({ isEditing: false });
  }
  render() {
    const { src, feature, severity, onRowClick } = this.props;
    const hasSeverity = !isNaN(severity);
    return (
      <div className="houseFeature" onClick={onRowClick}>
        <span className="src">{src}</span>
        <span
          className={
            this.props.isVerified ? 'feature' : 'feature noVerification'
          }
        >
          {feature}
        </span>
        {!this.state.isEditing ? (
          <span className="severity" onClick={this.onSeverityClick}>
            {hasSeverity ? severity : '-'}
          </span>
        ) : (
          <span className="severity">
            <input
              className="severityEdit"
              value={this.state.newSeverity}
              onChange={this.onSeverityChange}
              onBlur={this.saveSeverityChange}
              size={2}
              ref={ref => (this.inputRef = ref)}
            />
          </span>
        )}
        <span className="verified">
          {this.props.isVerified ? (
            <img
              onClick={() => this.props.verify(feature)}
              src="/images/checkSelected.png"
            />
          ) : (
            <img
              onClick={() => this.props.verify(feature)}
              src="/images/checkUnselected.png"
            />
          )}
        </span>
      </div>
    );
  }
}
