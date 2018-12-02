import React from 'react';

import './HouseFeature.css';

export default class HouseFeature extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isEditing: false,
      newSeverity: props.severity,
      isVerified: false
    };
    this.onSeverityClick = this.onSeverityClick.bind(this);
    this.onSeverityChange = this.onSeverityChange.bind(this);
    this.saveSeverityChange = this.saveSeverityChange.bind(this);
    this.verify = this.verify.bind(this);
    this.inputRef = null;
  }
  componentDidUpdate() {
    this.inputRef && this.inputRef.focus();
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
      this.props.handleSeverityChange(newSeverity);
    this.setState({ isEditing: false });
    this.verify();
  }
  verify() {
    this.setState({
      isVerified: true
    });
  }
  render() {
    const { src, feature, severity, onRowClick } = this.props;
    const hasSeverity = !isNaN(severity);
    return (
      <div className="houseFeature" onClick={onRowClick}>
        <span className="src">{src}</span>
        <span className={hasSeverity ? 'feature' : 'feature noSeverity'}>
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
          {this.state.isVerified ? <svg onClick={this.verify} /> : <svg />}
        </span>
      </div>
    );
  }
}
