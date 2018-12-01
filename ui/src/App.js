import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import Dashboard from './Dashboard';
import HomeInfo from './HomeInfo';
import HomeListing from './HomeListing';

const mockHighConfidence = Array(20).fill({
  address: '47 Tehama',
  cost: 44,
  confidencePercentile: 75
});

const mockLowConfidence = Array(20).fill({
  address: '629 Sterling',
  cost: 11,
  confidencePercentile: 37
});

class App extends Component {
  render() {
    return (
      <div className="App">
        <div className="left-panel">
          <Dashboard />
        </div>
        <div className="middle-panel">
          <HomeListing
            highConfidenceHomes={mockHighConfidence}
            lowConfidenceHomes={mockLowConfidence}
          />
        </div>
        <div className="right-panel" />
      </div>
    );
  }
}

export default App;
