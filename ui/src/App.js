import React, { Component } from 'react';
import './App.css';
import Dashboard from './Dashboard';
import HomePanel from './HomePanel';
import HomeListing from './HomeListing';
import Map from './Map';

const mockHighConfidence = Array(20).fill({
  address: '47 Tehama',
  cost: 44,
  confidencePercentile: 75
});

const mockLowConfidence = Array(20).fill({
  address: '629 Nahanton',
  cost: 11,
  confidencePercentile: 37
});

const mockHomeDetails = {
  streetAddr: '47 Tehama',
  cityAddr: 'San Francisco, CA',
  cost: 44,
  confidence: 75
};

class App extends Component {
  render() {
    return (
      <div className="App">
        <div className="left-panel">
          <Dashboard />
          <div className="mapContainer">
            <Map />
          </div>
        </div>
        <div className="middle-panel">
          <HomeListing
            highConfidenceHomes={mockHighConfidence}
            lowConfidenceHomes={mockLowConfidence}
          />
        </div>
        <div className="right-panel">
          <HomePanel {...mockHomeDetails} />
        </div>
      </div>
    );
  }
}

export default App;
