import React, { Component } from 'react';
import './App.css';
import Dashboard from './Dashboard';
import HomePanel from './HomePanel';
import HomeListing from './HomeListing';
import Map from './Map';
import { Menu } from './icons/menu';
import { Search } from './icons/search';
import { User } from './icons/userCopy';

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
      <div className="content">
        <div className="topBar">
          <div className="left-panel">
            <Menu />
            <h4 className="title">Disaster Claims</h4>
          </div>
          <div className="middle-panel">
            <Search />
            <span className="search">Type to search...</span>
          </div>
          <div className="right-panel">
            <div className="userInfo">
              <User />
              <span className="userName">Magi Karp</span>
            </div>
          </div>
        </div>
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
      </div>
    );
  }
}

export default App;
