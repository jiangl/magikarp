import React, { Component } from 'react';
import './App.css';
import Dashboard from './Dashboard';
import HomePanel from './HomePanel';
import HomeListing from './HomeListing';
import Map from './Map';
import { Menu } from './icons/menu';
import { Search } from './icons/search';
import { User } from './icons/userCopy';

const initialFeatureList = [
  {
    src: 'txt',
    feature: 'waterLevel',
    severity: '5',
    isVerified: false
  },
  {
    src: 'img',
    feature: 'doorDmg',
    severity: '3',
    isVerified: false
  },
  {
    src: 'img',
    feature: 'windowDmg',
    severity: '7',
    isVerified: false
  },
  {
    src: '-',
    feature: 'roofDmg',
    isVerified: false
  },
  {
    src: 'txt',
    feature: 'wallDmg',
    severity: '5',
    isVerified: false
  },
  {
    src: 'txt',
    feature: 'structureDmg',
    severity: '2',
    isVerified: false
  }
];

const streetNames = [
  'Arena St',
  'Binder Pl',
  'Bungalow Dr',
  'Concord Pl',
  'Continental Way',
  'Coral Cir',
  'Dune St',
  'E Elm Ave',
  'E Holly Ave',
  'E Imperial Ave',
  'E Mariposa Ave',
  'E Oak Ave',
  'E Pine Ave',
  'Elin Pointe Dr',
  'Eucalyptus Dr',
  'Grand Ave',
  'Illinois St',
  'Kansas St',
  'Lairport St',
  'Mccarthy Ct',
  'Nevada St',
  'Penn St',
  'S Allied Way',
  'S Douglas St',
  'Utah Ave',
  'Virginia St',
  'Vista Del Mar Blvd',
  'W Acacia Ave',
  'W Grand Ave',
  'W Holly Ave',
  'W Mariposa Ave',
  'W Oak Ave',
  'W Palm Ave',
  'W Pine Ave',
  'Washington St',
  'Whiting St'
];

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

const getClaimTotal = featureList => {
  //will make api call in future
  const claimPerFeature = (accumulator, feature) => {
    const numSeverity = parseInt(feature.severity)
      ? parseInt(feature.severity)
      : 0;
    return numSeverity + accumulator;
  };
  return featureList.reduce(claimPerFeature, 0);
};

const getConfidence = (featureList, baseConfidence) => {
  //will make api call in future
  const confidencePerFeature = (accumulator, feature) => {
    const addConfidence = feature.isVerified ? 2 : 0;
    return accumulator + addConfidence;
  };
  return featureList.reduce(confidencePerFeature, baseConfidence);
};

const generateHomes = () =>
  streetNames.map(streetName => {
    const houseNumber = Math.floor(Math.random() * 999);
    const baseConfidence = Math.floor(Math.random() * 40) + 40;
    const features = initialFeatureList.map(feature => {
      return { ...feature, severity: Math.floor(Math.random() * 10) };
    });
    return {
      address: houseNumber + ' ' + streetName,
      cost: getClaimTotal(features),
      confidencePercentile: baseConfidence,
      confidenceType: baseConfidence > 70 ? 'high' : 'low',
      features: features
    };
  });

class App extends Component {
  constructor(props) {
    super(props);
    const homes = generateHomes();
    this.state = {
      featureList: homes[1].features,
      selectedHouse: homes[1],
      homes: homes
    };
    this.verifyFeature = this.verifyFeature.bind(this);
    this.changeSeverity = this.changeSeverity.bind(this);
    this.selectHome = this.selectHome.bind(this);
  }
  verifyFeature(featureName, isVerified) {
    this.setState(prevState => {
      const newFeatureList = this.state.featureList.slice(0);
      const i = newFeatureList.findIndex(
        feature => feature.feature === featureName
      );
      newFeatureList[i] = {
        ...newFeatureList[i],
        isVerified: isVerified || !newFeatureList[i].isVerified
      };
      return { featureList: newFeatureList };
    });
  }
  changeSeverity(featureName, severity) {
    this.setState(prevState => {
      const newFeatureList = prevState.featureList.slice(0);
      const i = newFeatureList.findIndex(
        feature => feature.feature === featureName
      );
      newFeatureList[i] = {
        ...newFeatureList[i],
        severity: severity,
        src: newFeatureList[i].src !== '-' ? newFeatureList[i].src : 'usr',
        isVerified: true
      };
      return { featureList: newFeatureList };
    });
  }
  selectHome(address) {
    console.log('selecting', address);
    this.setState(prevState => {
      const i = prevState.homes.findIndex(
        feature => feature.address === address
      );
      if (i)
        return {
          selectedHouse: prevState.homes[i],
          featureList: prevState.homes[i].features
        };
      return prevState;
    });
  }
  render() {
    const confidence = getConfidence(
      this.state.featureList,
      this.state.selectedHouse.confidencePercentile
    );
    const claimTotal = getClaimTotal(this.state.featureList);
    const homeDetails = this.state.selectedHouse;
    homeDetails.cost = claimTotal;
    homeDetails.confidence = confidence;

    const highConfidenceHomes = this.state.homes.filter(
      home => home.confidenceType === 'high'
    );
    const lowConfidenceHomes = this.state.homes.filter(
      home => home.confidenceType !== 'high'
    );
    console.log('highConfidenceHomes', highConfidenceHomes);
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
              highConfidenceHomes={highConfidenceHomes}
              lowConfidenceHomes={lowConfidenceHomes}
              selectHome={this.selectHome}
            />
          </div>
          <div className="right-panel">
            <HomePanel
              {...homeDetails}
              verify={this.verifyFeature}
              featureList={this.state.featureList}
              changeSeverity={this.changeSeverity}
            />
          </div>
        </div>
      </div>
    );
  }
}

export default App;
