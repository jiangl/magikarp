import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import Dashboard from './Dashboard';

class App extends Component {
  render() {
    return (
      <div className="App">
        <div className="left-panel">
          <Dashboard />
        </div>
        <div className="middle-panel" />
        <div className="right-panel" />
      </div>
    );
  }
}

export default App;
