import React, { Component } from 'react';
import DashboardData from './DashboardData';
import './Dashboard.css';

export default class Dashboard extends Component {
  render() {
    return (
      <div className="dashboardContainer">
        <h4>Dashboard</h4>
        <div className="statsContainer">
          <DashboardData title="Total Claim Value">$43m</DashboardData>
          <DashboardData title="Number of Claims">445k</DashboardData>
          <DashboardData title="Claims Outstanding">
            <img src="/images/barchart.png" />
          </DashboardData>
          <DashboardData title="Claim Frequency Rate">
            <img src="/images/linegraph.png" />
          </DashboardData>
        </div>
      </div>
    );
  }
}
