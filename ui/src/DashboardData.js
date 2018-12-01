import React, { Component } from 'react';
import './DashboardData.css';

const DashboardData = ({ title, children }) => {
  return (
    <div className="dashboardDataContainer">
      <h6>{title}</h6>
      <div className="statData">{children}</div>
    </div>
  );
};

export default DashboardData;
