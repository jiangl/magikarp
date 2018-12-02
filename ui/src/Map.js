import React, { Component } from 'react';
import ReactMapGL from 'react-map-gl';
import { fullStyle } from './mapboxStyle';

class Map extends Component {
  constructor(props) {
    super();
    this.state = {
      viewport: {
        width: '100%',
        height: '100%',
        latitude: 37.7577,
        longitude: -122.4376,
        zoom: 8
      },
      bounds: null
    };
    this.onViewportChange = this.onViewportChange.bind(this);
    this.mapRef = null;
  }

  onViewportChange(viewport) {
    if (this.mapRef) {
      const mapGL = this.mapRef.getMap();
      const bounds = mapGL.getBounds();
      this.setState({ bounds });
    }
    this.setState({ viewport });
  }

  componentDidMount() {
    if (this.mapRef) {
      const mapGL = this.mapRef.getMap();
      const bounds = mapGL.getBounds();
      this.setState({ bounds });
    }
  }

  render() {
    return (
      <ReactMapGL
        {...this.state.viewport}
        mapboxApiAccessToken={
          'pk.eyJ1IjoiYms0NDEyIiwiYSI6ImNqZzAxa243ZTI2cXcyd24xd2xuZHNremcifQ.ttQhWPfF5YTeN5CvHQH1xQ'
        }
        onViewportChange={this.onViewportChange}
        style={{ width: '100%', height: '400px' }}
        mapStyle={fullStyle}
        reuseMaps={true}
        preserveDrawingBuffer={true}
        preventStyleDiffing={true}
        ref={ref => (this.mapRef = ref)}
      />
    );
  }
}
export default Map;
