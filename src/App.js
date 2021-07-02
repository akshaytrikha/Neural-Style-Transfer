// Import dependencies
import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import contentImageSource from './images/chai.jpg';
import styleImageSource from './images/starry_night.jpg';
import "./App.css";

tf.ENV.set('WEBGL_PACK', false);  // This needs to be done otherwise things run very slow v1.0.4

export default function App() {

  var [predictionModel, setPredictionModel] = useState(null);
  var [transferModel, setTransferModel] = useState(null);

  // Main function
  const runCoco = async () => {
    // 3. TODO - Load network 
    // e.g. const net = await cocossd.load();
    
    // //  Loop and detect hands
    // setInterval(() => {
    //   // detect(net);
    //   // transfer(prediction_model)
    // }, 10);

    console.log("rohan is fat");
  };

  const loadModels = async () => {
    predictionModel = setPredictionModel(await tf.loadGraphModel('http://127.0.0.1:8080/style-prediction/model.json'));
    // const prediction_model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json');
    console.log("prediction model loaded");

    transferModel = setTransferModel(await tf.loadGraphModel('http://127.0.0.1:8080/style-transfer/model.json'));
    console.log("transfer model loaded");
  }

  // React hook to 
  useEffect(() => {
    loadModels();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Neural Style Transfer</h1>
        <div className="inline-block" style={{background: "", textAlignVertical: "center"}}>
          <img src={contentImageSource} width="300px" style={{padding: "30px"}}></img>
          <h1 style={{display: "inline-block"}}>+</h1>
          <img src={styleImageSource} width="300px" style={{padding: "30px"}}></img>
          <h1 style={{display: "inline-block"}}>=</h1>
          <img src={contentImageSource} width="300px" style={{padding: "30px"}}></img>
        </div>
      </header>
    </div>
  );
}

// export default App;
