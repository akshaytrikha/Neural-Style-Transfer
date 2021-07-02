// Import dependencies
import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import contentImageSource from './images/chai.jpg';
import styleImageSource from './images/starry_night.jpg';
import "./App.css";

tf.ENV.set('WEBGL_PACK', false);  // This needs to be done otherwise things run very slow v1.0.4

export default function App() {

  var predictionModel = null;
  var transferModel = null;
  var stylizedImage = null;

  // Main function
  const runCoco = async () => {
    // 3. TODO - Load network 
    // e.g. const net = await cocossd.load();
    
    // //  Loop and detect hands
    // setInterval(() => {
    //   // detect(net);
    //   // transfer(prediction_model)
    // }, 10);
  };

  const loadModels = async () => {
    predictionModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-prediction/model.json');
    // const prediction_model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json');
    console.log("prediction model loaded");


    transferModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-transfer/model.json');
    console.log("transfer model loaded");
  }

  const predict = async () => {
    // First wait for models to load
    await loadModels();

    // Generate style representation
    await tf.nextFrame();
    let bottleneck = await tf.tidy(() => {
      const styleImage = new Image(300, 700);
      styleImage.src = styleImageSource;
      const styleImageTensor = tf.browser.fromPixels(styleImage).toFloat().div(tf.scalar(255)).expandDims();

      // return this.prediction_model.predict(tf.browser.fromPixels(this.styleImage).toFloat().div(tf.scalar(255)).expandDims());
      return predictionModel.predict(styleImageTensor);
    });

    // Use style representation to generate stylized tensor
    await tf.nextFrame();
    const stylized = await tf.tidy(() => {
      const contentImage = new Image(300,700);
      contentImage.src = contentImageSource;
      const contentImageTensor = tf.browser.fromPixels(contentImage).toFloat().div(tf.scalar(255)).expandDims();

      return transferModel.predict([contentImageTensor, bottleneck]).squeeze();
    });

    // stylizedImage = await tf.browser.toPixels(stylized);

    // console.log(stylizedImage instanceof HTMLCanvasElement);
  };

  // React hook to 
  useEffect(() => {
    tf.ready().then(() => {
      predict();
    });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Neural Style Transfer</h1>
        <div className="inline-block" style={{background: "blue", textAlignVertical: "center"}}>
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
