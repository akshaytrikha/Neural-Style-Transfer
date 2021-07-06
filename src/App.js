// Import dependencies
import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import MaterialUIImage from 'material-ui-image';
import Webcam from "react-webcam";
import contentImageSource from './images/chai.jpg';
import styleImageSource from './images/guernica.jpg';
import "./App.css";

tf.ENV.set('WEBGL_PACK', false);  // This needs to be done otherwise things run very slow v1.0.4

export default function App() {
  const webcamRef = useRef(null);
  var screenshot = useRef(null);
  const [screenshotSrc, setScreenshotSrc] = useState(null);

  // TODO: use state hooks
  var predictionModel = null;
  var transferModel = null;

  // Capture a screenshot from front-facing camera
  const capture = React.useCallback(() => {
    const screenshotSrc = webcamRef.current.getScreenshot();
    setScreenshotSrc(screenshotSrc);
    console.log("screenshot captured");
    }, 
    [webcamRef, setScreenshotSrc]
  );
  
  // Fetch models from a backend
  const fetchModels = async () => {
    predictionModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-prediction/model.json');
    console.log("prediction model loaded");

    transferModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-transfer/model.json');
    console.log("transfer model loaded");
  }

  // Main function
  const predict = async () => {
    // // First wait for models to load
    // await fetchModels();

    // // Generate style representation
    // await tf.nextFrame();
    // let bottleneck = await tf.tidy(() => {
    //   const styleImage = new Image(300,300);
    //   styleImage.src = styleImageSource;
    //   const styleImageTensor = tf.browser.fromPixels(styleImage).toFloat().div(tf.scalar(255)).expandDims();

    //   return predictionModel.predict(styleImageTensor);
    // });

    // Loop and take snapshots of webcam input at intervals of __x__ ms
    setInterval(() => {
      capture();
    }, 500);

    // // Use style representation to generate stylized tensor
    // await tf.nextFrame();
    // const stylized = await tf.tidy(() => {
    //   const contentImage = new Image(300,300);
    //   contentImage.src = contentImageSource;
    //   const contentImageTensor = tf.browser.fromPixels(contentImage).toFloat().div(tf.scalar(255)).expandDims();

    //   return transferModel.predict([contentImageTensor, bottleneck]).squeeze();
    // });

    // tf.browser.toPixels(stylized, document.getElementById('stylized-canvas'));
  };

  // React hook to run models
  useEffect(() => {
    tf.ready().then(() => {
      predict();
    });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Neural Style Transfer</h1>
        <div style={{display: "flex", flexDirection: "row"}}>
          <div style={{padding: "30px"}}>
            {/* <MaterialUIImage src={contentImageSource} style={{width: "300px"}} animationDuration={1500} cover={true}/> */}
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={{facingMode: "user"}}
              style={{
                margin: "0",
                textAlign: "center",
                zindex: 9,
                // width: 640,
                // height: 480,
                width: 300,
                height: 225
              }}
            />
            {/* <button onClick={capture}>Capture photo</button> */}
          </div>
          <div style={{padding: "30px"}}>
            <MaterialUIImage src={styleImageSource} style={{width: "300px"}} animationDuration={1500} cover={true}/>
          </div>
          <div style={{padding: "30px"}}>
            {/* TODO wrap in <Image> */}
            {screenshotSrc && (
            <img src={screenshotSrc}/>
            )}
            {/* <canvas id={"stylized-canvas"} width="300px" height="300px" style={{cover: "true", backgroundColor: "black"}}></canvas> */}
          </div>
          {/* <h1 style={{display: "inline-block"}}>+</h1> */}
          {/* <img src={styleImageSource} width="300px" height="undefined" style={{padding: "30px", objectFit: "cover"}}></img> */}
          {/* <h1 style={{display: "inline-block"}}>=</h1> */}
          {/* <img src={contentImageSource} width="300px" style={{padding: "30px"}}></img> */}
        </div>
      </header>
    </div>
  );
}
