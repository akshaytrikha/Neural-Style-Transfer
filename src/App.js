// Import dependencies
import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import MaterialUIImage from 'material-ui-image';
import Webcam from "react-webcam";
import Chai from './images/chai.jpg';
import Guernica from './images/guernica.jpg';
import StarryNight from './images/starry_night.jpg';
import "./App.css";

tf.ENV.set('WEBGL_PACK', false);  // This needs to be done otherwise things run very slow v1.0.4

export default function App() {
  const webcamRef = useRef(null);
  var screenshot = null;
  var bottleneck = null;

  // TODO: use state hooks
  var predictionModel = null;
  var transferModel = null;
  var styleImageSource = Guernica;

  // Fetch models from a backend
  const fetchModels = async () => {
    var t0 = performance.now();
    predictionModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-prediction/model.json');
    transferModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-transfer/model.json');
    var t1 = performance.now();
    console.log("Models loaded in " + (t1 - t0)/1000 + " seconds.");
  }

  // On file select (from the pop up)
  const uploadStyleImage = event => {
    // Check if user actually selected a file, TODO: does this actually work?
    if (event.target.files[0] !== undefined) {
      // styleImageSource = URL.createObjectURL(event.target.files[0]);
      // styleImageSource = Chai;
      document.getElementById("style-image-display").src = URL.createObjectURL(event.target.files[0]);;
      // generateStyleRepresentation(URL.createObjectURL(event.target.files[0]));
      generateStyleRepresentation(Guernica);
    } else {
      console.log("uploaded file was undefined")
    }
  }

  const capture = async () => {
    screenshot = webcamRef.current.getScreenshot();
  };
  
  // Learn the style of a given image
  const generateStyleRepresentation = async () => {
    console.log("document is active element", document.activeElement);
    await tf.nextFrame();
    bottleneck = await tf.tidy(() => {
      const styleImage = new Image(300,300);
      styleImage.src = styleImageSource;
      const styleImageTensor = tf.browser.fromPixels(styleImage).toFloat().div(tf.scalar(255)).expandDims();

      return predictionModel.predict(styleImageTensor);
    });

    // const warmupResult = transferModel.predict([tf.zeros([1,300,300,3]), bottleneck]);
    // warmupResult.dataSync(); // we don't care about the result
    // warmupResult.dispose();
  }

  // Generate and display stylized image
  const generateStylizedImage = async (curBottleneck) => {
    var t0 = performance.now();
    // Use style representation to generate stylized tensor
    await tf.nextFrame();
    if (screenshot != null) {
      const contentImage = new Image(300,225);
      await (contentImage.src = screenshot);
      const stylized = await tf.tidy(() => {
        // wait for contentImage Image object to fully read screenshot from memory
        if (contentImage.complete && contentImage.naturalHeight !== 0) {
          const contentImageTensor = tf.browser.fromPixels(contentImage).toFloat().div(tf.scalar(255)).expandDims();
          return transferModel.predict([contentImageTensor, curBottleneck]).squeeze();
        } else {
          return null
        }  
      });

      if (stylized !== null) {
        await tf.browser.toPixels(stylized, document.getElementById('stylized-canvas'));
      }
    }
    var t1 = performance.now();
    console.log("Generated stylized image in " + (t1 - t0)/1000 + " seconds.");
  }

  // Main function
  const predict = async () => {
    // First wait for models to load
    await fetchModels();

    // Generate style representation
    var t0 = performance.now();
    bottleneck = generateStyleRepresentation();
    var t1 = performance.now();
    console.log("Generated style representation in " + (t1 - t0)/1000 + " seconds.");

    setInterval(() => {
      console.log(document.hasFocus())
      // wait for webcam to load on screen
      if (webcamRef != null && document.hasFocus()) {
        // Loop and take and transfer snapshots of webcam input at intervals of __x__ ms
        capture();
        
        tf.ready().then(() => {
          generateStylizedImage(bottleneck);
        })
      }
    }, 1500);
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
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpg"
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
          </div>
          <div style={{padding: "30px", textAlign: "center", flexDirection: "column"}}>
            <MaterialUIImage id="style-image-display" src={styleImageSource} style={{width: "300px"}} animationDuration={1500} cover={true}/>
            <input
              id="upload-file-input"
              type="file"
              accept="image/*"
              // onClick={pauseModel}
              onChange={uploadStyleImage}
            />
          </div>
          <div style={{padding: "30px"}}>
            {/* TODO wrap in <Image> */}
            {screenshot && (
              <img src={screenshot}/>
            )}
            <canvas id={"stylized-canvas"} width="300px" height="225px" style={{cover: "true", backgroundColor: "black"}}></canvas>
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