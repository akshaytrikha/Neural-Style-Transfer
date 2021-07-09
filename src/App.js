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
  var screenshot = null;
  var bottleneck = null;
  // const [screenshotSrc, setScreenshotSrc] = useState(null);

  // TODO: use state hooks
  var predictionModel = null;
  var transferModel = null;

  // Fetch models from a backend
  const fetchModels = async () => {
    var t0 = performance.now();
    predictionModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-prediction/model.json');
    transferModel = await tf.loadGraphModel('http://127.0.0.1:8080/style-transfer/model.json');
    var t1 = performance.now();
    console.log("Models loaded in " + (t1 - t0)/1000 + " seconds.");
  }

  // Upload style image
  const uploadStyleImage = () => {
    console.log("style image uploaded");
  }

  // const init = (model) => {
  //   const dummy = tf.zeros([1, 10, 10, 3], 'int32');
  //   return model.executeAsync( {[INPUT_TENSOR]: dummy}, OUTPUT_TENSOR ).then(function(result){
  //     dummy.dispose();
  //     return result;
  //   });
  // }

  const capture = async () => {
    screenshot = webcamRef.current.getScreenshot();
  };

  // Generate and display stylized image
  const displayStylizedImage = async (curBottleneck) => {
    // Use style representation to generate stylized tensor

    await tf.nextFrame();
    if (screenshot != null) {
      const contentImage = new Image(300,300);
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
        console.log("here")
        await tf.browser.toPixels(stylized, document.getElementById('stylized-canvas'));
      }
    }
  }

  // Main function
  const predict = async () => {
    // First wait for models to load
    await fetchModels();

    // Generate style representation
    var t0 = performance.now();
    await tf.nextFrame();
    bottleneck = await tf.tidy(() => {
      const styleImage = new Image(300,300);
      styleImage.src = styleImageSource;
      const styleImageTensor = tf.browser.fromPixels(styleImage).toFloat().div(tf.scalar(255)).expandDims();

      return predictionModel.predict(styleImageTensor);
    });
    var t1 = performance.now();
    console.log("Generated style representation in " + (t1 - t0)/1000 + " seconds.");

    const warmupResult = transferModel.predict([tf.zeros([1,300,300,3]), bottleneck]);
    warmupResult.dataSync(); // we don't care about the result
    warmupResult.dispose();

    setInterval(() => {
      // wait for webcam to load on screen
      if (webcamRef != null) {
        // Loop and take and transfer snapshots of webcam input at intervals of __x__ ms
        capture();
        
        tf.ready().then(() => {
          displayStylizedImage(bottleneck);
        })
      }
    }, 50);
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
            {/* <MaterialUIImage src={screenshot} style={{width: "300px"}} animationDuration={1500} cover={true}/> */}
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
          <div style={{padding: "30px", flexDirection: "column"}}>
            <MaterialUIImage src={styleImageSource} style={{width: "300px"}} animationDuration={1500} cover={true}/>
            <button onClick={uploadStyleImage}>Upload Style Image</button>
          </div>
          <div style={{padding: "30px"}}>
            {/* TODO wrap in <Image> */}
            {screenshot && (
            <img src={screenshot}/>
            )}
            <canvas id={"stylized-canvas"} width="300px" height="300px" style={{cover: "true", backgroundColor: "black"}}></canvas>
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