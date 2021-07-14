import React, { useRef, useEffect } from 'react';
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
  var screenshot = null; // TODO: use state hooks
  var styleRepresentation = null;
  var predictionModel = null;
  var transferModel = null;
  var styleImage = null;
  var styleImageSource = Guernica;

  // Fetch models from a backend
  const fetchModels = async () => {
    const t0 = performance.now();
    predictionModel = await tf.loadGraphModel(process.env.PUBLIC_URL + '/models//style-prediction/model.json');
    transferModel = await tf.loadGraphModel(process.env.PUBLIC_URL + '/models//style-transfer/model.json');
    const t1 = performance.now();
    console.log("Models loaded in " + (t1 - t0)/1000 + " seconds.");
  }

  // Initializes a style image and generates a style representation based off it
  const initStyleImage = async () => {
    styleImage = new Image(300,300);
    styleImage.addEventListener("load", () => {
      console.log("Style image loaded");  // Code once style image has been loaded
      const t0 = performance.now();
      generateStyleRepresentation();
      const t1 = performance.now();
      console.log("Generated style representation in " + (t1 - t0) + " milliseconds.");
    })
    styleImage.src = styleImageSource  // Safely set styleImage.src
  }

  // On file select (from the pop up)
  const uploadStyleImage = async event => {
    // Check if user actually selected a file to upload
    if (event.target.files[0] !== undefined) {
      // For generateStyleRepresentation()
      styleImageSource = URL.createObjectURL(event.target.files[0]);
      // For displaying uploaded image
      document.getElementById("style-image-display").src = URL.createObjectURL(event.target.files[0]);
      // Initialize uploaded style image
      await initStyleImage();
    }
  }

  // Capture a screenshot from webcam
  const capture = () => {
    screenshot = webcamRef.current.getScreenshot();
  };
  
  // Learn the style of a given image
  const generateStyleRepresentation = async () => {
    await tf.nextFrame();
    styleRepresentation = await tf.tidy(() => {
      const styleImageTensor = tf.browser.fromPixels(styleImage).toFloat().div(tf.scalar(255)).expandDims();
      return predictionModel.predict(styleImageTensor);  // For cleanliness
    });
  }

  // Generate and display stylized image
  const generateStylizedImage = async () => {
    // Use style representation to generate stylized tensor
    await tf.nextFrame();
    if (screenshot != null) {
      const contentImage = new Image(300,225);
      await (contentImage.src = screenshot);  // // wait for contentImage Image object to fully read screenshot from memory
      const stylized = await tf.tidy(() => {
        // Double check contentImage has loaded
        if (contentImage.complete && contentImage.naturalHeight !== 0) {
          const contentImageTensor = tf.browser.fromPixels(contentImage).toFloat().div(tf.scalar(255)).expandDims();
          return transferModel.predict([contentImageTensor, styleRepresentation]).squeeze();  // For cleanliness
        } else {
          return null
        }  
      });

      // if stylized === null, the canvas doesn't get updated with a stylized image
      if (stylized !== null) {
        await tf.browser.toPixels(stylized, document.getElementById('stylized-canvas'));
      }
    }
    const t1 = performance.now();
  }

  // Main function
  const predict = async () => {
    // First wait for models and style image to load
    await fetchModels();
    await initStyleImage();  // also calls generateStyleRepresentation();

    var console_i = 0;  // only output generateStylizedImage logs 10 times

    setInterval(() => {
      // wait for webcam to load on screen
      if (webcamRef != null && document.hasFocus()) {
        // Loop and take and transfer screenshots of webcam input at intervals of 700 ms
        capture();
        // wait for tf to be ready then continously generate stylized images of screenshots
        tf.ready().then(() => {
          const t0 = performance.now();
          generateStylizedImage();
          const t1 = performance.now();
          if (console_i <= 10) {
            console.log("Generated stylized image in " + (t1 - t0) + " milliseconds.");
          }
          console_i += 1;
        })
      }
    }, 400);
  };

  // React hook to run main function
  useEffect(() => {
    tf.ready().then(() => {
      predict();
    });
  });

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
              screenshotQuality={0}
              videoConstraints={{facingMode: "user"}}
              style={{
                margin: "0",
                textAlign: "center",
                zindex: 9,
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
              onChange={uploadStyleImage}
            />
          </div>
          <div style={{padding: "30px"}}>
            {/* TODO wrap in <Image> */}
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