# Near Real Time Arbritrary Style Transfer Filter


<img src="https://media.giphy.com/media/ii4Wnf1nXK7vkWjkW7/giphy.gif" width="880" />


This project uses two pretrained Tensorflow.js neural networks, sourced from Reiichiro Nakano ['arbitrary-image-stylization-tfjs'](https://github.com/reiinakano/arbitrary-image-stylization-tfjs) project. The first network is used for _style prediction_, or 'learning' the style of a given image and generating a style representation. The second network is used for _style transfer_, or using the style representation to generate a stylized output image. For a more detailed breakdown of how the networks work, check out Reiichiro's [blog post](https://magenta.tensorflow.org/blog/2018/12/20/style-transfer-js/). Also, thanks to Nicholas Renotte for his [ReactComputerVisonTemplate project](https://github.com/nicknochnack/ReactComputerVisionTemplate).

A very fun fact about TensorFlow.js your browser is locally doing the machine learning, and all your data is kept on your device.