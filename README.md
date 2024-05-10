<h1>Handwritten Image Recognition(Devanagari)</h1>
 <h2>Introduction</h2>
    <p>This project is focused on recognizing handwritten characters from images using a Convolutional Neural Network (CNN). The implementation is based on the TensorFlow and Keras libraries to process images classified into 46 distinct classes, representing different handwritten characters.</p>
    <h2>Prerequisites</h2>
    <p>Before running the project, ensure that the following libraries are installed:</p>
    <ul>
        <li>matplotlib: For plotting images and graphs for visualization.</li>
        <li>numpy: For numerical operations on arrays.</li>
        <li>PIL: Python Imaging Library, used for opening, manipulating, and saving many different image file formats.</li>
        <li>tensorflow: An open source library for numerical computation and building neural networks.</li>
        <li>pandas: For data manipulation and analysis, particularly useful for handling structured data.</li>
        <li>time: For measuring the time intervals during processing.</li>
    </ul>
    <p>You can install these libraries using pip:</p>
    <pre><code>pip install matplotlib numpy pillow tensorflow pandas</code></pre>
    <h2>Datasets</h2>
    <p>
        Dataset Link: https://www.kaggle.com/datasets/ashokpant/devanagari-character-dataset-large/data<br>
        <strong>Training dataset:</strong> Contains 78,200 files spread across 46 classes.<br>
        <strong>Testing dataset:</strong> Contains 13,800 files spread across 46 classes.
    </p>
    <p>Additionally, there is a CSV file that includes metadata for each class with columns: Class, Label, Devanagari Label, Phonetic, and Type.</p>
    <h2>Model Architecture</h2>
    <p>The model is a sequential CNN, which is suitable for image recognition tasks. Here is a breakdown of the model's architecture:</p>
    <ul>
        <li><strong>Rescaling layer:</strong> Scales the pixel values in the image to a range of 0 to 1. This helps in faster convergence during training.</li>
        <li><strong>Convolutional Layers:</strong> The model includes multiple convolutional layers with 32 and 64 filters of size 3x3. These layers help in extracting features from the images. Each convolutional layer uses 'ReLU' activation function and 'same' padding.</li>
        <li><strong>MaxPooling Layers:</strong> These layers reduce the dimensionality of each feature, which helps in reducing the computational load and overfitting.</li>
        <li><strong>Dropout Layers:</strong> Dropout layers at a rate of 0.25 are included to prevent overfitting by randomly setting a fraction of input units to 0 during training.</li>
        <li><strong>Flatten Layer:</strong> This layer flattens the 3D feature maps to 1D before feeding them to the fully connected layers.</li>
        <li><strong>Dense Layers:</strong> After flattening the feature maps, the network uses dense layers with 256 neurons and 'ReLU' activation function, followed by a final dense layer with number of neurons equal to the number of classes with 'linear' activation to output the logits for each class.</li>
    </ul>
    <h2>Performance</h2>
    <p>The model achieves an accuracy of 0.9881 on the training set and 0.9899 on the validation set. The loss was recorded at 0.0376 for the training set and 0.0410 for the validation set. These metrics indicate a high level of accuracy and suggest that the model is effective in recognizing handwritten characters from the given datasets.</p>
    <h2>Usage</h2>
    <p>To train and evaluate the model, ensure that the datasets are correctly placed in the directory and run the script. The network parameters (like number of epochs, batch size etc.) can be adjusted depending on the computational resources and the desired accuracy.</p>\
    <h2>Conclusion</h2>
    <p>The CNN model developed for this project shows excellent performance in the task of handwritten character recognition. It can potentially be extended or adapted for similar image recognition tasks involving different sets of characters or even for other image classification problems.</p>

