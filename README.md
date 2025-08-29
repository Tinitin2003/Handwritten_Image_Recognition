<h1>Handwritten Image Recognition (Devanagari)</h1>

<h2>Introduction</h2>
<p>
    This project is focused on recognizing handwritten characters from images using Deep Learning models. 
    It includes both Convolutional Neural Network (CNN) and Artificial Neural Network (ANN) approaches. 
    The implementation is based on the TensorFlow and Keras libraries to process images classified into 46 distinct classes, 
    representing different handwritten characters.
</p>

<h2>Prerequisites</h2>
<p>Before running the project, ensure that the following libraries are installed:</p>
<ul>
    <li><strong>matplotlib:</strong> For plotting images and graphs for visualization.</li>
    <li><strong>numpy:</strong> For numerical operations on arrays.</li>
    <li><strong>PIL:</strong> Python Imaging Library, used for opening, manipulating, and saving many different image file formats.</li>
    <li><strong>tensorflow:</strong> An open source library for numerical computation and building neural networks.</li>
    <li><strong>pandas:</strong> For data manipulation and analysis, particularly useful for handling structured data.</li>
    <li><strong>time:</strong> For measuring the time intervals during processing.</li>
</ul>
<p>You can install these libraries using pip:</p>
<pre><code>pip install matplotlib numpy pillow tensorflow pandas</code></pre>

<h2>Datasets</h2>
<p>
    Dataset Link: 
    <a href="https://www.kaggle.com/datasets/ashokpant/devanagari-character-dataset-large/data" target="_blank">
        Devanagari Character Dataset
    </a><br>
    <strong>Training dataset:</strong> Contains 78,200 files spread across 46 classes.<br>
    <strong>Testing dataset:</strong> Contains 13,800 files spread across 46 classes.
</p>
<p>Additionally, there is a CSV file that includes metadata for each class with columns: Class, Label, Devanagari Label, Phonetic, and Type.</p>

<h2>CNN Model Architecture</h2>
<p>
    The CNN model is a sequential deep learning model, suitable for image recognition tasks. 
    It includes convolutional, max pooling, dropout, and dense layers for effective feature extraction and classification.
</p>
<ul>
    <li><strong>Rescaling Layer:</strong> Normalizes pixel values between 0 and 1.</li>
    <li><strong>Convolutional Layers:</strong> Multiple layers with 32 and 64 filters of size 3x3 using 'ReLU' activation.</li>
    <li><strong>MaxPooling Layers:</strong> Reduce dimensionality and control overfitting.</li>
    <li><strong>Dropout Layers:</strong> Applied at rate of 0.25 to prevent overfitting.</li>
    <li><strong>Flatten Layer:</strong> Converts 3D feature maps to 1D vectors.</li>
    <li><strong>Dense Layers:</strong> Fully connected layers with 256 neurons (ReLU), followed by output layer with 46 neurons (linear activation).</li>
</ul>

<h2>ANN Models</h2>
<p>
    Apart from the CNN, two Artificial Neural Network (ANN) models were trained for comparison. 
    These models are simpler and faster but achieved slightly lower accuracy compared to the CNN.
</p>

<h3>ANN Model (90% Accuracy)</h3>
<pre><code>model = Sequential([
    layers.Input(shape=(32,32,1)),
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.Flatten(),
    layers.Dense(100, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])
</code></pre>
<p>
    Training Results: <br>
    Accuracy: <strong>0.9608</strong> | Loss: <strong>0.1263</strong><br>
    Validation Accuracy: <strong>0.9070</strong> | Validation Loss: <strong>0.3899</strong>
</p>

<h3>ANN Model with Dropout (91% Accuracy)</h3>
<pre><code>model = Sequential([
    layers.Input(shape=(32,32,1)),
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.Flatten(),
    layers.Dense(100, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(num_classes, activation="softmax")
])
</code></pre>
<p>
    By adding a <strong>Dropout layer</strong> with rate 0.1, the ANN model achieved a validation accuracy of <strong>91%</strong>.
</p>

<h2>Performance</h2>
<ul>
    <li><strong>CNN Model:</strong> Training Accuracy: 98.81% | Validation Accuracy: 98.99%</li>
    <li><strong>ANN Model:</strong> Validation Accuracy: 90%</li>
    <li><strong>ANN Model with Dropout:</strong> Validation Accuracy: 91%</li>
</ul>

<h2>Usage</h2>
<p>
    To train and evaluate the models, ensure that the datasets are correctly placed in the directory and run the script. 
    Network parameters (like number of epochs, batch size, etc.) can be adjusted depending on the computational resources and the desired accuracy.
</p>

<h2>Conclusion</h2>
<p>
    Both CNN and ANN models demonstrate effective performance in handwritten character recognition. 
    The CNN model achieved the highest accuracy and is better suited for complex image recognition tasks. 
    However, the ANN models, especially with dropout, provide competitive results with faster training times. 
    This flexibility makes the project adaptable for different computational environments and use cases.
</p>
