<h1>Handwritten Digit Classification 🖊️🔢</h1>

<p>This project implements a deep learning model to classify handwritten digits (0–9) using TensorFlow/Keras. 
It trains a neural network on grayscale images of digits (32x32) and evaluates its performance on unseen validation data.</p>

<hr/>

<h2>🚀 Features</h2>
<p>
- Preprocessing with <b>Rescaling</b> (normalizing pixel values to [0,1]) <br/>
- <b>Data Augmentation</b> for better generalization (rotation, translation, zoom, contrast) <br/>
- <b>Dense Neural Network</b> with dropout regularization <br/>
- Training and validation accuracy visualization
</p>

<hr/>

<h2>📂 Project Structure</h2>
<pre>
├── dataset/              # Handwritten digits dataset
├── model.py              # Model architecture and training
├── train.py              # Training script
├── plots/                # Accuracy & Loss plots
└── README.md             # Project documentation
</pre>

<hr/>

<h2>🧠 Model Architecture</h2>
<p>
- Input Layer: 32x32x1 grayscale image <br/>
- Data Augmentation (rotation, zoom, translation, contrast) <br/>
- Rescaling layer (normalizes pixels 0–255 → 0–1) <br/>
- Flatten layer <br/>
- Dense layer (100 neurons, ReLU activation) <br/>
- Dropout layer (rate = 0.1) <br/>
- Output layer (Softmax activation, num_classes)
</p>

<hr/>

<h2>📊 Results</h2>
<p>
- Training Accuracy: ~95% <br/>
- Validation Accuracy: ~91% (plateau) <br/>
- Loss/Accuracy plots are saved in <code>plots/</code>
</p>

<hr/>

<h2>🔧 Installation & Usage</h2>
<pre>
# 1. Clone the repository
git clone https://github.com/yourusername/handwritten-digit-classification.git
cd handwritten-digit-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training
python train.py

# 4. Visualize accuracy & loss curves
python plot_results.py
</pre>

<hr/>

<h2>📝 Future Improvements</h2>
<p>
- Use <b>Convolutional Neural Networks (CNNs)</b> for higher accuracy <br/>
- Experiment with <b>Batch Normalization</b> <br/>
- Hyperparameter tuning (learning rate, dropout rate, hidden units) <br/>
- Deploy model with <b>Flask/Streamlit</b> for real-time prediction
</p>

<hr/>

<h2>📌 Requirements</h2>
<p>
- Python 3.8+ <br/>
- TensorFlow / Keras <br/>
- Matplotlib <br/>
- NumPy
</p>

<hr/>

<h2>🙌 Acknowledgements</h2>
<p>
- TensorFlow/Keras Documentation <br/>
- MNIST/Handwritten dataset
</p>
