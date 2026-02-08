# Neural Network Gradient Descent Simulator

A modern, interactive web application built with Streamlit for visualizing and understanding gradient descent in neural networks. Train a simple neural network with customizable weights and hyperparameters while watching the training dynamics in real-time.

## Features

✨ **Modern UI**
- Sleek gradient design with purple-blue theme
- Interactive card-based metrics with hover effects
- Responsive layout with real-time visualizations

📊 **Flexible Data Input**
- **Default Dataset**: Pre-loaded sample data for quick testing
- **Iris Dataset**: Classic ML dataset (150 samples, 3 features)
- **Custom CSV Upload**: Upload your own data (3 features + 1 label)

⚙️ **Customizable Training**
- Adjustable learning rate (0.001 - 0.5)
- Configurable epochs (50 - 2000)
- Fine-tune all network weights and biases using sliders

📈 **Real-Time Visualization**
- Dual-axis training convergence chart
- Error and output tracking across epochs
- Final metrics display with detailed weight analysis

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/harshitxix/Gradient-Descent-Simulator.git
cd Gradient-Descent-Simulator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. **Start the application**
```bash
streamlit run GDstreamlit/app.py
```

2. **Select Your Data Source**
   - Choose from Default Dataset, Iris Dataset, or Upload CSV
   - Preview your data in the interactive table

3. **Configure Training Parameters**
   - Set learning rate and number of epochs
   - Adjust initial weights and biases using sliders

4. **Run Training**
   - Click "Run Training" button
   - Watch the training convergence in real-time
   - Analyze final metrics and trained weights

## Data Format

If uploading a CSV file, ensure it follows this format:
- **First 3 columns**: Features (x1, x2, x3)
- **4th column**: Label (y)
- No header row required

Example:
```
0.5, 0.3, 0.8, 1
0.2, 0.6, 0.4, 0
...
```

## Network Architecture

- **Input Layer**: 3 nodes (3 features)
- **Hidden Layer**: 2 nodes with configurable weights (w1-w6) and biases (bh1, bh2)
- **Output Layer**: 1 node with configurable weights (w7, w8) and bias (bo)

## Requirements

- Python 3.7+
- streamlit >= 1.28.0
- matplotlib >= 3.7.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0

## Project Structure

```
.
├── GDstreamlit/
│   ├── app.py              # Main Streamlit application
│   ├── GD_backend.py       # Gradient descent implementation
│   └── sample_data.csv     # Sample dataset
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Contributing

Feel free to open issues or submit pull requests for improvements!

## License

This project is open source and available for educational purposes.
