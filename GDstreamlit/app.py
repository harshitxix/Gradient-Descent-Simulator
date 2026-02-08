import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from GD_backend import get_default_data, train_gd_model


st.set_page_config(page_title="GD Simulator", page_icon="🧠", layout="wide")

# Custom CSS for enhanced premium design
st.markdown("""
<style>
	@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
	
	* {font-family: 'Inter', sans-serif;}
	.main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem;}
	.block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 100%;}
	
	h1 {
		color: #ffffff; font-size: 2rem; margin-bottom: 0.3rem; 
		font-weight: 700; letter-spacing: -0.03em;
		text-align: center;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
	}
	
	h2, h3 {
		color: #e0e7ff; font-size: 1.1rem; margin: 0.5rem 0 0.4rem 0; 
		font-weight: 600; letter-spacing: -0.01em;
	}
	
	.stDataFrame {
		border-radius: 12px; 
		border: 1px solid rgba(102, 126, 234, 0.2);
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
		background: rgba(26, 31, 58, 0.6);
	}
	
	.stButton>button {
		width: 100%; 
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: #ffffff; font-weight: 600; 
		border: none; 
		padding: 0.6rem 1rem; 
		border-radius: 10px;
		transition: all 0.3s ease; 
		font-size: 0.95rem;
		box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	
	.stButton>button:hover {
		background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
		box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
		transform: translateY(-2px);
	}
	
	div[data-testid="stFileUploader"] {
		background: rgba(26, 31, 58, 0.6); 
		padding: 1rem; 
		border-radius: 12px; 
		border: 2px dashed rgba(102, 126, 234, 0.3);
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
		transition: all 0.3s ease;
	}
	
	div[data-testid="stFileUploader"]:hover {
		border-color: rgba(102, 126, 234, 0.6);
		background: rgba(26, 31, 58, 0.8);
	}
	
	.stSlider {padding: 0.2rem 0;}
	
	div[data-baseweb="slider"] {
		background: rgba(26, 31, 58, 0.4); 
		padding: 0.5rem; 
		border-radius: 8px;
	}
	
	.uploadedFile {
		background: rgba(102, 126, 234, 0.1); 
		border-radius: 8px; 
		border: 1px solid rgba(102, 126, 234, 0.3);
	}
	
	[data-testid="stExpander"] {
		background: rgba(26, 31, 58, 0.6); 
		border-radius: 12px; 
		border: 1px solid rgba(102, 126, 234, 0.2);
		margin-bottom: 0.8rem;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
	}
	
	.stAlert {
		border-radius: 10px; 
		padding: 0.6rem; 
		background: rgba(26, 31, 58, 0.8); 
		border: 1px solid rgba(102, 126, 234, 0.3);
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
	}
	
	div[data-testid="stMarkdownContainer"] p {
		font-size: 0.95rem; 
		color: #c7d2fe;
	}
	
	[data-testid="stCaption"] {
		font-size: 0.9rem; 
		color: #a5b4fc;
		font-weight: 300;
	}
	
	/* Metric cards */
	.metric-card {
		background: rgba(26, 31, 58, 0.8);
		border-radius: 10px;
		padding: 0.8rem;
		border: 1px solid rgba(102, 126, 234, 0.2);
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
		transition: all 0.3s ease;
	}
	
	.metric-card:hover {
		transform: translateY(-3px);
		box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
		border-color: rgba(102, 126, 234, 0.4);
	}
	
	/* Input styling */
	.stNumberInput input {
		background: rgba(26, 31, 58, 0.6) !important;
		color: #e0e7ff !important;
		border: 1px solid rgba(102, 126, 234, 0.2) !important;
		border-radius: 8px !important;
	}
	
	/* Success message */
	.stSuccess {
		background: rgba(34, 197, 94, 0.1) !important;
		border: 1px solid rgba(34, 197, 94, 0.3) !important;
		color: #86efac !important;
	}
	
	/* Spinner */
	.stSpinner > div {
		border-top-color: #667eea !important;
	}
	
	/* Radio buttons */
	div[role="radiogroup"] {
		background: rgba(26, 31, 58, 0.6);
		padding: 0.6rem;
		border-radius: 10px;
		border: 1px solid rgba(102, 126, 234, 0.2);
	}
	
	div[role="radiogroup"] label {
		color: #e0e7ff !important;
		font-weight: 500;
	}
	
	.stRadio > label {
		color: #e0e7ff !important;
		font-weight: 600 !important;
		font-size: 0.95rem !important;
	}
</style>
""", unsafe_allow_html=True)


def _load_csv(uploaded_file):
	try:
		content = uploaded_file.read()
		return pd.read_csv(io.BytesIO(content))
	except Exception:
		return None


def _load_iris_data():
	"""Load Iris dataset with 3 features and 1 label"""
	iris = load_iris()
	# Use first 3 features: sepal length, sepal width, petal length
	X = iris.data[:, :3].tolist()
	Y = iris.target.tolist()
	df = pd.DataFrame(iris.data[:, :3], columns=["sepal_length", "sepal_width", "petal_length"])
	df["species"] = iris.target
	return X, Y, df


st.title("Neural Network GD Simulator")
st.caption("🚀 Upload data, tune hyperparameters, and visualize training dynamics in real-time")

left, right = st.columns([0.28, 0.72], gap="medium")

with left:
	st.subheader("📊 Data Input")
	
	# Data source selection
	data_source = st.radio(
		"Select Data Source:",
		["Default Dataset", "Iris Dataset", "Upload CSV"],
		horizontal=True,
		help="Choose your data source"
	)
	
	if data_source == "Upload CSV":
		uploaded = st.file_uploader("", type=["csv"], help="📁 CSV with 3 features (x1,x2,x3) + 1 label (y)", label_visibility="collapsed")
		
		if uploaded is None:
			X, Y = get_default_data()
			st.info("⚠️ No file uploaded. Using default dataset")
			df_preview = pd.DataFrame(X, columns=["x1", "x2", "x3"])
			df_preview["y"] = Y
		else:
			df = _load_csv(uploaded)
			if df is None or df.shape[1] < 4:
				st.error("❌ Invalid CSV. Must have at least 4 columns (3 features + 1 label)")
				X, Y = get_default_data()
				df_preview = pd.DataFrame(X, columns=["x1", "x2", "x3"])
				df_preview["y"] = Y
			else:
				st.success("✅ CSV loaded successfully!")
				df_preview = df.copy()
				X = df_preview.iloc[:, 0:3].values.tolist()
				Y = df_preview.iloc[:, 3].values.tolist()
	
	elif data_source == "Iris Dataset":
		X, Y, df_preview = _load_iris_data()
		st.success("✅ Iris dataset loaded (150 samples, 3 features)")
	
	else:  # Default Dataset
		X, Y = get_default_data()
		st.info("✅ Using default dataset")
		df_preview = pd.DataFrame(X, columns=["x1", "x2", "x3"])
		df_preview["y"] = Y

	st.subheader("⚙️ Training Configuration")
	col_a, col_b = st.columns(2)
	with col_a:
		lr = st.number_input("📈 Learning Rate", 0.001, 0.5, 0.05, 0.001, format="%.3f")
	with col_b:
		epochs = st.number_input("🔄 Epochs", 50, 2000, 500, 50)
	
	st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
	run_training = st.button("Run Training", type="primary", use_container_width=True)

with right:
	st.subheader("🎛️ Model Configuration")
	
	with st.expander("⚖️ Initial Weights & Biases", expanded=True):
		c1, c2, c3, c4 = st.columns(4)
		
		with c1:
			w1 = st.slider("w1", -1.0, 1.0, 0.1, 0.05, format="%.2f")
			w2 = st.slider("w2", -1.0, 1.0, -0.2, 0.05, format="%.2f")
			w3 = st.slider("w3", -1.0, 1.0, 0.4, 0.05, format="%.2f")
		with c2:
			w4 = st.slider("w4", -1.0, 1.0, 0.2, 0.05, format="%.2f")
			w5 = st.slider("w5", -1.0, 1.0, -0.5, 0.05, format="%.2f")
			w6 = st.slider("w6", -1.0, 1.0, 0.1, 0.05, format="%.2f")
		with c3:
			w7 = st.slider("w7", -1.0, 1.0, 0.3, 0.05, format="%.2f")
			w8 = st.slider("w8", -1.0, 1.0, -0.3, 0.05, format="%.2f")
		with c4:
			bh1 = st.slider("bh1", -1.0, 1.0, 0.1, 0.05, format="%.2f")
			bh2 = st.slider("bh2", -1.0, 1.0, -0.1, 0.05, format="%.2f")
			bo = st.slider("bo", -1.0, 1.0, 0.2, 0.05, format="%.2f")

	st.markdown("<h3 style='font-size: 1.1rem; margin: 1rem 0 0.8rem 0; color: #e0e7ff; font-weight: 600;'>📈 Training Results</h3>", unsafe_allow_html=True)




	if run_training and X is not None and Y is not None:
		with st.spinner('🔄 Training neural network...'):
			result = train_gd_model(
				X, Y, lr=lr, epochs=epochs, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6,
				w7=w7, w8=w8, bh1=bh1, bh2=bh2, bo=bo, log_every=1
			)
			error_history = result["errors"]
			output_history = result["outputs"]
			final_weights = result["weights"]
	
		st.success(f"✅ Training Completed Successfully! | Final Error: {error_history[-1]:.6f}")
		
		# Enhanced Combined Chart with dual y-axes
		fig, ax1 = plt.subplots(figsize=(5, 4), facecolor='#0a0e27')
		
		# Generate epoch numbers for x-axis (1 to epochs)
		epoch_numbers = list(range(1, epochs + 1))
		
		# Error line (left y-axis) with gradient effect
		color1 = "#f59e0b"
		ax1.set_xlabel('Epoch', fontsize=9, fontweight='600', color='#e0e7ff')
		ax1.set_ylabel('Total Error', color=color1, fontsize=9, fontweight='600')
		line1 = ax1.plot(epoch_numbers, error_history, linewidth=2.5, color=color1, label='Error', alpha=0.95, marker='o', markersize=1, markevery=max(1, epochs//20))
		ax1.tick_params(axis='y', labelcolor=color1, labelsize=8, colors=color1)
		ax1.tick_params(axis='x', labelsize=8, colors='#c7d2fe')
		# Set x-axis to show full range from 0 to total epochs
		ax1.set_xlim(0, epochs)
		ax1.grid(True, alpha=0.15, linestyle='--', color='#4c5a7d')
		ax1.set_facecolor('#1a1f3a')
		ax1.spines['bottom'].set_color('#4c5a7d')
		ax1.spines['top'].set_color('#4c5a7d')
		ax1.spines['left'].set_color(color1)
		ax1.spines['right'].set_color('#4c5a7d')
		
		# Output line (right y-axis) with gradient effect
		ax2 = ax1.twinx()
		color2 = "#3b82f6"
		ax2.set_ylabel('Output', color=color2, fontsize=9, fontweight='600')
		line2 = ax2.plot(epoch_numbers, output_history, linewidth=2.5, color=color2, label='Output', alpha=0.95, marker='s', markersize=1, markevery=max(1, epochs//20))
		ax2.tick_params(axis='y', labelcolor=color2, labelsize=8, colors=color2)
		ax2.spines['right'].set_color(color2)
		
		# Enhanced Title and legend
		fig.suptitle('📊 Training Convergence Analysis', fontsize=11, fontweight='700', color='#e0e7ff', y=0.98)
		
		# Combine legends with enhanced styling
		lines = line1 + line2
		labels = [l.get_label() for l in lines]
		ax1.legend(lines, labels, loc='upper right', framealpha=0.92, fontsize=8, 
		          facecolor='#1a1f3a', edgecolor='#667eea', labelcolor='#e0e7ff', shadow=True)
		
		fig.tight_layout()
		st.pyplot(fig)
		
		# Enhanced Metrics row with card design
		st.markdown("<div style='margin: 1rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.markdown(f"""<div class='metric-card'>
				<div style='text-align:center; font-size: 10px; color:#a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Final Error</div>
				<div style='text-align:center; font-size: 16px; font-weight:700; color:#f59e0b;'>{error_history[-1]:.4f}</div>
			</div>""", unsafe_allow_html=True)
		with col2:
			st.markdown(f"""<div class='metric-card'>
				<div style='text-align:center; font-size: 10px; color:#a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Final Output</div>
				<div style='text-align:center; font-size: 16px; font-weight:700; color:#3b82f6;'>{output_history[-1]:.4f}</div>
			</div>""", unsafe_allow_html=True)
		with col3:
			st.markdown(f"""<div class='metric-card'>
				<div style='text-align:center; font-size: 10px; color:#a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Epochs</div>
				<div style='text-align:center; font-size: 16px; font-weight:700; color:#e0e7ff;'>{epochs}</div>
			</div>""", unsafe_allow_html=True)
		with col4:
			st.markdown(f"""<div class='metric-card'>
				<div style='text-align:center; font-size: 10px; color:#a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Learning Rate</div>
				<div style='text-align:center; font-size: 16px; font-weight:700; color:#e0e7ff;'>{lr}</div>
			</div>""", unsafe_allow_html=True)
		
		# Display final weights in compact format
		st.markdown("<div style='margin: 0.8rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)
		with st.expander("🎯 Final Weights & Biases", expanded=False):
			col1, col2 = st.columns(2)
			with col1:
				st.write("<p style='font-size: 11px; margin-bottom: 6px; color: #c7d2fe; font-weight: 600;'><b>📥 Input → Hidden Layer</b></p>", unsafe_allow_html=True)
				weights_ih = pd.DataFrame({
					'Weight': ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'bh1', 'bh2'],
					'Value': [f"{final_weights['w1']:.4f}", f"{final_weights['w2']:.4f}", 
					         f"{final_weights['w3']:.4f}", f"{final_weights['w4']:.4f}", 
					         f"{final_weights['w5']:.4f}", f"{final_weights['w6']:.4f}",
					         f"{final_weights['bh1']:.4f}", f"{final_weights['bh2']:.4f}"]
				})
				st.dataframe(weights_ih, use_container_width=True, hide_index=True)
			
			with col2:
				st.write("<p style='font-size: 11px; margin-bottom: 6px; color: #c7d2fe; font-weight: 600;'><b>📤 Hidden → Output Layer</b></p>", unsafe_allow_html=True)
				weights_ho = pd.DataFrame({
					'Weight': ['w7', 'w8', 'bo'],
					'Value': [f"{final_weights['w7']:.4f}", f"{final_weights['w8']:.4f}", 
					         f"{final_weights['bo']:.4f}"]
				})
				st.dataframe(weights_ho, use_container_width=True, hide_index=True)
	
	else:
		st.info("💡 Configure your data & initial weights, then click the training button to get started!")
