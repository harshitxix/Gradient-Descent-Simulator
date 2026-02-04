import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from GD_backend import get_default_data, train_gd_model


st.set_page_config(page_title="GD Simulator", page_icon="🧠", layout="wide")

# Custom CSS for minimal premium design
st.markdown("""
<style>
	.main {background: #0f1419; padding: 0.8rem;}
	.block-container {padding-top: 1rem; padding-bottom: 0.8rem; max-width: 100%;}
	h1 {color: #e8eaed; font-size: 1.6rem; margin-bottom: 0.2rem; font-weight: 500; letter-spacing: -0.02em;}
	h2, h3 {color: #b8bbbe; font-size: 0.95rem; margin: 0.3rem 0 0.2rem 0; font-weight: 500;}
	.stDataFrame {border-radius: 4px; border: 1px solid #2a2e33;}
	.stButton>button {
		width: 100%; background: #1a1f26; color: #e8eaed; font-weight: 500; 
		border: 1px solid #2a2e33; padding: 0.4rem 0.8rem; border-radius: 4px;
		transition: all 0.2s; font-size: 0.9rem;
	}
	.stButton>button:hover {background: #242a31; border-color: #3a3f44;}
	div[data-testid="stFileUploader"] {
		background: #161b22; padding: 0.6rem; border-radius: 4px; 
		border: 1px solid #2a2e33;
	}
	.stSlider {padding: 0.15rem 0;}
	div[data-baseweb="slider"] {background: transparent; padding: 0.3rem; border-radius: 3px;}
	.uploadedFile {background: #1a1f26; border-radius: 4px; border: 1px solid #2a2e33;}
	[data-testid="stExpander"] {
		background: #161b22; border-radius: 4px; border: 1px solid #2a2e33;
		margin-bottom: 0.5rem;
	}
	.stAlert {border-radius: 4px; padding: 0.4rem; background: #1a1f26; border: 1px solid #2a2e33;}
	div[data-testid="stMarkdownContainer"] p {font-size: 0.9rem; color: #b8bbbe;}
	[data-testid="stCaption"] {font-size: 0.85rem; color: #6e7681;}
</style>
""", unsafe_allow_html=True)


def _load_csv(uploaded_file):
	try:
		content = uploaded_file.read()
		return pd.read_csv(io.BytesIO(content))
	except Exception:
		return None


st.title("Neural Network GD Simulator")
st.caption("Upload data, tune hyperparameters, and visualize training dynamics")

left, right = st.columns([0.28, 0.72], gap="medium")

with left:
	st.subheader("Data Input")
	uploaded = st.file_uploader("", type=["csv"], help="CSV with 3 features (x1,x2,x3) + 1 label (y)", label_visibility="collapsed")

	if uploaded is None:
		X, Y = get_default_data()
		st.info("✓ Using default dataset")
		df_preview = pd.DataFrame(X, columns=["x1", "x2", "x3"])
		df_preview["y"] = Y
	else:
		df = _load_csv(uploaded)
		if df is None or df.shape[1] < 4:
			st.error("❌ Invalid CSV")
			X, Y = get_default_data()
			df_preview = pd.DataFrame(X, columns=["x1", "x2", "x3"])
			df_preview["y"] = Y
		else:
			df_preview = df.copy()
			X = df_preview.iloc[:, 0:3].values.tolist()
			Y = df_preview.iloc[:, 3].values.tolist()

	st.dataframe(df_preview, use_container_width=True, height=180)

	st.subheader("Training")
	col_a, col_b = st.columns(2)
	with col_a:
		lr = st.number_input("LR", 0.001, 0.5, 0.05, 0.001, format="%.3f")
	with col_b:
		epochs = st.number_input("Epochs", 50, 2000, 500, 50)
	
	run_training = st.button("Run Training", type="primary", use_container_width=True)

with right:
	st.subheader("Model Configuration")
	
	with st.expander("Weights", expanded=True):
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

	st.markdown("<h3 style='font-size: 0.95rem; margin-bottom: 8px; color: #b8bbbe; font-weight: 500;'>Results</h3>", unsafe_allow_html=True)




	if run_training and X is not None and Y is not None:
		with st.spinner('Training in progress...'):
			result = train_gd_model(
				X, Y, lr=lr, epochs=epochs, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6,
				w7=w7, w8=w8, bh1=bh1, bh2=bh2, bo=bo, log_every=1
			)
			error_history = result["errors"]
			output_history = result["outputs"]
			final_weights = result["weights"]
	
		st.success(f"Training Completed | Error: {error_history[-1]:.6f}")
		
		# Combined Chart with dual y-axes
		fig, ax1 = plt.subplots(figsize=(5, 4), facecolor='white')
		
		# Generate epoch numbers for x-axis (1 to epochs)
		epoch_numbers = list(range(1, epochs + 1))
		
		# Error line (left y-axis)
		color1 = '#ff6b6b'
		ax1.set_xlabel('Epoch', fontsize=8, fontweight='normal', color='#333')
		ax1.set_ylabel('Total Error', color=color1, fontsize=8, fontweight='normal')
		line1 = ax1.plot(epoch_numbers, error_history, linewidth=1.5, color=color1, label='Error', alpha=0.9)
		ax1.tick_params(axis='y', labelcolor=color1, labelsize=7, colors='#333')
		ax1.tick_params(axis='x', labelsize=7, colors='#333')
		# Set x-axis to show full range from 0 to total epochs
		ax1.set_xlim(0, epochs)
		ax1.grid(True, alpha=0.15, linestyle='-', color='#ddd')
		ax1.set_facecolor('white')
		ax1.spines['bottom'].set_color('#ddd')
		ax1.spines['top'].set_color('#ddd')
		ax1.spines['left'].set_color('#ddd')
		ax1.spines['right'].set_color('#ddd')
		
		# Output line (right y-axis)
		ax2 = ax1.twinx()
		color2 = '#4ecdc4'
		ax2.set_ylabel('Output', color=color2, fontsize=8, fontweight='normal')
		line2 = ax2.plot(epoch_numbers, output_history, linewidth=1.5, color=color2, label='Output', alpha=0.9)
		ax2.tick_params(axis='y', labelcolor=color2, labelsize=7, colors='#333')
		
		# Title and legend
		fig.suptitle('Training Convergence', fontsize=10, fontweight='normal', color='#333')
		
		# Combine legends
		lines = line1 + line2
		labels = [l.get_label() for l in lines]
		ax1.legend(lines, labels, loc='upper right', framealpha=0.95, fontsize=7, 
		          facecolor='white', edgecolor='#ddd', labelcolor='#333')
		
		fig.tight_layout()
		st.pyplot(fig)
		
		# Metrics row (compact text)
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.markdown(f"<div style='text-align:center; font-size: 9px; color:#6e7681;'>Error</div><div style='text-align:center; font-size: 11px; font-weight:500; color:#e8eaed;'>{error_history[-1]:.4f}</div>", unsafe_allow_html=True)
		with col2:
			st.markdown(f"<div style='text-align:center; font-size: 9px; color:#6e7681;'>Output</div><div style='text-align:center; font-size: 11px; font-weight:500; color:#e8eaed;'>{output_history[-1]:.4f}</div>", unsafe_allow_html=True)
		with col3:
			st.markdown(f"<div style='text-align:center; font-size: 9px; color:#6e7681;'>Epochs</div><div style='text-align:center; font-size: 11px; font-weight:500; color:#e8eaed;'>{epochs}</div>", unsafe_allow_html=True)
		with col4:
			st.markdown(f"<div style='text-align:center; font-size: 9px; color:#6e7681;'>LR</div><div style='text-align:center; font-size: 11px; font-weight:500; color:#e8eaed;'>{lr}</div>", unsafe_allow_html=True)
		
		# Display final weights in compact format
		with st.expander("Final Weights", expanded=False):
			col1, col2 = st.columns(2)
			with col1:
				st.write("<p style='font-size: 10px; margin-bottom: 4px; color: #b8bbbe;'><b>Input → Hidden</b></p>", unsafe_allow_html=True)
				weights_ih = pd.DataFrame({
					'Weight': ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'bh1', 'bh2'],
					'Value': [f"{final_weights['w1']:.4f}", f"{final_weights['w2']:.4f}", 
					         f"{final_weights['w3']:.4f}", f"{final_weights['w4']:.4f}", 
					         f"{final_weights['w5']:.4f}", f"{final_weights['w6']:.4f}",
					         f"{final_weights['bh1']:.4f}", f"{final_weights['bh2']:.4f}"]
				})
				st.dataframe(weights_ih, use_container_width=True, hide_index=True)
			
			with col2:
				st.write("<p style='font-size: 10px; margin-bottom: 4px; color: #b8bbbe;'><b>Hidden → Output</b></p>", unsafe_allow_html=True)
				weights_ho = pd.DataFrame({
					'Weight': ['w7', 'w8', 'bo'],
					'Value': [f"{final_weights['w7']:.4f}", f"{final_weights['w8']:.4f}", 
					         f"{final_weights['bo']:.4f}"]
				})
				st.dataframe(weights_ho, use_container_width=True, hide_index=True)
	
	else:
		st.info("Configure data & weights, then click training button")
