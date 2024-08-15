
import streamlit as st
import pandas as pd
import torch
from chronos import ChronosModel  # Updated import based on the latest documentation
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
import os

# Load API key from environment variable
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    st.error("API key for Google Gemini is not set.")
    st.stop()

# Initialize Google Gemini model
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to validate the uploaded CSV file
def validate_csv(df):
    required_columns = ['Date', 'SensorReading']
    if not all(column in df.columns for column in required_columns):
        st.error("CSV must contain the following columns: 'Date' and 'SensorReading'")
        return False
    return True

# Streamlit app
def main():
    st.title("Predictive Maintenance with Chronos-T5 (Tiny)")
    st.write("Upload your CSV file to forecast future sensor readings and perform predictive maintenance.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Validate CSV structure
        if validate_csv(df):
            # Allow user to adjust prediction parameters
            st.sidebar.header("Prediction Parameters")
            prediction_length = st.sidebar.slider("Prediction Length", 1, 36, 12)  # 1 to 36 months
            torch_dtype = st.sidebar.selectbox("Torch Data Type", ["float32", "bfloat16"])
            device_map = st.sidebar.selectbox("Device Map", ["cpu", "cuda"])

            # Load Chronos model
            model = ChronosModel.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map=device_map,
                torch_dtype=torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32,
            )

            # Perform prediction
            context = torch.tensor(df['SensorReading'].values)
            forecast = model.predict(context, prediction_length)

            # Generate and display the graph
            forecast_index = range(len(df), len(df) + prediction_length)
            low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

            st.subheader("Forecasted Sensor Readings")
            plt.figure(figsize=(10, 6))
            plt.plot(df['SensorReading'], color="royalblue", label="Historical Data")
            plt.plot(forecast_index, median, color="tomato", label="Median Forecast")
            plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% Prediction Interval")
            plt.xlabel("Date")
            plt.ylabel("Sensor Reading")
            plt.title("Predictive Maintenance Forecast")
            plt.legend()
            plt.grid()
            st.pyplot(plt)

            # Analyze the graph using Google Gemini
            st.subheader("Graph Analysis")
            analysis_prompt = f"Analyze the graph of sensor readings over time and explain the trends, anomalies, and predictions."
            response = model.generate_content(analysis_prompt)
            st.write(response.text)

        else:
            st.error("Please upload a well-structured CSV file with the required columns.")

# Run the app
if __name__ == "__main__":
    main()
