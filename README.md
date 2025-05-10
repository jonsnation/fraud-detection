# Fraud Detection of Financial Transactions

### Project Members:
- **Danara Sahadeo**  
- **Jonathan Swamber**   
- **Jakob Young Cassim** 

## Problem Statement
Financial fraud is a growing global issue, causing billions of dollars in losses each year. Traditional fraud detection methods struggle to adapt to evolving fraud patterns. The goal of this project is to create a real-time fraud detection system that improves accuracy and adaptability by leveraging machine learning and explainable AI tools. The system aims to identify fraudulent transactions more effectively and provide transparency through interpretability techniques.

## Justification
Fraud not only results in significant financial losses but also undermines public trust in financial systems. Current rule-based detection systems have high false-positive rates and fail to adapt to emerging threats. A machine learning-driven approach can enhance detection by learning complex patterns from transaction data. Additionally, using explainable AI tools such as SHAP and LIME will improve model interpretability, making the detection process more reliable and trustworthy.

## Expected Outcomes
The project aims to produce a high-accuracy fraud detection model capable of analyzing transactions in real-time. A monitoring dashboard built with Streamlit or Dash will provide visual insights into transaction risks, while an API developed using Flask will allow real-world integration. Explainable AI methods will enhance the interpretability of the model's predictions. The final deliverables will include a comprehensive report detailing the project's methodology, findings, and recommendations for practical implementation.

## Instructions for Streaming
To test the streaming a docker instance must be started using the files contained in the streaming folder and the commands: docker-compose up --build to start and Ctrl + C to stop as well as docker-compose down to clean the instance. The api should be started via a python terminal with the dependencies outlined in the requirements.txt file through the command python3 fraud_detection_api.py within the api subfolder. The sample transactions will be streamed automatically and the results can be observed in the terminal running the docker instance. 
To change the model being used for evaluating the transactions the relevant lines need to be changed to the commented out lines within producer.py and spark_app.py

# Streaming Dashboard Setup Guide (Separate illustration from Kafka streaming)

This guide explains how to simulate real-time fraud detection using CSV-based streaming and visualize predictions via a Streamlit dashboard. This setup is designed to mimic real-time inference locally and does **not** use Kafka.

---

## 1. Download Required CSV Files

Download the following two files from the Google Drive folder:

* `X_subset_features.csv`
* `reduced_features.csv`

**Google Drive Link:**
[https://drive.google.com/drive/folders/1pIXpGTUdobSCH83eT-EHW6scEudkvIpu?usp=drive\_link](https://drive.google.com/drive/folders/1pIXpGTUdobSCH83eT-EHW6scEudkvIpu?usp=drive_link)

Place the files in the following structure relative to the project root:

```
fraud-detection-project/
├── data/processed/X_subset_features.csv
├── gnn/reduced_features.csv
```

---

## 2. Update File Paths in `csv_stream_producer.py`

Navigate to the file `dashboard/csv_stream_producer.py` and ensure that the file paths are set using **relative paths**:

```python
tree_df = pd.read_csv("data/processed/X_subset_features.csv")
gnn_df = pd.read_csv("gnn/reduced_features.csv")
```

These two datasets will be used to simulate streaming input for different models.

---

## 3. Run the Flask API

In a new terminal, navigate to the project root and start the API server:

```bash
python api/fraud_detection_api.py
```

This will launch the RESTful API which exposes multiple prediction endpoints.

---

## 4. Simulate Streaming Input

Once the API is running, open a second terminal and execute the stream producer script:

```bash
python dashboard/csv_stream_producer.py
```

This script streams transactions one by one from the CSVs to the API to simulate real-time behavior.

---

## 5. Launch the Streamlit Dashboard

Open a third terminal window and run:

```bash
streamlit run dashboard/dashboard.py
```

This will start the dashboard locally at:

```
http://localhost:8501
```

You can now visualize:

* Model predictions
* Anomaly flags from the autoencoder
* SHAP-based local explanations (if supported)
* Comparison across models

---

## 6. Summary of Component Roles

| Component                          | Description                                               |
| ---------------------------------- | --------------------------------------------------------- |
| `api/fraud_detection_api.py`       | Flask API serving model predictions (manual, stream, GNN) |
| `dashboard/csv_stream_producer.py` | Simulates real-time transaction stream using CSVs         |
| `dashboard/dashboard.py`           | Streamlit-based front-end dashboard for visualization     |

---

## Notes

* This setup is for demonstration only and replaces Kafka with CSV streaming.
* Ensure the models and preprocessors are trained and saved in the correct directories before running.
* Run the API **before** the stream producer.
