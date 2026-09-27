Neuro.ai — Explainable Multimodal AI for Stroke Assessment

Neuro.ai is an explainable multimodal AI framework for AI-assisted stroke assessment that integrates brain CT analysis, real-time facial motor assessment, and clinical risk indicators into a unified workflow.

The system combines deep learning, computer vision, and clinical risk modeling to provide structured assessment results, visual explanations, case storage, automated PDF reports, and an AI-based clinical assistant.

Neuro.ai is a research and decision-support prototype and is not intended to replace professional medical diagnosis or clinical judgment.

🚀 Key Features
🧠 Brain CT Analysis
ResNet18-based deep learning model
Classifies CT scans into:
Normal
Ischemic Stroke
Hemorrhagic Stroke
93.60% test accuracy
93.61% Macro F1-score
Class probability estimation
Grad-CAM-based visual explanation
🙂 Facial Motor Assessment
Real-time webcam-based assessment
MediaPipe facial landmark detection
Guided facial movements:
Neutral face
Smile
Eyebrow movement
Eye closure
Mouth movement
Left-right facial asymmetry measurement
Prototype severity categorization:
Low asymmetry
Mild asymmetry
Moderate asymmetry
High asymmetry
🏥 Clinical Risk Analysis
Random Forest-based clinical risk model
Uses demographic, medical, and lifestyle indicators
Input factors include:
Age
Gender
Hypertension
Heart disease
Average glucose level
BMI
Smoking status
Work type
Residence type
Marital status
Produces a clinical risk probability and risk category

Test Performance:

Accuracy: 86.69%
Balanced Accuracy: 74.03%
F1-score: 30.61%
ROC-AUC: 79.77%
🔍 Explainable AI

Grad-CAM is used to visualize image regions contributing to the CT model's prediction.

🔗 Multimodal Fusion

The framework combines:

CT prediction
Facial motor assessment
Clinical risk indicators

into a structured AI-assisted assessment using a rule-based fusion layer.

📄 Automated Reporting
Stores assessment cases in SQLite
Generates structured PDF reports
Includes:
Patient information
CT prediction
Class probabilities
Facial assessment
Clinical risk
Grad-CAM analysis
Multimodal assessment
Limitations and disclaimer
💬 AI Clinical Assistant

The integrated assistant provides explanations of:

CT predictions
Facial assessment
Clinical risk
Multimodal results
Grad-CAM
Stroke types
Clinical indicators
🏗️ System Architecture
                    ┌──────────────────────┐
                    │      Patient         │
                    │     Information      │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
      │  Brain CT   │   │    Face     │   │  Clinical   │
      │   Analysis  │   │ Assessment  │   │   Indicators│
      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                 │
             ▼                 ▼                 ▼
       ResNet18 +        MediaPipe +       Random Forest
       Grad-CAM          Asymmetry         Risk Prediction
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Multimodal Fusion    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        AI Assessment     PDF Report       AI Assistant
              │                │                │
              └────────────────┴────────────────┘
                         Case Database
📊 Model Performance
Brain CT Model
Metric	Result
Model	ResNet18
Task	3-Class Classification
Classes	Normal, Ischemic, Hemorrhagic
Accuracy	93.60%
Macro F1	93.61%
Balanced Accuracy	93.60%
Clinical Risk Model
Metric	Result
Model	Random Forest
Task	Binary Clinical Risk Prediction
Accuracy	86.69%
Balanced Accuracy	74.03%
F1-score	30.61%
ROC-AUC	79.77%
Facial Motor Assessment

The live facial module uses facial landmark-based asymmetry analysis rather than relying on the trained facial classifier for the final live assessment.

Prototype asymmetry thresholds are used to categorize the observed asymmetry and are not clinically validated thresholds.

🛠️ Technology Stack

Programming

Python

Deep Learning

PyTorch
ResNet18
CNN
Transfer Learning

Computer Vision

OpenCV
MediaPipe
PIL
Grad-CAM

Machine Learning

Scikit-learn
Random Forest

Application

Streamlit

Database

SQLite

Reporting

ReportLab
Automated PDF generation
📁 Project Structure
Neuro.ai/
│
├── app.py
├── neuro_ai.py
├── case_database.py
├── clinical_assistant.py
├── stroke_report.py
├── pdf_generator.py
│
├── model/
│   └── best_model.pth
│
├── ct_module/
│   ├── stroke_model.py
│   ├── neuro_ct_predictor.py
│   ├── gradcam_utils.py
│   ├── train_model.py
│   └── split_dataset.py
│
├── facial_palsy/
│   ├── live/
│   │   ├── live_camera.py
│   │   └── models/
│   ├── src/
│   └── outputs/
│
├── clinical module/
│   ├── data/
│   ├── output/
│   │   └── clinical_random_forest.joblib
│   └── src/
│       ├── clinical_predictor.py
│       ├── train_clinical.py
│       ├── train_random_forest.py
│       └── train_xgboost.py
│
├── ui/
│   └── neuro_ui.py
│
├── static/
├── templates/
├── test images/
│
├── requirements.txt
└── README.md
⚙️ Installation
1. Clone the repository
git clone <YOUR-REPOSITORY-URL>
cd Neuro.ai
2. Create a virtual environment
python -m venv venv
3. Activate the environment

Windows:

venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
▶️ Running the Application

Launch the Streamlit interface:

streamlit run ui/neuro_ui.py

The application provides navigation for:

Home
│
├── Patient Information
├── CT Analysis
├── Facial Assessment
├── Clinical Risk
├── Multimodal Analysis
├── Case History
└── AI Assistant
🔬 Workflow
1. Enter Patient Information
            ↓
2. Upload Brain CT
            ↓
3. CT Classification
            ↓
4. Grad-CAM Explanation
            ↓
5. Perform Facial Motor Assessment
            ↓
6. Enter / Process Clinical Indicators
            ↓
7. Generate Clinical Risk Estimate
            ↓
8. Multimodal Fusion
            ↓
9. Save Case
            ↓
10. Generate PDF Report
            ↓
11. Query AI Clinical Assistant
📌 CT Dataset

The CT module uses a merged dataset containing three classes:

Class	Total
Hemorrhagic	765
Ischemic	791
Normal	765
Total	2,321

Dataset split:

Split	Images
Training	1,623
Validation	346
Testing	352

The CT model uses 224 × 224 input images and ImageNet normalization.

🧠 Explainability with Grad-CAM

Grad-CAM is applied to the final convolutional feature layer of the ResNet18 model.

The generated heatmap provides a visual representation of regions that contributed to the model's classification.

CT Image
   ↓
ResNet18
   ↓
Predicted Class
   ↓
Gradient Computation
   ↓
Grad-CAM
   ↓
Heatmap
   ↓
CT + Heatmap Overlay

Grad-CAM is intended as a model-explanation mechanism and should not be interpreted as clinically validated lesion localization.

💾 Case Management

Neuro.ai stores assessment cases using SQLite.

Each case can contain:

Patient information
CT prediction
Prediction confidence
Class probabilities
Facial assessment
Facial asymmetry
Clinical risk probability
Grad-CAM information
Lesion analysis
Multimodal assessment
Timestamp
Generated report information
📄 Generated Reports

The system can generate a PDF report containing the complete assessment.

Example:

NeuroAI_Case_001.pdf

The report includes a disclaimer that the system is intended for research and decision-support purposes.

⚠️ Limitations
Neuro.ai is a research prototype, not a clinical diagnostic system.
The CT model has been evaluated on a specific dataset and may not generalize to different hospitals, scanners, populations, or acquisition protocols.
The facial assessment uses prototype asymmetry thresholds that have not been clinically validated.
The clinical module estimates a risk indicator and should not be interpreted as an acute stroke diagnosis.
The clinical test set contains a relatively small number of positive cases, which affects the stability of positive-class metrics.
Multimodal fusion is currently rule-based, rather than a learned multimodal neural network.
Clinical validation and prospective evaluation are required before real-world medical deployment.
🔮 Future Work

Potential future improvements include:

Larger and more diverse clinical datasets
Prospective clinical validation
Improved facial motor assessment
Speech-based stroke indicators
Learned multimodal fusion
Integration with electronic health records
More advanced CT architectures
Improved uncertainty estimation
External validation across institutions
Clinician-in-the-loop evaluation
👩‍💻 Authors

Kajal Koli
Tirtha Mhabade
Prachiti Shivalkar

Department of Artificial Intelligence
Usha Mittal Institute of Technology
SNDT Women's University, Mumbai, India

📜 Disclaimer

Neuro.ai is developed for academic and research purposes. It is an AI-assisted decision-support prototype and does not provide medical diagnosis, treatment recommendations, or emergency medical advice. Results should not be used as a substitute for assessment by qualified healthcare professionals.
