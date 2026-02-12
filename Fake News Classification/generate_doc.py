
def create_rtf_content():
    header = r"{\rtf1\ansi\ansicpg1252\deff0\nouicompat\deflang1033{\fonttbl{\f0\fnil\fcharset0 Calibri;}{\f1\fnil\fcharset0 Arial;}}"
    header += "\n" + r"{\*\generator Riched20 10.0.19041}\viewkind4\uc1"
    header += "\n" + r"\pard\sa200\sl276\slmult1\b\f0\fs32 Fake News Classification Project - Interview Guide\par"
    
    sections = [
        (r"\b0\fs28 1. Project Overview (Start to End)\par", 
         r"\b0\fs22\par \b Goal: \b0 The objective of this project is to build a Machine Learning system capable of automatically classifying news articles as 'Factual' or 'Fake' based on their textual content.\par\par" +
         r"\b Workflow:\b0\par" +
         r"1. \b Data Ingestion:\b0 We read raw data from a SQLite database (classification.db), specifically looking for tables containing text and labels.\par" +
         r"2. \b Data Cleaning & EDA:\b0 The raw data is cleaned to remove rows with missing values, normalize labels (handling 'True'/'False' strings and 0/1 integers), and handle duplicates.\par" +
         r"3. \b Model Training:\b0 We implemented a \b Naive Bayes Classifier from scratch\b0 (probabilistic model) to predict the likelihood of a news piece being fake or real based on word frequencies.\par" +
         r"4. \b Evaluation:\b0 The model is evaluated using Accuracy and Confusion Matrix metrics.\par" +
         r"5. \b Deployment (UI):\b0 A user-friendly web interface was built using \b Streamlit\b0 to allow users to interact with the model, train it on the fly, and classify new text inputs.\par" +
         r"6. \b Monitoring:\b0 Every training run and inference result is logged into a CSV file (Project_Summary.csv) to track performance and business impact over time.\par"),
         
        (r"\b0\fs28 2. How to Build and Run\par",
         r"\b0\fs22\par \b Prerequisites:\b0 Python 3.x, pandas, numpy, streamlit, sqlite3, pickle.\par\par" +
         r"\b Steps:\b0\par" +
         r"1. \b Setup Environment:\b0 Ensure all dependencies are installed.\par" +
         r"2. \b Prepare Data:\b0 Place the classification.db file in the project root.\par" +
         r"3. \b Run Application:\b0 Execute the command: streamlit run streamlit_app.py\par" +
         r"4. \b Interact:\b0 Click 'Train Model' to load and train. Click 'Load Saved Model' to use existing artifacts. Enter text and click 'Classify' to test.\par"),
         
        (r"\b0\fs28 3. Challenges Faced\par",
         r"\b0\fs22\par \b Data Quality:\b0 Mixed types in labels ('True', 'False', 1, 0) and missing text. Solved by implementing a robust cleaning function.\par" +
         r"\b Class Imbalance:\b0 Real-world data is often skewed. We monitored the Confusion Matrix to ensure we weren't just predicting the majority class.\par" +
         r"\b Text Processing:\b0 Noise in text (punctuation, case). Solved with custom Regex tokenization.\par" +
         r"\b Model Persistence:\b0 Needed to save models without retraining. Solved using Python's pickle module.\par"),
         
        (r"\b0\fs28 4. Improvements & Key Features\par",
         r"\b0\fs22\par \b Custom Implementation:\b0 Implemented Naive Bayes from scratch instead of just using scikit-learn, demonstrating deep understanding of probability theory.\par" +
         r"\b Interactive UI:\b0 Built a Streamlit dashboard for non-technical stakeholders.\par" +
         r"\b Experiment Tracking:\b0 Automated logging to CSV acts as a 'lab notebook' for reproducibility.\par" +
         r"\b Error Handling:\b0 Robust checks for missing DBs or models.\par"),
         
        (r"\b0\fs28 5. Libraries Used & Justification\par",
         r"\b0\fs22\par \b Streamlit:\b0 For rapid data app creation without HTML/CSS.\par" +
         r"\b Pandas:\b0 Industry standard for tabular data manipulation.\par" +
         r"\b NumPy:\b0 Efficient numerical operations for probability calculations.\par" +
         r"\b SQLite3:\b0 Lightweight, serverless database for standalone projects.\par" +
         r"\b Pickle:\b0 Standard object serialization for saving models.\par"),
         
        (r"\b0\fs28 6. How to Use (User Guide)\par",
         r"\b0\fs22\par 1. \b Training:\b0 Click 'Train Model' and wait for accuracy metrics.\par" +
         r"2. \b Inference:\b0 Type news text and click 'Classify' to see 'Factual' or 'Fake'.\par" +
         r"3. \b Review:\b0 Check 'Latest Metrics' for model performance details.\par")
    ]
    
    content = header
    for title, body in sections:
        content += "\n" + r"\pard\sa200\sl276\slmult1" + title + body
        
    content += "\n}"
    return content

if __name__ == "__main__":
    rtf_content = create_rtf_content()
    # Write to .doc file (Word opens RTF perfectly)
    with open("Project details.doc", "w", encoding="utf-8") as f:
        f.write(rtf_content)
    print("Project details.doc created successfully.")
