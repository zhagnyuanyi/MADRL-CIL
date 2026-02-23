## **Data Overview and Processing**

This section provides a comprehensive look at the dataset lifecycle, from the initial collection of signal data to the final refined outputs used for model training and evaluation.

### **Data Structure & Contents**

* **Raw Data (`data.sql`):** The primary source of raw WiFi signal strength (RSS) data, stored in SQL format for efficient querying and historical tracking.
* **Processed Data Directory (`Processed sample data/`):** Contains representative samples of the data after noise filtering and feature extraction.


### **Processing Workflow**

1. **Ingestion:** Raw RSS values are pulled from the SQL database.
2. **Refinement:** Data is passed through processing scripts to handle missing values and signal outliers.
3. **Output Generation:** The final processed outputs serve as the ground truth for training the MS-FM and MADRL modules.

