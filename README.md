# SF State Salary Data Dashboard

Welcome to the **SF State Salary Data Dashboard** project! 
This Streamlit application provides an interactive platform for exploring and analyzing San Francisco city employee salary data. 
It features various functionalities including dynamic visualizations, interactive chat with data, and SQL database integration.

## Demo





https://github.com/user-attachments/assets/117ccc98-3605-4529-8195-0da50e99b7c9




## Dataset

The dataset used in this project is sourced from Kaggle and can be accessed [here](https://www.kaggle.com/datasets/kaggle/sf-salaries). 
It contains information about San Francisco city employees, including their names, job titles, and compensation from 2011 to 2014.


## Features

1. **Main Metrics and Visualizations**: The dashboard displays key metrics and visualizations, such as salary distributions, top earners, and income by job title.

2. **Year Slider**: A slider in the sidebar allows users to filter visualizations by year, providing insights into different time periods.

3. **Chat with Your DataFrame**: Interact with the SF Salaries dataset using LangChain's DataFrame LLM agent. Users can ask questions about the data, and receive answers based on the dataset.

4. **File Uploader**: Upload CSV or Excel files to update the dataset. The uploaded data is processed and saved into an SQL database.

5. **Chat with Your SQL**: After uploading new data, users can query the SQL database created from the uploaded files using LangChain’s LLM SQL agent. The section allows users to interact with the database through natural language queries.


## Installation

Follow these instructions to set up and run this project on your local machine.


 #### Clone the Repository:

   ```bash
   git clone https://github.com/your-username/retail-sales-dashboard.git
   ```
#### Setup

1. **Navigate to the  directory**:
 ```bash
   cd retail-sales-dashboard

   ```
2. **Create a virtual environment**:
 ```bash
python -m venv venv

 ```


3. **Activate the virtual environment**:
  
    On Windows:

```bash

venv\Scripts\activate
```
   On macOS and Linux:

 ```bash

source venv/bin/activate
```
4. **Install the dependencies**:

```bash

pip install -r requirements.txt
```



 5. **Set up Google API Key:**
   - Obtain a OPENAI API key and set it in the `.env` file.

   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```


6. **Run the streamlit app**:
   
In Terminal :


```bash

streamlit run app.py
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
Contact

For any questions or feedback, please reach out to MohamedenZeyad@outlook.com
