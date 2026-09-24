# Code for: [Deep entorhinal feedback controls integration of temporally separated events]

This repository contains the custom Python code and statistical summary data used to generate the figures and analyses for the manuscript.

## 1. System Requirements
*   **Operating System:** Tested on Linux (Oracle linux 8.10).
*   **Python Version:** Python 3.10
*   **Key Dependencies:** A complete list of dependencies is provided in `environment.yml`.

## 2. Installation Guide
To reproduce this environment, you must have Miniconda or Miniforge installed. 

1. Clone this repository to your local machine.
2. Open your terminal or Conda Prompt and navigate to the repository folder.
3. Create the environment from the provided file:
   ```bash
   conda env create -f environment.yml
   ```
## 3. Data Setup
This repository contains the code and lightweight statistical summaries (.csv). 
The intermediate processed multidimensional calcium data (.nc files) are hosted separately due to size constraints.

To run the full pipeline:
1. Download the  dataset from Figshare:
2. Extract the Figshare archive.
3. Place the extracted folder named data exactly one level above the code folder. Your directory must look like this:

   project_root/
├── data/                  <-- Downloaded from Zenodo
│   ├── 09.Anatomy_behavior
│   ├── 10.Post_TFC_DCZ/
│	.
│	.
│   ├── 15.Post_TFC-0/
└── code/                  <-- This GitHub repository
    ├── fig1/
    ├── fig2/
	.
	.
    └── utils/

## 4. Instructions for Use
All analyses are separated into distinct Jupyter Notebooks corresponding to the main manuscript figures.

Launch Jupyter Notebook within the activated environment.

Navigate to a figure folder (e.g., code/fig1/) and open the .ipynb file.

Run the cell clusters sequentially, first the cell of function definitions, then the cell of plotting codes.

## 5. License
This project is covered under the MIT License.
