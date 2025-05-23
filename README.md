# Final Year Project: Layout Retrieval System

## Overview
This is the project repository for my Final Year Project on Document Layout Detection, Recognition, and Understanding. Its main purpose is to apply retrieval techniques to existing single-page layout analysis models in an academic paper understanding and question-answering scenario. 

The source code in src/ contains the [document analysis](src/document_analysis.py) backend, [GUI interface](src/interface.py), and [evaluation](src/test.ipynb) code. The paper regarding my project can be found [here]().

### Main libraries used
[LayoutParser](https://github.com/Layout-Parser/layout-parser) is used for the layout detection backbone, and [Panel](https://panel.holoviz.org) is used to construct the GUI.

The embeddings used in this project are from [Voyage AI](https://www.voyageai.com). If you wish to execute the project, be sure to obtain an API key and include it in a .env file as described in Step 3 below. They offer a free trial for a set amount of tokens, and both API key and payment method can be revoked and removed after usage.

## Installation and Setup
**Requisites**:  
- Operating System: Windows 11+
- Code editor used: VSCode
- Python version: **3.12.7**
- Microsoft Visual C++ 14.0 or greater is required to compile Detectron2. Individual components on the Visual Studio Installer:
  - Visual Studio Build Tools 2022
  - MSVC v143 - VS 2022 C++ x64/86 build tools (latest)
  - Windows 11 SDK (10.0.22000.0) 
- At least 3 GB of disk space to install relevant packages and models
- Git, to clone the project for local use.  


Using the Windows Command Prompt (cmd) (Both Windows and VSCode terminals work fine):
1. Pull the project from the repository link with Git.

    git clone https://github.com/SigmaProtogen/FYP_Layout_Retrieval_System.git

2. Navigate to the downloaded directory with: 

    cd \path\to\dir\FYP_Layout_Retrieval_System`

3. Create a .env file using .env.example, and paste your API key in the "VOYAGE_API_KEY" field.
4. Create a virtual environment.

    python -m venv .venv

5. Activate the virtual environment.

    .venv\Scripts\activate

6. Install packages according to requirements.txt in .venv. It may take a moment to properly install all packages.

    pip install -r requirements.txt

7. Install Detectron2 for Windows.

    pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

8. Install iopath from GitHub directly (since on Windows, the dependency for Detectron2 contains a faulty line) 

    pip install git+https://github.com/facebookresearch/iopath.git

9. Once everything is installed, to run the interface. 
    python src/main.py

On first execution, both Detectron2 and CrossEncoder models will be downloaded, and may take a while depending on your Internet connection speed.

The [test.ipynb](src/test.ipynb) Jupyter Notebook contains the relevant code used to assess the framework's performance.

## Author
[Qihon Chng (Lewis)](https://linkedin.com/in/lewischng)  
Email: lewischng24@gmail.com  
GitHub: [@SigmaProtogen](https://github.com/sigmaprotogen) 

## License
This project is licensed under the [Apache License 2.0](LICENSE).