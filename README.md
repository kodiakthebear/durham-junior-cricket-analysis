# Durham Junior Cricket Analysis

A comprehensive data analysis pipeline for evaluating and validating U15 cricket player selection methods in Durham Junior Cricket.

## Overview

This project is part of a Master of Data Science research dissertation at Durham University (2025), focusing on statistical analysis and machine learning approaches to improve youth cricket talent identification and selection processes. The research aims to provide evidence-based recommendations for optimizing player selection criteria and development pathways in junior cricket.

## Research Objectives

- Analyze historical player performance data to identify key performance indicators
- Validate current selection methodologies using statistical techniques
- Develop predictive models for player development trajectories
- Compare traditional scouting methods with data-driven approaches
- Provide actionable insights for coaches and selectors

## Features

- **Player Performance Analysis**: Comprehensive statistical analysis of batting, bowling, and fielding metrics
- **Selection Criteria Validation**: Rigorous testing of existing selection criteria effectiveness
- **Statistical Modeling**: Advanced statistical models for player development prediction
- **Machine Learning Pipeline**: Automated feature selection and model optimization
- **Comparative Analysis**: Side-by-side comparison of different selection methodologies
- **Interactive Dashboards**: Web-based visualization tools for stakeholders
- **Automated Reporting**: Generate standardized reports for coaching staff

## Project Structure

```
durham-junior-cricket-analysis/
├── data/
│   ├── raw/            # Original cricket databases and match records
│   ├── processed/      # Cleaned and transformed datasets
│   └── external/       # Third-party cricket statistics
├── src/
│   ├── data_processing/    # ETL scripts and data cleaning
│   ├── analysis/          # Statistical analysis modules
│   ├── modeling/          # Machine learning models
│   ├── visualization/     # Plotting and dashboard creation
│   └── utils/            # Helper functions and utilities
├── notebooks/
│   ├── exploratory/      # Initial data exploration
│   ├── analysis/         # Detailed statistical analysis
│   └── modeling/         # Model development and testing
├── results/
│   ├── figures/          # Generated plots and charts
│   ├── reports/          # Analysis reports and summaries
│   └── models/           # Trained model artifacts
├── tests/              # Unit tests and validation scripts
├── config/             # Configuration files and parameters
└── docs/              # Documentation and research papers
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Jupyter Notebook/Lab for interactive analysis
- Required packages listed in `requirements.txt`

### System Requirements

- Minimum 8GB RAM (16GB recommended for large datasets)
- 5GB free disk space
- Internet connection for data updates

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/durham-junior-cricket-analysis.git
cd durham-junior-cricket-analysis
```

2. Create a virtual environment:
```bash
python -m venv cricket_analysis_env
source cricket_analysis_env/bin/activate  # On Windows: cricket_analysis_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp config/config_template.yaml config/config.yaml
# Edit config.yaml with your specific settings
```

## Usage

### Quick Start

Run the complete analysis pipeline:
```bash
python src/main.py --config config/config.yaml
```

### Step-by-Step Analysis

1. **Data Processing**:
```bash
python src/data_processing/clean_data.py
```

2. **Exploratory Analysis**:
```bash
jupyter notebook notebooks/exploratory/initial_analysis.ipynb
```

3. **Statistical Analysis**:
```bash
python src/analysis/performance_analysis.py
python src/analysis/selection_validation.py
```

4. **Model Training**:
```bash
python src/modeling/train_models.py --model-type all
```

5. **Generate Reports**:
```bash
python src/visualization/generate_dashboard.py
```

### Configuration Options

Key configuration parameters in `config/config.yaml`:
- Data source paths
- Analysis parameters
- Model hyperparameters
- Visualization settings
- Output directories

## Data Sources

- Durham Junior Cricket League match records (2018-2024)
- Player registration and demographic data
- Training session attendance records
- External cricket statistics APIs
- Weather and ground condition data

## Methodology

### Statistical Analysis
- Descriptive statistics and trend analysis
- Correlation analysis between performance metrics
- Time series analysis of player development
- Hypothesis testing for selection criteria validation

### Machine Learning Approaches
- Supervised learning for performance prediction
- Unsupervised clustering for player categorization
- Feature importance analysis
- Cross-validation and model evaluation

## Results and Findings

Detailed results are available in the `results/reports/` directory, including:
- Player performance trend analysis
- Selection criteria effectiveness study
- Predictive model performance metrics
- Recommendations for coaching staff

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

For coverage reports:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Contributing

This is an academic research project. For collaboration inquiries:
1. Review the research objectives and methodology
2. Check existing issues and discussions
3. Contact the author through Durham University channels
4. Follow the code of conduct for academic research

## Ethical Considerations

- All player data is anonymized and handled according to GDPR guidelines
- Research ethics approval obtained from Durham University
- Data sharing agreements in place with Durham Junior Cricket
- Results shared with stakeholders for validation

## License

This project is part of academic research at Durham University. The code is available under the MIT License for educational and research purposes. Data usage is subject to separate agreements with Durham Junior Cricket.

## Citation

If you use this work in your research, please cite:
```
[Author Name]. (2025). Durham Junior Cricket Analysis: Statistical Validation of U15 Player Selection Methods. 
Master of Data Science Dissertation, Durham University.
```

## Contact

For questions regarding this research:
- Primary Researcher: [Author] - [email]@durham.ac.uk
- Supervisor: [Supervisor Name] - Durham University
- Durham Junior Cricket: [Contact Information]

## Acknowledgments

- Durham University Department of Computer Science
- Durham Junior Cricket League
- Research participants and coaching staff
- Open-source community for tools and libraries used

