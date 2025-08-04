# DRC Names Gender Prediction Pipeline: A Culturally-Aware NLP System for Congolese Name Analysis

A comprehensive, research-friendly pipeline for analyzing Congolese names and predicting gender using culturally-aware machine learning models. 
This system provides advanced data processing, experiment management, and an intuitive web interface for non-technical users.

## Overview

Despite the growing success of gender inference models in Natural Language Processing (NLP), these tools often underperform when applied to culturally diverse African contexts due to the lack of culturally-representative training data. 
This project introduces a comprehensive pipeline for Congolese name analysis with a large-scale dataset of over 7 million names from the Democratic Republic of Congo (DRC) annotated with gender and demographic metadata.

Our approach involves:

- **(1) Advanced data processing pipeline** with batching, checkpointing, and parallel processing
- **(2) Modular experiment framework** for systematic model comparison and research iteration  
- **(3) Multiple feature extraction strategies** leveraging name components, linguistic patterns, and demographic data
- **(4) Culturally-aware gender prediction models** trained specifically on Congolese naming patterns
- **(5) User-friendly web interface** enabling non-technical users to run experiments and make predictions
- **(6) Comprehensive research tools** for reproducible experimentation and result analysis

## Key Features

### **Advanced Data Processing**
- **Batched processing** with configurable batch sizes and parallel execution
- **Automatic checkpointing** and resume capability for large datasets
- **LLM-powered annotation** with rate limiting and retry logic
- **Memory-efficient** chunked data loading for datasets of any size

### **Research-Friendly Experiment Framework**
- **Modular model architecture** - easily add new models and features
- **Systematic experiment tracking** with automatic result storage
- **Feature ablation studies** and component analysis tools
- **Cross-validation** and statistical significance testing
- **Automated baseline comparisons** and performance analysis

### **Intuitive Web Interface**
- **No-code experiment creation** with visual parameter selection
- **Real-time monitoring** of data processing and training progress
- **Interactive result visualization** with charts and comparisons
- **Batch prediction capabilities** for CSV file upload and processing
- **Model comparison tools** with automatic performance rankings

### **Comprehensive Analytics**
- **Feature importance analysis** showing which name components matter most
- **Province-specific studies** examining regional naming patterns
- **Learning curve analysis** for understanding data requirements
- **Prediction confidence scoring** and error analysis tools

## Quick Start

### Using Make Commands (Recommended)

```bash
# Complete setup and basic processing
make quick-start

# Launch web interface
make web

# Run research workflow  
make research-flow

# Show all available commands
make help
```

### Manual Installation

```bash
git clone https://github.com/bernard-ng/drc-ners-nlp.git
cd drc-ners-nlp

# Setup environment
make setup
make process

# Launch web interface
make web
```

## Usage

### Web Interface (Recommended for Non-Technical Users)

Launch the Streamlit web application:
```bash
make web
```

The interface provides:
- **Dashboard**: Overview of datasets and recent experiments
- **Data Overview**: Interactive data exploration and statistics  
- **Data Processing**: Monitor and control the processing pipeline
- **Experiments**: Create and manage machine learning experiments
- **Results & Analysis**: Compare models and analyze performance
- **Predictions**: Make predictions on new names or upload CSV files
- **Settings**: Configure the system and manage data

### Research & Experiments

#### Quick Research Studies
```bash
# Compare different approaches (full name vs native vs surname)
make baseline

# Analyze which name components are most effective
make components  

# Test feature importance through ablation study
make ablation

# View all experiment results
make list-experiments

# Export results for publication
make export-results
```

#### Custom Experiments
```bash
# Run specific experiment via command line
python research/cli.py run \
  --name "native_name_study" \
  --features native_name \
  --model-type logistic_regression \
  --description "Test native name effectiveness"

# Compare multiple experiments
python research/cli.py compare <exp_id_1> <exp_id_2>

# View detailed results
python research/cli.py show <experiment_id>
```

### Data Processing Pipeline

#### Basic Processing (No LLM)
```bash
make process-basic    # Fast processing without LLM annotation
```

#### Complete Processing (With LLM)
```bash
make process         # Full pipeline including LLM annotation
make process-dev     # Development mode with smaller batches
```

#### Monitor Progress
```bash
make monitoring         # Show current pipeline status
make status          # Show overall system status
```

#### Resume Interrupted Processing
```bash
make process-resume  # Resume from last checkpoint
```

### Available Models and Features

#### Models
- **Logistic Regression**: Character n-gram based classification
- **Random Forest**: Engineered feature-based classification
- **LSTM**: Sequential neural network (planned)
- **Transformer**: Attention-based model (planned)

#### Features
- **Full Name**: Complete name as given
- **Native Name**: Identified native/given name component  
- **Surname**: Family name component
- **Name Length**: Character count features
- **Word Count**: Number of words in name
- **Province**: Geographic/demographic features
- **Name Beginnings/Endings**: Prefix/suffix patterns
- **Character N-grams**: Linguistic pattern features

## Configuration

### Environment Configurations

```bash
# Switch to development configuration (smaller batches, more logging)
make config-dev

# Switch to production configuration (optimized for performance) 
make config-prod

# View current configuration
make show-config
```

### Custom Configuration

Edit configuration files in `config/`:
- `pipeline.yaml` - Main configuration
- `pipeline.development.yaml` - Development overrides  
- `pipeline.production.yaml` - Production settings

Example configuration:
```yaml
processing:
  batch_size: 1000
  max_workers: 4
  
llm:
  model_name: "mistral:7b"
  requests_per_minute: 60
  
data:
  split_evaluation: true
  split_by_gender: true
```

## Research Capabilities

### Systematic Experimentation

The framework supports systematic research through:

1. **Baseline Studies**: Compare fundamental approaches
2. **Feature Studies**: Test individual name components  
3. **Ablation Studies**: Identify most important features
4. **Cross-Province Analysis**: Test generalization across regions
5. **Hyperparameter Optimization**: Systematic parameter tuning

### Reproducible Research

- **Experiment Tracking**: All experiments automatically logged with full configuration
- **Result Export**: CSV export for publication and further analysis
- **Statistical Testing**: Cross-validation and confidence intervals
- **Version Control**: Configuration-based approach enables easy replication

### Publication-Ready Output

```bash
# Generate comprehensive results for publication
make research-flow
make export-results

# Get best models for each approach  
make list-completed
python research/cli.py list --status completed | head -10
```

## Development

### Code Quality and Testing
```bash
make format          # Format code with black
make lint           # Lint with flake8  
make check-deps     # Verify dependencies
```

### Development Workflow
```bash
make daily-work     # Daily development setup
make notebook       # Launch Jupyter for analysis
make web-dev        # Launch web interface with auto-reload
```

### Data Management
```bash
make check-data     # Verify all data files
make data-stats     # Show dataset statistics
make backup-data    # Create timestamped backup
make clean-checkpoints  # Clean processing checkpoints
```

## Project Structure

```
├── Makefile                    # All command shortcuts
├── streamlit_app.py           # Web interface application
├── config/                    # Configuration files
│   ├── pipeline.yaml         # Main configuration
│   ├── pipeline.development.yaml  # Dev settings
│   └── pipeline.production.yaml   # Prod settings
├── core/                      # Core framework
│   ├── config.py             # Configuration management
│   ├── domain.py             # Domain-specific data
│   └── utils.py              # Reusable utilities
├── processing/                # Data processing pipeline
│   ├── main.py               # Main pipeline script
│   ├── pipeline.py           # Pipeline framework
│   ├── steps_config.py       # Configurable processing steps
│   └── monitor.py            # Monitoring utilities
├── research/                  # Research and experiments
│   ├── cli.py                # Command-line interface
│   ├── experiment.py         # Experiment management
│   ├── models.py             # Model implementations
│   └── runner.py             # Experiment execution
└── dataset/                   # Data files
    └── names.csv             # Raw dataset
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{drc_names_pipeline,
  title={DRC Names Gender Prediction Pipeline: A Culturally-Aware NLP System},
  author={Your Name},
  year={2025},
  url={https://github.com/bernard-ng/drc-ners-nlp}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Democratic Republic of Congo population data contributors
- Open source NLP and machine learning communities
- Cultural linguistics research communities
