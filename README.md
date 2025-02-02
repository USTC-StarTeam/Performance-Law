# README for Predictive Modeling in Sequential Recommendation: Bridging Performance Laws with Data Quality Insights

This README provides an overview of the project and instructions on how to navigate and utilize the different components available in the `/General_Transformer` and `/Performance_Law_Appendix_Result` directories.

## Project Overview

The project focuses on advancing sequential recommendation systems through innovative models and performance law fitting strategies. It is divided into two main components:

1. **General Transformer for Sequential Recommendation**: Located in the `/General_Transformer` directory, this component implements a general transformer architecture for sequential recommendation tasks.
2. **Performance Law Fitting Analysis**: Found in the `/Performance_Law_Appendix_Result` directory, this component focuses on fitting performance laws for metrics like HR (Hit Rate) and NDCG (Normalized Discounted Cumulative Gain).

### Directory Structure

- **/General_Transformer**: Contains scripts and code for training and evaluating transformer models for recommendation systems.
- **/Performance_Law_Appendix_Result**: Includes scripts and generated results for performance law fitting analysis, along with supplementary images referenced in the research paper.

## Details

### General Transformer for Sequential Recommendation

The scripts in the `/General_Transformer` directory are designed to train transformer models tailored for sequential recommendation tasks. The main features are:

- **Model Training**: Utilizing DDP (Distributed Data Parallel) to efficiently train on multiple GPUs.
- **Hyperparameter Configurations**: Flexible adjustments for layers, heads, batch sizes, and more.
- **Logging and Evaluation**: Detailed performance metrics are logged using libraries like WandB.

To learn more about using these scripts, refer to the README provided within the `/General_Transformer` directory.

### Performance Law Fitting Analysis

Located in the `/Performance_Law_Appendix_Result`, this segment of the project analyzes performance laws through an innovative fitting approach. Key elements include:

- **Performance Law Fitting**: Detailed scripts for fitting performance laws to key metrics.
- **Supplementary Images**: Includes images such as `PerformanceLaw_HR`, `PerformanceLaw_NDCG`, `ScalingLaw_HR`, and `ScalingLaw_NDCG` for deeper insights and validation of research findings these images serve as supplementary material for the paper.

To understand the scripts and their execution, refer to the README within the `/Performance_Law_Appendix_Result` directory.

## Getting Started

1. **Navigate to the relevant directory**:
   - For transformer models, explore `/General_Transformer`.
   - For performance law analysis, visit `/Performance_Law_Appendix_Result`.
   
2. **Install required dependencies**: Ensure all necessary Python libraries are installed as indicated in the README files within each directory.

3. **Run the scripts**: Follow the instructions to execute model training, evaluation, or performance fitting as required.

4. **Explore Results and Graphs**: Analyze outputs, performance metrics, and graphical results included in each component.
