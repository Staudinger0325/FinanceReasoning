## FinanceReasoning

The data and code for the paper `FinanceReasoning: Make Financial Numerical Reasoning More Credible, Comprehensive, and Challenging`.

**FinanceReasoning** is a knowledge-intensive dataset focused on financial numerical reasoning. It requires the model to comprehend specialized financial terminology and to interpret tabular data presented in the questions. 

## FinanceReasoning Dataset
All the data examples were divided into three subsets: *hard*, *medium*, and *easy*.

- **hard**: 238 examples used for model development, validation, or for those with limited computing resources.
- **medium**: 1000 examples for standard evaluation. We will not publicly release the annotated solution and answer for the test set.
- **easy**: 1000 examples for standard evaluation. We will not publicly release the annotated solution and answer for the test set.

The dataset is provided in json format and contains the following attributes at the `data` directory:

```json
{
    "question_id": "[string] Unique identifier for the question",
    "question": "[string] The question text, typically a financial data analysis problem",
    "context": "[string] Background information for the question, including tabular data in Markdown format",
    "statistics": {
        "number_statistics": "[object] Statistics about numbers, including count of numbers in the question",
        "operator_statistics": "[object] Statistics about operator usage, tracking frequency of different operators",
        "code_statistics": "[object] Code-related statistics, such as number of code lines"
    },
    "python_solution": "[string] Python solution code written by financial experts, with clear variable names and execution logic",
    "ground_truth": "[number / boolean] The standard answer, typically the result of executing the Python solution",
    "difficulty": "[float] Difficulty coefficient of the question, higher values indicate greater difficulty",
    "level": "[string] Difficulty level classification of the question (e.g., hard, medium, easy)",
    "source": "[string] Source identifier of the question"
}
```

## Experiments

### Environment Setup
You can install the dependencies by the following command:
```bash
pip install -r requirements.txt
```

### Configuration
The `config/config.yaml` file controls all aspects of inference and evaluation:
- Inference settings (e.g., dataset, subset, model, prompt type)
- Evaluation settings
- Model configurations (API keys, base URLs, sampling parameters)

### Running Inference
We support inference with various LLM models through two approaches:

1. **Configuration-based Inference**
   ```bash
   python inference.py --config config/config.yaml
   ```
   This method uses the configuration file to specify model settings, dataset parameters, and inference options. Key configurations include:
   - Model selection (`model_name`)
   - Dataset subset (`hard`, `medium`, `easy`)
   - Prompt type (`cot`, `pot`)
   - Output directory

2. **Batch API Inference**
   ```bash
   python utils/openai_batch.py \
     --dataset "FinanceReasoning" \
     --subset "hard" \
     --prompt "cot" \
     --model "your_model_id" \
     --api_key "your_api_key" \
     --base_url "your_base_url"
   ```
   This method allows you to get 50% discount on the openai inference cost.

### Model Output
Inference results are stored in the `results` directory, organized by:
- Dataset name
- Dataset subset (`hard`, `medium`, `easy`)
- Prompt type
- Model name

### Automated Evaluation
Evaluate model performance using:
```bash
python evaluation.py --config config/config.yaml
```