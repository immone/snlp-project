## DSPy optimizer for binary classification

1. Create [OpenRouter](https://openrouter.ai) account
2. Fetch API key and set it to `.env` as `OPENROUTER_API_KEY`
3. Choose an LLM from [this](https://models.litellm.ai) list and set it to `model` var
4. Choose an DSPy [optimizer](https://dspy.ai/learn/optimization/optimizers/) if you wish and modify config in `optimize_program` 
5. Run `python main.py` with optional flags: `--optimizer` runs DSPy optimizer, `--debug` only fetches the first 10 rows from the dataset (to not do too many API calls)

See [here](https://openrouter.ai/docs/api-reference/limits) for API usage limits wrt different LLMs.