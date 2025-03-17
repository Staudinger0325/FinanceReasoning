import json
import time
from tqdm import tqdm
import os
from openai import OpenAI
import anthropic
import requests
from utils.openai_utils import *
from utils.prompts import *
import asyncio
from dotenv import load_dotenv

def prepare_model_inputs(system_inputs, user_inputs, model_name, api_based=True, tokenizer=None):
    model_inputs = []
    for system_input, user_input in zip(system_inputs, user_inputs):
        models_without_system = ("gemma", "OLMo", "Mistral", "Mixtral", "starcoder2")
        if any(model in model_name for model in models_without_system):
            model_input = [
                {"role": "user", "content": system_input + "\n" + user_input}
            ]
        else:
            model_input = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input}
            ]
        if not api_based:
            model_input = tokenizer.apply_chat_template(model_input, tokenize=False)
        model_inputs.append(model_input)
    
    return model_inputs

def generate_response_parallel(model_inputs, model_name):
    # 加载 .env 文件
    load_dotenv()

    if "deepseek" in model_name or "qwen" in model_name:
        api_base = os.environ["ALIYUN_BASE_URL"]
        api_key = os.environ["ALIYUN_API_KEY"]
    else:
        api_base = os.environ["OPENAI_BASE_URL"]
        api_key = os.environ["OPENAI_API_KEY"]
    
    os.environ["OPENAI_BASE_URL"] = api_base
    os.environ["OPENAI_API_KEY"] = api_key
    print(api_base, api_key)
    client = AsyncOpenAI()
    AsyncOpenAI.api_key = api_key



    raw_outputs, reasoning_contents, usages = asyncio.run(generate_from_openai_chat_completion( 
                                        client = client,
                                        messages = model_inputs,
                                        engine_name = model_name, 
                                        temperature = 1.0, 
                                        top_p = 1.0, 
                                        max_tokens = 8192,
                                        requests_per_minute = 200,))

    return raw_outputs, reasoning_contents, usages
    

def test_retriever(queries, top_k=30, model="contriever-msmarco"):
    retrieved_functions = []
    for query in queries:
        response = requests.post(
            "http://localhost:8088/retrieve",
            json={
                "query": query,
                "top_k": top_k,
                "model": model,
            },
        )
        json_response = response.json()
        retrieved_functions.append([doc['function'] for doc in json_response[0]['retrieved_documents']])
    return retrieved_functions

def judge_useful_functions_from_cache(cache_file, llm_instruct):
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    if llm_instruct:
        retrieval_queries = [example['retrieval_query'] for example in cache_data]
    else:
        retrieval_queries = [example['question'] for example in cache_data]
    useful_functions = [example['useful_functions'] for example in cache_data]
    
    return retrieval_queries, useful_functions

def judge_useful_functions(question_inputs, function_inputs, model_name, top_k):
    system_inputs = []
    user_inputs = []
    for question_input, function_input in zip(question_inputs, function_inputs):
        system_input = JUDGE_USEFUL_FUNCTIONS_SYSTEM_INPUT
        user_input = JUDGE_USEFUL_FUNCTIONS_USER_INPUT.format(question_input=question_input, function_input=function_input)
        system_inputs.append(system_input)
        user_inputs.append(user_input)
    
    model_inputs = prepare_model_inputs(system_inputs, user_inputs, model_name)
    raw_outputs, _, _ = generate_response_parallel(model_inputs, model_name)
    useful_functions = []
    for raw_output, function_input in zip(raw_outputs, function_inputs):
        # delete the first and last character
        judgments = raw_output[1:-1].split(',')
        # control the number of useful functions < 3, if > 3, take the first 3
        useful_function = [function_input[index] for index, judgment in enumerate(judgments) if judgment.strip() == 'USEFUL' and index < top_k]
        if len(useful_function) > 3:
            useful_function = useful_function[:3]
        useful_functions.append(useful_function)
    
    return useful_functions

def generate_retrieval_queries(question_inputs, model_name):
    system_inputs = []
    user_inputs = []

    for question_input in question_inputs:
        system_input = GENERATE_RETRIEVAL_QUERY_SYSTEM_INPUT
        user_input = GENERATE_RETRIEVAL_QUERY_USER_INPUT.format(question_input=question_input)
        system_inputs.append(system_input)
        user_inputs.append(user_input)
    
    model_inputs = prepare_model_inputs(system_inputs, user_inputs, model_name)

    retrieval_queries, _, _ = generate_response_parallel(model_inputs, model_name)
    return retrieval_queries

def get_retrieved_functions_from_cache(qa_data, cache_file, retrieved_type, llm_instruct=False, model_name=None, top_k=30):
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    if llm_instruct:
        retrieval_queries = [cache_data[example['question_id']][model_name]['query'] for example in qa_data]
        if retrieved_type == 'passage':
            retrieved_functions = [cache_data[example['question_id']][model_name]['retrieved_passages'][:top_k] for example in qa_data]
        elif retrieved_type == 'summary':
            retrieved_functions = [cache_data[example['question_id']][model_name]['reasoning_summary']['o1'] for example in qa_data]
        else:
            retrieved_functions = [cache_data[example['question_id']][model_name]['retrieved_functions'][:top_k] for example in qa_data]
    else:
        retrieval_queries = [cache_data[example['question_id']]['Vanilla']['query'] for example in qa_data]
        if retrieved_type == 'passage':
            retrieved_functions = [cache_data[example['question_id']]['Vanilla']['retrieved_passages'][:top_k] for example in qa_data]
        elif retrieved_type == 'summary':
            retrieved_functions = [cache_data[example['question_id']]['Vanilla']['reasoning_summary']['o1'] for example in qa_data]
        else:
            retrieved_functions = [cache_data[example['question_id']]['Vanilla']['retrieved_functions'][:top_k] for example in qa_data]
    
    return retrieval_queries, retrieved_functions


def prepare_inference_inputs(qa_data, prompt_type, model_name, llm_instruct, use_retrieved_cache, retrieved_type, use_useful_cache, judge_useful_functions, top_k, api_based=True, tokenizer=None):
    system_inputs = []
    user_inputs = []
    question_inputs = []
    retrieval_queries = []
    useful_functions = []

    for example in qa_data:
        system_input = MODEL_PROMPT_DICT[prompt_type]["system"]
        system_inputs.append(system_input)

        if example.get("tables", None):
            table_input = "Table:\n"
            table_input += "\n\n".join(example["tables"]) + "\n"
            question_input = f"{table_input}\nQuestion: {example['question']}\n"
        elif example.get("context", None):
            context_input = "The following question context is provided for your reference."
            context_input += example["context"] + "\n"
            question_input = f"{context_input}\nQuestion: {example['question']}\n"
        else:
            question_input = f"Question: {example['question']}\n"
        question_inputs.append(question_input)

    if "rag" in prompt_type:
        if use_useful_cache:
            cache_file = f"/home/bupt/FinanceAgent/outputs/{dataset}/{subset}/raw_pot_outputs/{model_name}-rag.json"
            retrieval_queries, useful_functions = judge_useful_functions_from_cache(cache_file, llm_instruct)
        else:
            if use_retrieved_cache:
                cache_file = f"/home/bupt/FinanceAgent/cache/cache-{dataset}-{subset}.json"
                retrieval_queries, retrieved_functions = get_retrieved_functions_from_cache(qa_data, cache_file, retrieved_type, llm_instruct, model_name, top_k)
            else:
                if llm_instruct:
                    retrieval_queries = generate_retrieval_queries(question_inputs, model_name)
                    retrieved_functions = test_retriever(retrieval_queries, top_k)
                else:
                    retrieved_functions = test_retriever(example["question"], top_k)
            if judge_useful_functions:
                useful_functions = judge_useful_functions(question_inputs, retrieved_functions, model_name, top_k)
            else:
                if retrieved_type == 'passage':
                    useful_functions = [retrieved_function[:10] for retrieved_function in retrieved_functions]
                elif retrieved_type == 'summary':
                    useful_functions = [retrieved_function for retrieved_function in retrieved_functions]
                else:
                    useful_functions = [retrieved_function[:3] for retrieved_function in retrieved_functions]
        
        for question_input, useful_function in zip(question_inputs, useful_functions):
            if len(useful_function) > 0:
                if retrieved_type == 'passage':
                    knowledge_input = "The following are the financial knowledge for your reference.\n"
                    knowledge_input += "\n".join(useful_function) + "\n"
                elif retrieved_type == 'function':
                    knowledge_input = "The following are the financial functions for your reference.\n"
                    knowledge_input += "\n".join(useful_function) + "\n"
                else:
                    knowledge_input = "The following are the summary of reasoning process provided by another reasoning model you should strictly follow.\n"
                    knowledge_input += "\n".join(useful_function) + "\n"
            else:
                knowledge_input = ""
            
            program_prefix_input = MODEL_PROMPT_DICT[prompt_type]["program_prefix"]
            user_input = "\n".join([question_input, knowledge_input, program_prefix_input])
            user_inputs.append(user_input)
    else:
        for question_input in question_inputs:
            program_prefix_input = MODEL_PROMPT_DICT[prompt_type]["program_prefix"]        
            user_input = "\n".join([question_input, program_prefix_input])
            user_inputs.append(user_input)
            retrieval_queries.append("")
            useful_functions.append([])

    

    model_inputs = prepare_model_inputs(system_inputs, user_inputs, model_name, api_based, tokenizer)

    return model_inputs, retrieval_queries, useful_functions, system_inputs, user_inputs

def main(input_file, output_file, prompt_type, model_name, llm_instruct, use_retrieved_cache, retrieved_type, use_useful_cache, judge_useful_functions, top_k):
    with open(input_file, 'r') as f:
        qa_data = json.load(f)

    model_inputs, retrieval_queries, useful_functions, system_inputs, user_inputs = prepare_inference_inputs(qa_data, prompt_type, model_name, llm_instruct, use_retrieved_cache, retrieved_type, use_useful_cache, judge_useful_functions, top_k, api_based=True)
    
    # raw_outputs, reasoning_contents, usages = generate_response_parallel(model_inputs, model_name)
    
    # for index, (raw_output, qa, retrieval_query, useful_function) in enumerate(zip(raw_outputs, qa_data, retrieval_queries, useful_functions)):
    #     qa["output"] = [raw_output]
    #     qa["retrieval_query"] = retrieval_query
    #     qa["useful_functions"] = useful_function
    #     if model_name == "deepseek-r1":
    #         qa["reasoning_content"] = reasoning_contents[index]
    #     qa["usage"] = usages[index]
    # json.dump(qa_data, open(output_file, "w"), indent=4, ensure_ascii=True)

    prompts_file = f"/home/bupt/FinanceAgent/outputs/FinanceReasoning/hard/prompts/{prompt_type}.json"
    prompts_data = []
    for qa, system_input, user_input in zip(qa_data, system_inputs, user_inputs):
        prompts_data.append({
            "question_id": qa["question_id"],
            "system_input": system_input,
            "user_input": user_input,
            "ground_truth": qa["ground_truth"]
        })
    json.dump(prompts_data, open(prompts_file, "w"), indent=4, ensure_ascii=True)

if __name__ == "__main__":
    prompt_types = ['cot', 'pot', 'cot_rag', 'pot_rag', 'cot_rag_reasoning', 'pot_rag_reasoning', 'cot_rag_reasoning_mini', 'pot_rag_reasoning_mini']
    dataset = 'FinanceReasoning'
    subset = 'hard'
    prompt_type = 'cot_rag'
    model_name = 'gpt-4o-2024-11-20'
    model_name_file = 'gpt-4o-2024-11-20'
    llm_instruct = True
    use_article = True
    use_reasoning = False
    use_retrieved_cache = False
    retrieved_type = 'function'
    judge_useful_functions = True
    use_useful_cache = True
    top_k = 30
    input_file = f'./data/{dataset}/{subset}.json'
    if 'rag' in prompt_type:
        if llm_instruct:
            output_file = f'./results/{dataset}/{subset}/raw_{prompt_type[:3]}_outputs/{model_name_file}-rag-llm-instruct.json'
        else:
            output_file = f'./results/{dataset}/{subset}/raw_{prompt_type[:3]}_outputs/{model_name_file}-rag.json'
        if use_article:
            output_file = output_file.replace('.json', '-article.json')
        if use_reasoning:
            output_file = output_file.replace('.json', '-reasoning.json')
        if "mini" in prompt_type:
            output_file = output_file.replace('.json', '-mini.json')
        if "without_judge" in prompt_type:
            output_file = output_file.replace('.json', '-without_judge.json')
        if "passage" in prompt_type:
            output_file = output_file.replace('.json', '-passage.json')
        if "summary" in prompt_type:
            output_file = output_file.replace('.json', '-summary.json')
    else:
        output_file = f'/home/bupt/FinanceAgent/outputs/{dataset}/{subset}/raw_{prompt_type[:3]}_outputs/{model_name_file}.json'
        if use_article:
            output_file = output_file.replace('.json', '-article.json')
        if use_reasoning:
            output_file = output_file.replace('.json', '-reasoning.json')
        if "mini" in prompt_type:
            output_file = output_file.replace('.json', '-mini.json')
        if "without_judge" in prompt_type:
            output_file = output_file.replace('.json', '-without_judge.json')
        if "passage" in prompt_type:
            output_file = output_file.replace('.json', '-passage.json')
        if "summary" in prompt_type:
            output_file = output_file.replace('.json', '-summary.json')
    
    main(input_file, output_file, prompt_type, model_name, llm_instruct, use_retrieved_cache, retrieved_type, use_useful_cache, judge_useful_functions, top_k)