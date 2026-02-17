from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import hashlib

from test_types import Agent, AgentContructor, TestFunc, as_list
from mlux_reactly import ReactlyAgent, LLM, Tracer, ZeroTracer
from llama_index_agent import LlamaFunctionAgentWrapper, LlamaReActAgentWrapper
from run_evaluation_qa_file import qa_file_test_fn



argp = argparse.ArgumentParser()

argp.add_argument(
    "-tests", "-test", "-t",
    nargs="+",
    required=True,
    help="Test set name"
)
argp.add_argument(
    "-agents", "-agent", "-a",
    nargs="+",
    help="List of agent names"
)
argp.add_argument(
    "-llms", "-llm", "-l",
    nargs="+",
    help="List of LLM identifiers",
    default="qwen2.5:7b-instruct-q8_0"
)





@dataclass
class EvalRun:
    test: str
    test_param: str|None
    agent_name: str
    llm: str


available_tests: Dict[str, TestFunc] = {
    'hotpot': qa_file_test_fn,
    'qa': qa_file_test_fn,
}
available_agents: Dict[str, AgentContructor] = {
    'reactly': ReactlyAgent,
    'llama-react': LlamaReActAgentWrapper,
    'llama-func': LlamaFunctionAgentWrapper,
}
available_llms: Dict[str, LLM] = {
    'qwen2.5:3b-instruct-q4_K_M': LLM('qwen2.5:3b-instruct-q4_K_M'),
    'qwen2.5:7b-instruct-q8_0': LLM('qwen2.5:7b-instruct-q8_0'),
    'qwen2.5:14b-instruct-q4_K_M': LLM('qwen2.5:14b-instruct-q4_K_M'),
    'llama3.1:8b-instruct-q8_0': LLM('llama3.1:8b-instruct-q8_0'),
    'deepseek-coder:6.7b-instruct': LLM('deepseek-coder:6.7b-instruct'),
}

def assert_arg(name: str, arg: str, availables: Dict[str, Any]):
    if arg not in availables.keys():
        raise AssertionError(f"command line argument {name} '{arg}' not available")



def args_to_eval_runs(args_as_strings: List[str]|None = None) -> List[EvalRun]:
    args = argp.parse_args(args_as_strings)

    for test in as_list(args.tests):
        assert_arg('tests', test.split('/')[0], available_tests)
    for agent in as_list(args.agents):
        assert_arg('agents', agent, available_agents)
    for llm in as_list(args.llms):
        assert_arg('llms', llm, available_llms)

    runs: List[EvalRun] = []

    for test in as_list(args.tests):
        for llm in as_list(args.llms):
            for agent in as_list(args.agents):
                test_param = "/".join(test.split('/')[1:])
                runs.append(EvalRun(
                    test=test.split('/')[0],
                    test_param=test_param if test_param else None,
                    agent_name=agent,
                    llm=llm
                ))

    return runs



# run evaluations

csv_entries_run = ['version', 'formathash', 'eval_prog', 'eval_time_start', 
                   'run_index', 'test', 'test_param', 'agent', 'llm']
csv_entries_evaluation = ['nr_total', 'nr_finished', 'nr_failed', 
               'duration_total', 'duration_avg', 'duration_min', 'duration_max',
               'em', 'f1', 'prec', 'recall', 
               'nr_task_avg', 'nr_toolrun_avg', 'nr_llmcall_avg', 'nr_retry_avg']
csv_entries = csv_entries_run + csv_entries_evaluation
h_blake2b_csv_entries = hashlib.blake2b(digest_size=8)
h_blake2b_csv_entries.update('\0'.join(csv_entries).encode())
csv_entries_hash = int.from_bytes(h_blake2b_csv_entries.digest(), "little")

def evaluation_to_csv(run: EvalRun, run_index: int, evaluation: Dict[str, float], *, eval_time_start: datetime) -> str:
    values = {
        'version': 1,
        'formathash': csv_entries_hash,
        'eval_prog': "eval.py",
        'eval_time_start': eval_time_start.timestamp(),
        'run_index': run_index,
        'test': run.test,
        'test_param': run.test_param,
        'agent': run.agent_name,
        'llm': run.llm
    }
    for entry_name in csv_entries_evaluation:
        values[entry_name] = evaluation.get(entry_name)
    
    entries: List[str] = []
    for entry_name in csv_entries:
        value = values.get(entry_name)
        value_str: str = ""
        if type(value) in [int, float]:
            value_str = str(value)
        elif value is not None:
            value_str = str(f'"{value}"')
        entries.append(value_str)
    return ";".join(entries)



async def main_eval(runs: List[EvalRun], tracer: Tracer = ZeroTracer(), talky: bool = True) -> None:
    eval_time_start = datetime.now()
    results = []
    for i, run in enumerate(runs):
        if talky:
            print(f"running {i}: {run}")

        test_fn = available_tests[run.test]
        agent = available_agents[run.agent_name]
        llm = available_llms[run.llm]

        evaluation = await test_fn(run.test, run.test_param, agent, llm, tracer, talky=talky, agent_name=run.agent_name)
        results.append({
            'i': i,
            'run': run,
            'evaluation': evaluation,
        })

        if talky:
            print(f"---> evaluation {i}: {evaluation}")

    results_file_path = Path('evaluation_results.csv')
    results_file_exists = results_file_path.exists()

    csv_lines = [] if results_file_exists else [";".join(csv_entries)]
    for result in results:
        csv_lines.append(evaluation_to_csv(result['run'], result['i'], result['evaluation'], eval_time_start=eval_time_start))

    with open(results_file_path, 'a') as csv_file:
        csv_file.write('\n'.join(csv_lines) + '\n')
    



if __name__ == '__main__':
    runs = args_to_eval_runs()

    for i, run in enumerate(runs):
        print(f"{i}: {run}")
    print()

    asyncio.run(main_eval(runs))
    print("Done")