from typing import List, Dict, Tuple
import inspect
import traceback
import time
import math
import json
from hotpot_based_evaluation import eval_results, Result
from mlux_reactly import LLM, Tracer
from test_types import Agent, AgentContructor, Example, ExampleCase
from eval_tracer import EvalTracer, TraceRecordConfig



async def run_example_on_agent(example: Example, agent: Agent) -> Tuple[bool, Result, float]:
    # TODO: maybe retry, logging
    start_time = time.perf_counter()
    try:
        result = agent.query("Answer the following question by *just* stating the questioned fact in a few words (For example, 'Where is the Eiffel Tower' would be best answered by 'Paris, France'. No explanation.)."
                             " Question: " + example.question)
        answer: str
        duration: float
        if inspect.isawaitable(result):
            answer = await result
        else:
            answer = result
        duration = time.perf_counter() - start_time
        return True, Result(example.id, answer, None), duration
    except Exception as e:
        print(f"catched exception running example {example.id} on agent {agent.__class__.__name__}: {e}")
        #traceback.print_exc()
        return False, Result(example.id, "", None), math.nan
    

async def run_hotpotlike_examples(example_cases: List[ExampleCase], agent_constr: AgentContructor, tracer: Tracer, llm: LLM|None = None, talky: bool = True, evalRunCsv: str = "") -> Dict[str, float]:
    agent_results: List[Result] = []

    duration_total: float = 0
    duration_min: float = math.inf
    duration_max: float = -math.inf
    nr_finished: int = 0
    trace_counters: Dict[str, int] = {}
    example_log_csv = ""

    for example_case in example_cases:
        evaltracer = EvalTracer(session="eval"+example_case.example.id, transparent=tracer)
        agent = agent_constr(tools=example_case.agent_config.tools, tracer=evaltracer, llm=llm)

        finished, agent_result, duration = await run_example_on_agent(example_case.example, agent)

        if talky:
            print(f"=> {example_case.example.id[:6]}: '{example_case.example.question}' good: '{example_case.example.answer}' A: '{agent_result.answer}'")
        
        #example_log_csv += f'{evalRunCsv};{evaltracer.created.timestamp()};"{llm.model}";{example_case.example.id};"{example_case.example.question}";"{example_case.example.answer}";"{agent_result.answer}"\n'
        example_log_csv += json.dumps({
            'eval_run': evalRunCsv,
            'time': evaltracer.created.timestamp(),
            'llm': llm.model,
            'id': example_case.example.id,
            'question': example_case.example.question,
            'good': example_case.example.answer,
            'answer': agent_result.answer,
        }, ensure_ascii=False) + "\n"

        agent_results.append(agent_result)
        if finished:
            duration_total += duration
            duration_min = min(duration_min, duration)
            duration_max = max(duration_max, duration)
            nr_finished += 1

            for k, v in evaltracer.counters.items():
                k2 = f"nr_{k}_avg"
                if trace_counters.get(k2) is None:
                    trace_counters[k2] = v
                else:
                    trace_counters[k2] += v

    with open('evaluation_log_all.jsonl', 'a') as lf:
        lf.write(example_log_csv)
        lf.flush()

    correct_results = [Result(example_case.example.id, example_case.example.answer, None) for example_case in example_cases]
    evaluation: Dict[str, float] = eval_results(correct_results, agent_results)

    evaluation |= {
        'duration_total': duration_total,
        'duration_min': duration_min,
        'duration_max': duration_max,
        'duration_avg': (duration_total / nr_finished) if nr_finished else math.nan,
        'nr_total': len(example_cases),
        'nr_finished': nr_finished,
        'nr_failed': len(example_cases) - nr_finished,
    } | {k: ((c / nr_finished) if nr_finished else math.nan) for k, c in trace_counters.items()}

    return evaluation