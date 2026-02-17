from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import json
import sys
import os

from mlux_reactly import LLM, Tracer
from test_types import Example, ExampleCase, AgentConfig, AgentContructor, at_or
from test_support import run_hotpotlike_examples
from tools import calculator, text_count, wikipedia_search, make_rag_for_documents, Document


def parse_hotspot_file(file_data: List[Dict[str, Any]]) -> List[ExampleCase]:
    datapoints = []
    for dp in file_data:
        example = Example(
            dp.get('_id'),
            dp.get('question'),
            dp.get('answer'),
        )
        documents = [Document(text="\n".join(doc[1]), metadata={'document_name': doc[0]}) for doc in dp.get('context')]
        agent_config = AgentConfig(
            tools=[make_rag_for_documents(documents), calculator, text_count]
        )
        datapoints.append(ExampleCase(example, agent_config))
    return datapoints


def parse_custom_qa_file(file_data: List[Dict[str, Any]]) -> List[ExampleCase]:
    example_cases = []

    if len(file_data) > 0 and file_data[0].get('justification') is not None:
        with open("test-files/fox/fox-docs.json", "r", encoding="utf-8") as f:
            docs_data = json.load(f)
            documents = [Document(text=doc['content'], metadata={'document_name': doc['title']}) for doc in docs_data]

    for i, dp in enumerate(file_data):
        example = Example(
            'QA' + str(i),
            dp.get('question'),
            dp.get('answer'),
        )
        agent_config = AgentConfig(
            tools=[calculator, text_count]
        )

        if documents:
            agent_config.tools.append(make_rag_for_documents(documents))
        else:
            agent_config.tools.append(wikipedia_search)

        example_cases.append(ExampleCase(example, agent_config))
    return example_cases

def _load_available_sets(folder_path: str) -> Dict[str, str]:
    try:
        files = os.listdir(folder_path)
        return {f.split('.')[0]: f"{folder_path}/{f}" for f in files if f.endswith('.json')}
    except:
        return {}


@dataclass
class TestInfo:
    parse_fn: Callable[[Any], List[ExampleCase]]
    default_set: str
    available_sets: Dict[str, str]

available_tests: Dict[str, TestInfo] = {
    'hotpot': TestInfo(parse_hotspot_file, 'train', {
        'train': 'test-files/hotpot/hotpot_train_v1.1_FIRST_1000.json'
    } | _load_available_sets('test-files/hotpot')),
    'qa': TestInfo(parse_custom_qa_file, '', _load_available_sets('test-files/custom-qa-sets')),
}


async def qa_file_test_fn(test_name: str, test_param: str|None, agent_constr: AgentContructor, llm: LLM, tracer: Tracer, talky: bool = True, agent_name: str = ""):
    test = available_tests[test_name]
    param_splits = (test_param or "").split(':')
    set_name = at_or(param_splits, 0, test.default_set)
    start_pos = int(at_or(param_splits, 1, 0))
    size = int(at_or(param_splits, 2, 1))
    end_pos = start_pos+size
    if set_name not in test.available_sets:
        raise AssertionError(f"{test_name} test: param set name '{set_name}' not available. Available: {test.available_sets}")
    file_name = test.available_sets[set_name]

    with open(file_name, 'r') as file:
        data = json.load(file)
        parse_fn = test.parse_fn
        if start_pos < 0 or start_pos >= len(data) or end_pos <= start_pos or end_pos > len(data):
            raise ValueError(f"invalid test range: got {start_pos} .. {end_pos}, available 0 .. {len(data)}")
        datapoints = parse_fn(data[start_pos:end_pos])
        evaluation = await run_hotpotlike_examples(datapoints, agent_constr, tracer, llm, talky=talky, evalRunCsv={'test': test_name, 'param':test_param, 'agent': agent_name})

        return evaluation