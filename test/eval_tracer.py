from dataclasses import dataclass, asdict, is_dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from io import TextIOWrapper
import json
from mlux_reactly import Tracer



@dataclass
class TraceRecordConfig:
    session: str
    record_file: TextIOWrapper|None = None

class EvalTracer(Tracer):
    config: TraceRecordConfig
    counters: Dict[str, int]
    transparent: Tracer|None
    created: datetime

    def __init__(self, *, 
                 config: TraceRecordConfig|None = None, 
                 session: str|None = None,
                 record_file: TextIOWrapper|None = None,
                 counters: Dict[str, int] = {},
                 transparent: Tracer = None):
        self.config = config or TraceRecordConfig(session=session or f"EVAL", record_file=record_file)
        self.counters = counters
        self.transparent = transparent
        self.created = datetime.now()

    def on(self, key: str, args: Dict[str, Any]) -> "EvalTracer":
        time = datetime.now()

        self._increment('nr_events')
        self._increment(key)

        if key == "try" and True:#args.get('nr', -100) > 0:
            self._increment('retry')
        
        return EvalTracer(config=self.config, counters=self.counters, 
                          transparent=self.transparent.on(key, args) 
                          if self.transparent is not None else None)
    
    def add_arg(self, arg_name: str, arg: Any):
        if self.transparent is not None:
            self.transparent.add_arg(arg_name, arg)

    def _increment(self, counter_key: str):
        if self.counters.get(counter_key) is None:
            self.counters[counter_key] = 1
        else:
            self.counters[counter_key] += 1

    def _record_to_file(self, user_question: str, answer: str) -> None:
        if self.config.record_file is not None:
            self_as_dict = {
                'session': str(self.created) + self.config.session,
                'query': user_question,
                'response': answer,
                'diagnostics': {
                    'counters': self.counters
                }
            }
            self.config.record_file.write(json.dumps(self_as_dict) + "\n")
            self.config.record_file.flush()