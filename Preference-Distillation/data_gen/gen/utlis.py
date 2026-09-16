import json
import numpy as np
import torch
from string import Template
from jinja2.exceptions import TemplateError


TEMPLATE_PATH = 'data_gen/gen/mcq-template.json'


class float32_context:
    def __enter__(self):
        self.prev_dtype = torch.get_default_dtype()  # 保存当前的默认dtype
        torch.set_default_dtype(torch.float32)       # 设置为float32

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.prev_dtype)     # 退出上下文后恢复原来的dtype


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_template(tokenizer):
    with open(TEMPLATE_PATH, encoding='utf-8') as f:
        template = json.load(f)
    try:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": template['system']},
                {'role': 'user', 'content': template['user']}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
    except TemplateError:
        prompt = tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': f"{template['system']}\n\n{template['user']}"}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
    finally:
        return Template(prompt)
