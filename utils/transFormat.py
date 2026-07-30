import json
def format_json_for_prompt(data):
    """格式化 format json for prompt 相关内容。"""
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))