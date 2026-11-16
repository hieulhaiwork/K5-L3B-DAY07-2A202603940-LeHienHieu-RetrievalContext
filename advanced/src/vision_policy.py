"""Scheduling and per-run spending controls for GPT-4o extraction."""

def plan_requests(records, task_for, table_ratio=.85, text_ratio=.5, min_records=10):
    populations = {"table":0,"text":0}
    failures = {"table":0,"text":0}
    jobs = []
    for position, record in enumerate(records):
        if record.get("status","").startswith("excluded_"):
            continue
        group = ("table" if record.get("layout_label")=="table" or "table_candidate_text" in record
                 or record["content_type"]=="table" else
                 "text" if record["content_type"]=="text" else None)
        task = task_for(record)
        if group:
            populations[group] += 1
            failures[group] += task is not None
        if task:
            priority = {"visual":0,"table":1,"text":2}[task]
            jobs.append((priority,position,record,task))
    ratios = {group:{"eligible":populations[group],"needs_fallback":failures[group],
                    "ratio":round(failures[group]/populations[group],4) if populations[group] else 0}
              for group in populations}
    guards = [group for group,limit in [("table",table_ratio),("text",text_ratio)]
              if populations[group]>=min_records and failures[group]/populations[group]>limit]
    jobs.sort(key=lambda item:item[:2])
    return jobs,ratios,guards

def reserve_request(stats, prompt, task, token_budget, call_limit):
    if stats["new_calls"] >= call_limit:
        return None,"vision_run_call_limit_reached"
    # Conservative text-byte upper bound plus protocol allowance and up to
    # sixteen 512px tiles for a high-detail GPT-4o image in a 2048px square.
    # Reference: https://developers.openai.com/api/docs/guides/images-vision
    input_estimate = len(prompt.encode("utf-8")) + 1024 + 85 + 170*16
    remaining = token_budget - stats["budget_charged_tokens"]
    maximum = min(6000, remaining-input_estimate)
    minimum = 1024 if task=="table" else 256 if task=="text" else 384
    if maximum < minimum:
        return None,"vision_token_budget_reached"
    allocation = {"estimated_input_tokens":input_estimate,
                  "max_output_tokens":int(maximum),
                  "reserved_tokens":input_estimate+int(maximum)}
    stats["budget_charged_tokens"] += allocation["reserved_tokens"]
    return allocation,None

def settle_usage(stats, allocation, usage):
    # If the provider returned no usable usage (e.g. a network failure), keep
    # the entire reservation charged instead of assuming that the call was free.
    if (isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0
            for k in ["prompt_tokens","completion_tokens"])):
        actual = usage["prompt_tokens"] + usage["completion_tokens"]
        stats["budget_charged_tokens"] += actual-allocation["reserved_tokens"]
        stats["prompt_tokens"] += usage["prompt_tokens"]
        stats["completion_tokens"] += usage["completion_tokens"]
        if actual > allocation["reserved_tokens"]:
            stats["token_estimate_exceeded"] = True
        return "provider_usage"
    stats["usage_unknown_calls"] += 1
    return "reserved_unknown_usage"
