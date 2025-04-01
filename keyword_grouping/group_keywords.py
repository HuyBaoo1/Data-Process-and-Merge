from collections import defaultdict
from fuzzywuzzy import fuzz

def group_keywords(keywords):
    keyword_groups = defaultdict(list)
    
    for keyword in keywords:
        matched = False
        for group in keyword_groups:
            if fuzz.ratio(keyword, group) > 80:
                keyword_groups[group].append(keyword)
                matched = True
                break
        if not matched:
            keyword_groups[keyword].append(keyword)

    return {variant: main for main, variants in keyword_groups.items() for variant in variants}
