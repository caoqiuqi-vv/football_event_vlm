"""Event-specific human review requirements; applied only to new confirmations."""
from copy import deepcopy

RULES_VERSION = 'event-attribution-20260909'
TEAM_SET_PIECES = {'corner', 'free_kick', 'penalty'}


def normalize_segment_attribution(payload):
    data = deepcopy(payload)
    if data.get('review_outcome') == 'needs_confirmation':
        return data
    selected = data.get('selected_labels', [])
    attributions = data.get('attribution_by_label') or {}
    if not isinstance(attributions, dict):
        raise ValueError('事件归属格式错误')
    details = data.get('secondary_labels_by_label') or {}
    if not isinstance(details, dict):
        raise ValueError('事件细分类格式错误')
    for label in selected:
        value = attributions.get(label) or {}
        if not isinstance(value, dict):
            raise ValueError('事件归属格式错误')
        if label == 'save':
            if value.get('field_side') not in ('left', 'right'):
                raise ValueError('扑救事件请选择左半场或右半场')
            value = {'event_team': 'unknown', 'field_side': value['field_side'], 'goal_side': 'unknown'}
        elif label == 'throw_in':
            value = {'event_team': 'unknown', 'field_side': 'unknown', 'goal_side': 'not_applicable'}
        elif label == 'shot' or (label == 'set_piece' and TEAM_SET_PIECES.intersection(details.get('set_piece', []))):
            if value.get('event_team') not in ('teamA', 'teamB'):
                raise ValueError('射门、角球、任意球、点球事件请选择队伍颜色')
            value = {'event_team': value['event_team'], 'field_side': 'unknown',
                     'goal_side': 'not_applicable' if label == 'set_piece' else 'unknown'}
        attributions[label] = value
    data['attribution_by_label'] = attributions
    return data
