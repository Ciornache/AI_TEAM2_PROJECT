import json

kg = json.load(open('knowledge_graph.json', encoding='utf-8'))

print("Edges to 8-Puzzle:")
print("="*60)
for e in kg['edges']:
    if '8-puzzle' in e['target'].lower():
        print(f"{e['source']:25s} -> {e['target']:20s} ({e['relation_type']:15s}, conf={e['confidence']:.2f})")

print("\n\nEdges from algorithms:")
print("="*60)
algo_edges = [e for e in kg['edges'] if e['source'].startswith('algo_') and e['relation_type'] in ['solves', 'solved_by', 'uses', 'applicable_to']]
for e in algo_edges[:20]:
    print(f"{e['source']:25s} -> {e['target']:20s} ({e['relation_type']:15s})")
