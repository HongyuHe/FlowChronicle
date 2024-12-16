from model import Model

import sys
import json


def interpret_model(model: Model, name: str):
    patterns = {}
    for i, p in enumerate(model.pattern_set):
        patterns[i] = p.get_real_value_repr(model.cover.dataset)
    
    #* Save to json file
    with open(f'../results/interpreted_{name}.json', 'w') as f:
        json.dump(patterns, f, indent=4)
		# print(p.get_real_value_repr(model.cover.dataset))
		# print("\n")

if __name__ == "__main__":
    file = sys.argv[1]
    model = Model.load_model(file)
    name = file.split('/')[-1].split('.')[0]
    interpret_model(model, name)