from math import log

def calculate_shannon_entropy(data_set):
    num_entries = len(data_set)
    label_counts = {}

    for feature_vector in data_set:
        current_label = feature_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
            label_counts[current_label] += 1
    
    shannon_entropy = 0

    for key in label_counts:
        prob = float(label_counts[key]/ num_entries)
        shannon_entropy -= prob * log(prob, 2)
    
    return shannon_entropy

def create_dataset():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 1, 'no'], [0, 1, 'no'], \
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

if __name__ == '__main__':
    my_data, labels = create_dataset()
    print(my_data)
    print(calculate_shannon_entropy(my_data))
