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

def split_dataset(data_set, axis, value):
    ret_data_set = []

    for feature_vector in data_set:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1: ])
            ret_data_set.append(reduced_feature_vector)
    return ret_data_set

def choose_best_feture_to_split(data_set):
    num_features = len(data_set[0] - 1)
    base_entropy = calculate_shannon_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        feature_list = [example[i] for example in data_set]
        unique_vals = set(feature_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_dataset(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


if __name__ == '__main__':
    my_data, labels = create_dataset()
    print(my_data)
    print(calculate_shannon_entropy(my_data))
    print(split_dataset(my_data, 0, 1))
    print(split_dataset(my_data, 0, 0))