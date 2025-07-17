import numpy as np
import csv

class LabeledArray:
    def __init__(self):
        self.data = None
        self.labels = None

    def new(self, labels, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy ndarray")
        if not isinstance(labels, list) or not all(isinstance(l, list) for l in labels):
            raise TypeError("labels must be a list of lists")
        if any(':' in label_list for label_list in labels):
            raise ValueError("Label ':' is reserved and cannot be used as a label.")
        if data.ndim != len(labels):
            raise ValueError(f"Data has {data.ndim} dimensions but {len(labels)} label lists provided")
        for i, (label_list, dim_size) in enumerate(zip(labels, data.shape)):
            if len(label_list) != dim_size:
                raise ValueError(f"Dimension {i} has size {dim_size}, but {len(label_list)} labels provided")
        self.data = data
        self.labels = labels

    def shape(self):
        return self.data.shape if self.data is not None else ()

    def ndim(self):
        return self.data.ndim if self.data is not None else 0

    def is_empty(self):
        return self.data is None

    def get(self, *label_path):
        if self.data is None or self.labels is None:
            raise ValueError("LabeledArray is not initialized.")
        if len(label_path) != len(self.labels):
            raise ValueError(f"Expected {len(self.labels)} labels, got {len(label_path)}")
        indices = []
        for i, label in enumerate(label_path):
            try:
                idx = self.labels[i].index(label)
            except ValueError:
                raise KeyError(f"Label '{label}' not found in dimension {i}")
            indices.append(idx)
        return self.data[tuple(indices)]

    def gets(self, *label_lists):
        if self.data is None or self.labels is None:
            raise ValueError("LabeledArray is not initialized.")
        if len(label_lists) != len(self.labels):
            raise ValueError(f"Expected {len(self.labels)} label lists, got {len(label_lists)}")
        index_slices = []
        for i, label_list in enumerate(label_lists):
            if label_list is None or label_list == [':']:
                index_slices.append(slice(None))
            else:
                indices = []
                for label in label_list:
                    try:
                        idx = self.labels[i].index(label)
                    except ValueError:
                        raise KeyError(f"Label '{label}' not found in dimension {i}")
                    indices.append(idx)
                index_slices.append(indices)
        return self.data[np.ix_(*index_slices)]

    def set(self, *label_path_and_value):
        if self.data is None or self.labels is None:
            raise ValueError("LabeledArray is not initialized.")
        if len(label_path_and_value) != len(self.labels) + 1:
            raise ValueError("Must provide one label per dimension followed by a value.")
        *label_path, value = label_path_and_value
        indices = []
        for i, label in enumerate(label_path):
            try:
                idx = self.labels[i].index(label)
            except ValueError:
                raise KeyError(f"Label '{label}' not found in dimension {i}")
            indices.append(idx)
        self.data[tuple(indices)] = value

    def sets(self, *label_lists_and_value):
        if self.data is None or self.labels is None:
            raise ValueError("LabeledArray is not initialized.")
        if len(label_lists_and_value) != len(self.labels) + 1:
            raise ValueError("Must provide one label list per dimension followed by values.")
        *label_lists, values = label_lists_and_value
        index_slices = []
        for i, label_list in enumerate(label_lists):
            if label_list is None or label_list == [':']:
                index_slices.append(slice(None))
            else:
                indices = []
                for label in label_list:
                    try:
                        idx = self.labels[i].index(label)
                    except ValueError:
                        raise KeyError(f"Label '{label}' not found in dimension {i}")
                    indices.append(idx)
                index_slices.append(indices)
        self.data[np.ix_(*index_slices)] = values

    def as_numpy(self):
        return self.data.copy() if self.data is not None else np.array([])

    def to_csv(self, filepath, sep='__'):
        if self.data is None or self.labels is None:
            raise ValueError("No data to export. Call .new(...) first.")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['index']
            for i1 in range(self.data.shape[1]):
                for i2 in range(self.data.shape[2]):
                    label = sep.join([self.labels[1][i1], self.labels[2][i2]])
                    header.append(label)
            writer.writerow(header)
            for i0 in range(self.data.shape[0]):
                row = [self.labels[0][i0]]
                flat = self.data[i0].flatten()
                row.extend(flat.tolist())
                writer.writerow(row)

    @classmethod
    def from_csv(cls, filepath, sep='__'):
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[0][1:]  # skip 'index'
        dim0_labels = [row[0] for row in rows[1:]]
        raw_data = [list(map(float, row[1:])) for row in rows[1:]]
        keys = [col.split(sep) for col in header]
        dim1_labels = sorted(set(k[0] for k in keys))
        dim2_labels = sorted(set(k[1] for k in keys))
        shape = (len(dim0_labels), len(dim1_labels), len(dim2_labels))
        data = np.array(raw_data).reshape(shape)
        labels = [dim0_labels, dim1_labels, dim2_labels]
        obj = cls()
        obj.new(labels, data)
        return obj
