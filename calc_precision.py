file = open(
    '/Users/haraldott/Development/thesis/anomaly_detection_main/data/openstack/utah/raw/openstack_18k_anomalies')
lines = file.readlines()
outliers = open('/Users/haraldott/Downloads/150_epochs/outliers_values')
outliers = outliers.readlines()

instances_containing_anomalies = [
    "544fd51c-4edc-4780-baae-ba1d80a0acfc",
    "ae651dff-c7ad-43d6-ac96-bbcd820ccca8",
    "a445709b-6ad0-40ec-8860-bec60b6ca0c2",
    "1643649d-2f42-4303-bfcd-7798baec19f9"
]
anomaly_idx = []
for i, line in enumerate(lines):
    if any(substring in line for substring in instances_containing_anomalies):
        anomaly_idx.append(i + 1)

detected_anomalies = []
for line in outliers:
    x = line.split(',')
    detected_anomalies.append(int(x[0]))

tp = 0
fp = 0

for el in anomaly_idx:
    if el in detected_anomalies:
        tp += 1
    else:
        fp += 1