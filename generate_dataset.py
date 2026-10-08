import random
import csv

TOTAL_RECORDS = 100000
data = []

for _ in range(TOTAL_RECORDS):

    condition = random.choice(["normal","load","damage"])

    if condition == "normal":
        # almost no vibration
        vibration = round(random.uniform(0.00,0.15),3)

    elif condition == "load":
        # car movement / mild vibration
        vibration = round(random.uniform(0.2,1.5),3)

    else:  # damage
        # strong taps / structural events
        vibration = round(random.uniform(1.5,8.0),3)

    data.append([vibration,condition])

# save CSV
with open("dataset.csv","w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vibration","label"])
    writer.writerows(data)

print("✅ ESP32-style dataset (100k) created!")