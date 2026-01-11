import pandas as pd
import random
import math

NUM_ROWS = 3000

WEIGHTS = {
    "car": 2.0,
    "bike": 0.5,
    "bus": 3.5,
    "truck": 4.0
}

rows = []

for _ in range(NUM_ROWS):

    # Foreign traffic style
    cars = random.randint(5, 30)
    bikes = random.randint(0, 6)
    buses = random.randint(0, 5)
    trucks = random.randint(0, 8)

    total = cars + bikes + buses + trucks

    # Weighted score
    weighted_score = (
        cars   * WEIGHTS["car"] +
        bikes  * WEIGHTS["bike"] +
        buses  * WEIGHTS["bus"] +
        trucks * WEIGHTS["truck"]
    )

    # -------- GREEN TIME LOGIC --------
    base_time = 8 + math.sqrt(weighted_score) * 6

    # Random noise for diversity
    noise = random.uniform(-8, 10)

    green_time = int(base_time + noise)

    # Safety bounds
    green_time = max(10, min(green_time, 120))

    rows.append([
        cars,
        bikes,
        buses,
        trucks,
        total,
        green_time
    ])

df = pd.DataFrame(
    rows,
    columns=["cars", "bikes", "buses", "trucks", "total", "green_time"]
)

df.to_csv("traffic_green_time_dataset.csv", index=False)

print("✅ High-variance dataset generated")
