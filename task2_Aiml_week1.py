import pandas as pd

# Dictionary to store counts
log_summary = {}

# Open log file
with open("application.log", "r") as file:

    # Read each line
    for line in file:

        # Remove newline
        line = line.strip()

        # Split line into parts
        parts = line.split()

        # Extract fields
        level = parts[2]
        module = parts[3]

        # Process only ERROR and WARNING
        if level in ["ERROR", "WARNING"]:

            key = (module, level)

            if key in log_summary:
                log_summary[key] += 1
            else:
                log_summary[key] = 1

# Convert dictionary to DataFrame
report = pd.DataFrame(
    [(module, level, count)
     for (module, level), count in log_summary.items()],
    columns=["Module", "Log_Level", "Frequency"]
)

# Save CSV
report.to_csv("log_report.csv", index=False)

print(report)