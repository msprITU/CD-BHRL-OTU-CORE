import subprocess

# Define your items and the range of indices you want to iterate through
items = ["ballet", "bicycle", "group1", "group2", "group3", "kitesurfing", "longboard", "person2", "person4", "person5",
"person7", "person14", "person17", "person19", "person20", "rollerman", "skiing", "sup", "tightrope", "warmup", "wingsuit"]
indices = range(1, 6)  # For example, from 1 to 10

# sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
# wingsuit 1
# sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
# wingsuit 2
# sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
# wingsuit 3
# sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
# wingsuit 4
# sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
# wingsuit 5

for item_idx, item in enumerate(items):
    if not item_idx == 0:
        continue
    for i in indices:
        # Construct the sbatch command with the item and i as arguments
        cmd = ["sbatch", "bhrl.sh", item, str(i)]
        
        # Execute the command
        subprocess.run(cmd)
        print(item, i)
