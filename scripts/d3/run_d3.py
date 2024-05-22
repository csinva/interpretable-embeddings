import os

from d3 import TASKS_D3

if __name__ == "__main__":
    all_tasks = TASKS_D3.keys()
    all_tasks = list(sorted(all_tasks))
    for task in all_tasks:
        command = f"python run_eval.py {task}"
        print(command)
        os.system(command)