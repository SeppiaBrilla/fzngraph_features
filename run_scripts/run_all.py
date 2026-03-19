import subprocess
import concurrent.futures
def run_command(command):
    """Execute a single command and return its result"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return {
            'command': command,
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.CalledProcessError as e:
        return {
            'command': command,
            'success': False,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'returncode': e.returncode
        }

def run_commands_parallel(commands):
    """Run multiple commands in parallel using ThreadPoolExecutor"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as executor:
        # Submit all commands
        futures = [executor.submit(run_command, cmd) for cmd in commands]
        # Wait for all to complete and collect results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

if __name__ == "__main__":
    # Example commands - replace these with your actual commands
    commands = [
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m rnd-forest --cv-fold 0 --result results3/as_wlc-1-forest-0.json --reduction -1 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m rnd-forest --cv-fold 1 --result results3/as_wlc-1-forest-1.json --reduction -1 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m rnd-forest --cv-fold 2 --result results3/as_wlc-1-forest-2.json --reduction -1 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m rnd-forest --cv-fold 3 --result results3/as_wlc-1-forest-3.json --reduction -1 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m rnd-forest --cv-fold 4 --result results3/as_wlc-1-forest-4.json --reduction -1 -t 60",

        "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 0 --result results3/as_wlc-2-95-forest-0.json --reduction 95 -t 60",
        "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 1 --result results3/as_wlc-2-95-forest-1.json --reduction 95 -t 60",
        "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 2 --result results3/as_wlc-2-95-forest-2.json --reduction 95 -t 60",
        "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 3 --result results3/as_wlc-2-95-forest-3.json --reduction 95 -t 60",
        "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 4 --result results3/as_wlc-2-95-forest-4.json --reduction 95 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 0 --result results3/as_wlc-2-120-forest-0.json --reduction 120 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 1 --result results3/as_wlc-2-120-forest-1.json --reduction 120 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 2 --result results3/as_wlc-2-120-forest-2.json --reduction 120 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 3 --result results3/as_wlc-2-120-forest-3.json --reduction 120 -t 60",
        # "python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m rnd-forest --cv-fold 4 --result results3/as_wlc-2-120-forest-4.json --reduction 120 -t 60",
    ]
    print("Starting commands in parallel...")
    results = run_commands_parallel(commands)
    print("\nAll commands completed!\n")
    print("=" * 50)
    # Display results
    for result in results:
        print(f"\nCommand: {result['command']}")
        print(f"Success: {result['success']}")
        print(f"Return Code: {result['returncode']}")
        if result['stdout']:
            print(f"Output: {result['stdout'].strip()}")
        if result['stderr']:
            print(f"Error: {result['stderr'].strip()}")
        print("-" * 50)
