import subprocess


def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(["python", script_name], check=True)
        print(f"{script_name} completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    run_script("train.py")

    run_script("predict.py")
