import os
import shutil
import subprocess

# Define the base directory where checkpoint folders are stored
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "checkpoints"))
LOG_FILENAME = "successful_checkpoints.log"

def get_model_folders():
    """Get all folders that contain model checkpoints."""
    return [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f)) and f.startswith("2025")]

def get_checkpoints(model_folder):
    """Get all checkpoint models inside a model folder."""
    model_path = os.path.join(BASE_DIR, model_folder)
    return sorted([f for f in os.listdir(model_path) if f.startswith("model_")])

def copy_and_test_model(model_folder, checkpoint_name):
    """Copy the model to 'checkpoints/' and run the test command."""
    model_source = os.path.join(BASE_DIR, model_folder, checkpoint_name)
    model_dest = os.path.join(BASE_DIR, "model")  # Copy to 'checkpoints/', not inside a folder

    # Copy model
    shutil.copy(model_source, model_dest)
    print(f"Testing model: {checkpoint_name} from {model_folder}")

    # Run the test
    command = [
        "python", "run_robosuite.py",
        "--env=MyBlockStackingEnvRM1",
        "--num_timesteps=300",
        "--alg=dhrm",
        "--play"
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=os.getcwd())

    # Print stdout and stderr for debugging
    print(result.stdout)
    print(result.stderr)

    # Check if the model succeeded
    if "SUCCESS: self.env.current_u_id == -1" in result.stdout:
        log_success(model_folder, checkpoint_name)

def log_success(model_folder, checkpoint_name):
    """Log the successful model checkpoint inside its respective folder."""
    log_path = os.path.join(BASE_DIR, model_folder, LOG_FILENAME)
    with open(log_path, "a") as log_file:  # "a" means append mode
        log_file.write(f"{checkpoint_name}\n")
    print(f"Checkpoint {checkpoint_name} reached termination condition (-1) and was logged.")

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: Checkpoints folder '{BASE_DIR}' does not exist.")
        return

    model_folders = get_model_folders()
    if not model_folders:
        print("No model folders found in the 'checkpoints' directory.")
        return

    for model_folder in model_folders:
        checkpoints = get_checkpoints(model_folder)
        for checkpoint in checkpoints:
            copy_and_test_model(model_folder, checkpoint)

if __name__ == "__main__":
    main()
