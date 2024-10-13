To debug your file with the specified arguments in Visual Studio Code, follow these steps:

1. **Open Visual Studio Code** and load your workspace or folder containing `run.py`.

2. **Open the Debug Tab**:
   - On the left sidebar, click on the Debug icon (a play button with a bug).
   
3. **Create a launch configuration**:
   - If this is your first time setting up a debug configuration, click on the "create a `launch.json` file" link.
   - In the dropdown that appears, choose **Python**.
   
4. **Edit the `launch.json` file**:
   - A `launch.json` file will be generated. You need to modify it to include your script's arguments.
   - Update the file to look something like this:

   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: run.py",
               "type": "python",
               "request": "launch",
               "program": "${workspaceFolder}/run.py",  // Make sure this path points to your 'run.py'
               "args": ["--env=Office-v0", "--num_timesteps=1e5", "--gamma=0.9", "--alg=qlearning"],
               "console": "integratedTerminal"  // This ensures the output is shown in the VSCode terminal
           }
       ]
   }
   ```

   - Ensure the `program` field points to the correct path of your `run.py` file. The `args` field includes the arguments you provided: `--env=Office-v0`, `--num_timesteps=1e5`, `--gamma=0.9`, `--alg=qlearning`.

5. **Start Debugging**:
   - Go back to the Debug tab and select your newly created configuration (`Python: run.py`).
   - Press the green play button to start the debugging process.

Now, Visual Studio Code will run your Python file with the specified arguments in debug mode. You can set breakpoints, step through your code, and inspect variables as needed.