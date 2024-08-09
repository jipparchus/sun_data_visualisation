# Anaconda environment prep

- Please set up the environment to run ``` main.py ``` by executing the below. The name of your environment, "foo", can be set to anything you like.

```bash
conda env create -n foo -f requirement.yml
```
```bash
conda activate foo
```

# Run the main code
- Please download ``` main.py ``` and place it in an arbitrary directory.

- Move to the directory and try:
    ```bash
    python main.py
    ```

- ``` main.py ``` will automatically download necessary dataset, ``` /aia_data ```, to the same directory as ``` main.py ```. If the download fails, please download ``` /aia_data ``` manually.
