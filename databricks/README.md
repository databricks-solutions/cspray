### Downloading and processing data from CELLxGENE

The following notebooks will download data from CELLxGENE and process the raw data with cspray. Finally, we will perform reference-free cell annotation at the cluster level using majority voting over LLM calls, and merge these cell labels back into the cell-level data.

What you need to do:
 - Rename `config.yaml.example` to `config.yaml` and set the catalog and schema (ensure those catalog and schema exist).
 - Set up a compute cluster with some number of workers, let's say 4.
 - Run the 00 and 01 notebooks with this compute.
   - In the 01 notebook, you will want to change the WORKER_RAM : int = 32 to the RAM of your workers if it differs from 32GB.
 - Use serverless compute to run the 02 notebook.

Now, you'll download some data from CELLxGENE, process all the files simultaneously with cspray, and perform cell type labeling on the gold dataset.

The gold dataset is perfect for building dashboards and apps on top of it—allowing searching over cell types and other metadata to find samples of interest. Later, users can get the pre-processed data from the silver tables and perform aggregated analyses.
